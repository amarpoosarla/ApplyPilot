"""
Unified LLM client for ApplyPilot.

Auto-detects providers from environment:
  GEMINI_API_KEY  -> Google Gemini (default: gemini-2.0-flash)  [scoring / fast tasks]
  NVIDIA_API_KEY  -> NVIDIA NIM / Llama   (default: meta/llama-3.3-70b-instruct) [writing tasks]
  OPENAI_API_KEY  -> OpenAI (default: gpt-4o-mini)
  LLM_URL         -> Local llama.cpp / Ollama compatible endpoint

Task routing via get_client(role):
  "scoring"  -> fast/cheap model (Gemini Flash) for scoring, enrichment, extraction, judge
  "writing"  -> quality model (Llama 70B via NVIDIA NIM) for tailoring and cover letters
  default    -> same as "scoring"

If only one provider is configured, both roles use it.
The writing client automatically falls back to the scoring client on rate limits.

Model overrides:
  SCORING_MODEL  -> override model for scoring tasks
  WRITING_MODEL  -> override model for writing tasks
  LLM_MODEL      -> override model for any single-provider setup
"""

import logging
import os
import threading
import time

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

_NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
_GEMINI_COMPAT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_GEMINI_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _detect_scoring_provider() -> tuple[str, str, str]:
    """Return (base_url, model, api_key) for the fast/scoring provider.

    Priority: NVIDIA NIM > OpenAI > local > Gemini (last resort).
    NVIDIA is preferred because it has higher parallel request limits.
    """
    nvidia_key = os.environ.get("NVIDIA_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    local_url = os.environ.get("LLM_URL", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    model_override = os.environ.get("SCORING_MODEL") or os.environ.get("LLM_MODEL", "")

    if nvidia_key and not local_url:
        return (
            _NVIDIA_BASE,
            model_override or "meta/llama-3.1-8b-instruct",
            nvidia_key,
        )

    if openai_key and not local_url:
        return (
            "https://api.openai.com/v1",
            model_override or "gpt-4o-mini",
            openai_key,
        )

    if local_url:
        return (
            local_url.rstrip("/"),
            model_override or "local-model",
            os.environ.get("LLM_API_KEY", ""),
        )

    if gemini_key:
        return (
            _GEMINI_COMPAT_BASE,
            model_override or "gemini-2.0-flash",
            gemini_key,
        )

    raise RuntimeError(
        "No LLM provider configured. "
        "Set NVIDIA_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, or LLM_URL in your environment."
    )


def _detect_writing_provider() -> tuple[str, str, str]:
    """Return (base_url, model, api_key) for the quality/writing provider.

    Priority: NVIDIA NIM > OpenAI > local > Gemini (last resort).
    """
    nvidia_key = os.environ.get("NVIDIA_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    local_url = os.environ.get("LLM_URL", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    model_override = os.environ.get("WRITING_MODEL") or os.environ.get("LLM_MODEL", "")

    if nvidia_key and not local_url:
        return (
            _NVIDIA_BASE,
            model_override or "meta/llama-3.3-70b-instruct",
            nvidia_key,
        )

    if openai_key and not local_url:
        return (
            "https://api.openai.com/v1",
            model_override or "gpt-4o-mini",
            openai_key,
        )

    if local_url:
        return (
            local_url.rstrip("/"),
            model_override or "local-model",
            os.environ.get("LLM_API_KEY", ""),
        )

    if gemini_key:
        return (
            _GEMINI_COMPAT_BASE,
            model_override or "gemini-2.0-flash",
            gemini_key,
        )

    raise RuntimeError(
        "No LLM provider configured. "
        "Set NVIDIA_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, or LLM_URL in your environment."
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_MAX_RETRIES = 5
_TIMEOUT = 120  # seconds

# Base wait on first 429/503 (doubles each retry, caps at 60s).
_RATE_LIMIT_BASE_WAIT = 10

# ---------------------------------------------------------------------------
# Global rate limiter for Gemini (15 RPM free tier)
# Serializes ALL Gemini calls across parallel workers so we never exceed the limit.
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Thread-safe token bucket that enforces a minimum interval between calls."""

    def __init__(self, calls_per_minute: int) -> None:
        self._interval = 60.0 / calls_per_minute
        self._lock = threading.Lock()
        self._last_call: float = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()


# One shared limiter for all Gemini clients (14 RPM = safe headroom under 15)
_gemini_rate_limiter = _RateLimiter(calls_per_minute=14)


class LLMClient:
    """Thin LLM client supporting OpenAI-compatible and native Gemini endpoints.

    For Gemini keys, starts on the OpenAI-compat layer. On a 403 (which
    happens with preview/experimental models not exposed via compat), it
    automatically switches to the native generateContent API and stays there
    for the lifetime of the process.
    """

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self._client = httpx.Client(timeout=_TIMEOUT)
        # True once we've confirmed the native Gemini API works for this model
        self._use_native_gemini: bool = False
        self._is_gemini: bool = base_url.startswith(_GEMINI_COMPAT_BASE)

    # -- Native Gemini API --------------------------------------------------

    def _chat_native_gemini(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the native Gemini generateContent API."""
        contents: list[dict] = []
        system_parts: list[dict] = []

        for msg in messages:
            role = msg["role"]
            text = msg.get("content", "")
            if role == "system":
                system_parts.append({"text": text})
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        _gemini_rate_limiter.acquire()
        url = f"{_GEMINI_NATIVE_BASE}/models/{self.model}:generateContent"
        resp = self._client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    # -- OpenAI-compat API --------------------------------------------------

    def _chat_compat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the OpenAI-compatible endpoint."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if self._is_gemini:
            _gemini_rate_limiter.acquire()

        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )

        # 403 on Gemini compat = model not available on compat layer.
        if resp.status_code == 403 and self._is_gemini:
            raise _GeminiCompatForbidden(resp)

        return self._handle_compat_response(resp)

    @staticmethod
    def _handle_compat_response(resp: httpx.Response) -> str:
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # -- public API ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request and return the assistant message text."""
        # Qwen3 optimization: prepend /no_think to skip chain-of-thought
        if "qwen" in self.model.lower() and messages:
            first = messages[0]
            if first.get("role") == "user" and not first["content"].startswith("/no_think"):
                messages = [{"role": first["role"], "content": f"/no_think\n{first['content']}"}] + messages[1:]

        for attempt in range(_MAX_RETRIES):
            try:
                if self._use_native_gemini:
                    return self._chat_native_gemini(messages, temperature, max_tokens)

                return self._chat_compat(messages, temperature, max_tokens)

            except _GeminiCompatForbidden:
                log.warning(
                    "Gemini compat endpoint returned 403 for model '%s'. "
                    "Switching to native generateContent API.",
                    self.model,
                )
                self._use_native_gemini = True
                try:
                    return self._chat_native_gemini(messages, temperature, max_tokens)
                except httpx.HTTPStatusError as native_exc:
                    raise RuntimeError(
                        f"Both Gemini endpoints failed. Compat: 403 Forbidden. "
                        f"Native: {native_exc.response.status_code} — "
                        f"{native_exc.response.text[:200]}"
                    ) from native_exc

            except httpx.HTTPStatusError as exc:
                resp = exc.response
                if resp.status_code in (429, 503) and attempt < _MAX_RETRIES - 1:
                    retry_after = (
                        resp.headers.get("Retry-After")
                        or resp.headers.get("X-RateLimit-Reset-Requests")
                    )
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except (ValueError, TypeError):
                            wait = _RATE_LIMIT_BASE_WAIT * (2 ** attempt)
                    else:
                        wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)

                    log.warning(
                        "LLM rate limited (HTTP %s) on %s. Waiting %ds before retry %d/%d.",
                        resp.status_code, self.model, wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise

            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)
                    log.warning(
                        "LLM request timed out on %s, retrying in %ds (attempt %d/%d)",
                        self.model, wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise

        raise RuntimeError("LLM request failed after all retries")

    def ask(self, prompt: str, **kwargs) -> str:
        """Convenience: single user prompt -> assistant response."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        self._client.close()


class _FallbackLLMClient:
    """Wraps a primary LLMClient with a fallback for rate-limit exhaustion.

    When the primary client raises after all its internal retries (typically
    a final 429/503), this wrapper transparently retries with the fallback
    client instead of propagating the error.
    """

    def __init__(self, primary: LLMClient, fallback: LLMClient) -> None:
        self.primary = primary
        self.fallback = fallback
        self.model = primary.model  # expose for logging

    def chat(self, messages: list[dict], **kwargs) -> str:
        try:
            return self.primary.chat(messages, **kwargs)
        except (httpx.HTTPStatusError, RuntimeError) as exc:
            # Fall back on rate-limit / server errors or exhausted retries.
            is_rate_limit = (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response.status_code in (429, 503)
            ) or (
                isinstance(exc, RuntimeError)
                and "after all retries" in str(exc)
            )
            if is_rate_limit:
                log.warning(
                    "Primary LLM (%s) rate limited — switching to fallback (%s) immediately.",
                    self.primary.model, self.fallback.model,
                )
                return self.fallback.chat(messages, **kwargs)
            raise

    def ask(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        self.primary.close()


class _GeminiCompatForbidden(Exception):
    """Sentinel: Gemini OpenAI-compat returned 403. Switch to native API."""
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"Gemini compat 403: {response.text[:200]}")


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_scoring_instance: LLMClient | None = None
_writing_instance: LLMClient | _FallbackLLMClient | None = None


def get_client(role: str = "scoring") -> LLMClient | _FallbackLLMClient:
    """Return the LLM client for the given role.

    Args:
        role: "scoring" (fast, many calls) or "writing" (quality, fewer calls).
              Any other value defaults to "scoring".

    Routing:
        - "scoring"  → Gemini Flash (or single configured provider)
        - "writing"  → NVIDIA NIM / Llama 70B, falling back to scoring client
                       on rate-limit exhaustion

    Both singletons are created lazily on first use.
    """
    global _scoring_instance, _writing_instance

    if role == "writing":
        if _writing_instance is None:
            _writing_instance = _build_writing_client()
        return _writing_instance

    # scoring / default
    if _scoring_instance is None:
        _scoring_instance = _build_scoring_client()
    return _scoring_instance


def _build_scoring_client() -> LLMClient | _FallbackLLMClient:
    """NVIDIA primary (high rate limits) → Gemini fallback."""
    w_url, w_model, w_key = _detect_writing_provider()
    s_url, s_model, s_key = _detect_scoring_provider()

    primary = LLMClient(w_url, w_model, w_key)
    log.info("Scoring LLM (primary): %s  model: %s", w_url, w_model)

    if w_url != s_url or w_model != s_model:
        fallback = LLMClient(s_url, s_model, s_key)
        log.info("Scoring LLM (fallback): %s  model: %s", s_url, s_model)
        return _FallbackLLMClient(primary=primary, fallback=fallback)

    return primary


def _build_writing_client() -> LLMClient | _FallbackLLMClient:
    """NVIDIA primary → Gemini fallback."""
    w_url, w_model, w_key = _detect_writing_provider()
    s_url, s_model, s_key = _detect_scoring_provider()

    primary = LLMClient(w_url, w_model, w_key)
    log.info("Writing LLM (primary): %s  model: %s", w_url, w_model)

    if w_url != s_url or w_model != s_model:
        fallback = LLMClient(s_url, s_model, s_key)
        log.info("Writing LLM (fallback): %s  model: %s", s_url, s_model)
        return _FallbackLLMClient(primary=primary, fallback=fallback)

    return primary
