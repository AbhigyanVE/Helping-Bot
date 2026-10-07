"""OpenAI response layer with useful terminal diagnostics for live meetings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI

from modules.bot_context import BOT_PERSONA, build_user_input

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna").strip()
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "12"))
MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "0"))
MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "260"))


@dataclass(slots=True)
class LLMResult:
    text: str
    ok: bool = True
    error_code: Optional[str] = None
    request_id: Optional[str] = None


client = (
    AsyncOpenAI(
        api_key=API_KEY,
        timeout=TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )
    if API_KEY
    else None
)


def print_llm_startup_status() -> None:
    """Print non-secret configuration so startup problems are obvious."""
    key_state = "configured" if API_KEY else "MISSING"
    sdk_version = getattr(openai, "__version__", "unknown")
    ssl_cert_file = os.getenv("SSL_CERT_FILE", "").strip()
    print(
        "[LLM Config] "
        f"model={MODEL} | api_key={key_state} | "
        f"timeout={TIMEOUT_SECONDS:.1f}s | retries={MAX_RETRIES} | "
        f"openai_sdk={sdk_version}"
    )
    if ssl_cert_file:
        print(f"[TLS Config] SSL_CERT_FILE={ssl_cert_file}")
    if not API_KEY:
        print(
            "[LLM Config] OPENAI_API_KEY is missing. Add it to .env before "
            "starting a meeting."
        )


def _root_cause_text(exc: BaseException) -> str:
    cause = exc.__cause__
    if cause is None:
        return "No lower-level transport error was provided by the SDK."

    parts: list[str] = []
    current: Optional[BaseException] = cause
    for _ in range(4):
        if current is None:
            break
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__
    return " -> ".join(parts)


def _connection_diagnosis(exc: BaseException) -> tuple[str, str]:
    details = _root_cause_text(exc)
    lowered = details.lower()

    if any(token in lowered for token in ("getaddrinfo", "name resolution", "nodename")):
        return (
            "DNS resolution failure",
            "Check internet/DNS settings and whether api.openai.com resolves from this machine.",
        )
    if any(token in lowered for token in ("certificate", "ssl", "tls")):
        if "self-signed certificate in certificate chain" in lowered:
            return (
                "TLS certificate trust failure (self-signed certificate in chain)",
                "Python does not trust a certificate inserted into the HTTPS chain. This is usually corporate proxy/VPN/antivirus HTTPS inspection, not an API-key or timeout problem. Install/export the trusted root CA or set SSL_CERT_FILE to its PEM bundle. Do not disable TLS verification.",
            )
        return (
            "TLS/SSL connection failure",
            "Check the machine trust store, corporate certificates, antivirus HTTPS inspection, VPN/proxy settings, and system time.",
        )
    if "proxy" in lowered:
        return (
            "Proxy connection failure",
            "Check HTTP(S)_PROXY settings or your corporate proxy configuration.",
        )
    if any(token in lowered for token in ("connection refused", "actively refused")):
        return (
            "Connection refused",
            "A firewall, proxy, VPN, or network rule may be blocking the outbound API connection.",
        )
    if any(token in lowered for token in ("connection reset", "forcibly closed", "remoteprotocolerror")):
        return (
            "Connection was reset",
            "The network path, VPN/proxy, or remote connection closed unexpectedly. Retry after checking connectivity.",
        )
    if any(token in lowered for token in ("network is unreachable", "no route to host")):
        return (
            "Network unreachable",
            "Check internet connectivity, VPN routing, firewall rules, and proxy settings.",
        )

    return (
        "Network/transport connection failure",
        "Check internet access, DNS, proxy/VPN, firewall rules, and SSL certificates.",
    )


def _log_error(
    *,
    kind: str,
    diagnosis: str,
    action: str,
    exc: BaseException,
    request_id: Optional[str] = None,
) -> None:
    print("\n" + "=" * 78)
    print(f"[LLM ERROR] {kind}")
    print(f"Diagnosis : {diagnosis}")
    print(f"Action    : {action}")
    print(f"Model     : {MODEL}")
    print(f"Timeout   : {TIMEOUT_SECONDS:.1f}s")
    if request_id:
        print(f"Request ID: {request_id}")
    print(f"SDK error : {type(exc).__name__}: {exc}")
    if isinstance(exc, openai.APIConnectionError):
        print(f"Transport : {_root_cause_text(exc)}")
    print("=" * 78 + "\n")


async def get_llm_response(
    latest_utterance: str,
    meeting_transcript: str = "",
) -> LLMResult:
    """Return a short answer while classifying API failures precisely."""
    latest_utterance = (latest_utterance or "").strip()

    if not latest_utterance:
        return LLMResult(
            text="I did not catch a complete question. Please say it again.",
            ok=False,
            error_code="empty_utterance",
        )

    if client is None:
        message = "OPENAI_API_KEY is not configured in the .env file."
        _log_error(
            kind="ConfigurationError",
            diagnosis="API key is missing",
            action="Add OPENAI_API_KEY to .env and restart the server.",
            exc=RuntimeError(message),
        )
        return LLMResult(
            text="The technical assistant is not configured yet. Please check the server terminal.",
            ok=False,
            error_code="missing_api_key",
        )

    try:
        response = await client.responses.create(
            model=MODEL,
            instructions=BOT_PERSONA,
            input=build_user_input(
                latest_utterance=latest_utterance,
                meeting_transcript=meeting_transcript,
            ),
            max_output_tokens=MAX_OUTPUT_TOKENS,
            store=False,
        )

        answer = (response.output_text or "").strip()
        request_id = getattr(response, "_request_id", None)

        if not answer:
            return LLMResult(
                text="I could not form a complete answer. Please repeat the question.",
                ok=False,
                error_code="empty_response",
                request_id=request_id,
            )

        if request_id:
            print(f"[LLM OK] request_id={request_id}")

        return LLMResult(text=answer, request_id=request_id)

    except openai.APITimeoutError as exc:
        _log_error(
            kind="APITimeoutError",
            diagnosis=f"The OpenAI request exceeded the {TIMEOUT_SECONDS:.1f}s timeout.",
            action="Check network latency or increase LLM_TIMEOUT_SECONDS slightly.",
            exc=exc,
        )
        return LLMResult(
            text="The technical assistant timed out. Please try that question once more.",
            ok=False,
            error_code="timeout",
        )

    except openai.AuthenticationError as exc:
        request_id = getattr(exc, "request_id", None)
        _log_error(
            kind="AuthenticationError (401)",
            diagnosis="The API key/token is invalid, expired, revoked, or otherwise rejected.",
            action="Verify OPENAI_API_KEY and replace it with an active key if necessary.",
            exc=exc,
            request_id=request_id,
        )
        return LLMResult(
            text="The technical assistant could not authenticate. Please check the server terminal.",
            ok=False,
            error_code="authentication",
            request_id=request_id,
        )

    except openai.PermissionDeniedError as exc:
        request_id = getattr(exc, "request_id", None)
        _log_error(
            kind="PermissionDeniedError (403)",
            diagnosis="The API credentials do not have permission for this request/model/project.",
            action="Check project permissions, model access, and the API key's project.",
            exc=exc,
            request_id=request_id,
        )
        return LLMResult(
            text="The API credentials do not have permission for this request. Check the terminal.",
            ok=False,
            error_code="permission_denied",
            request_id=request_id,
        )

    except openai.RateLimitError as exc:
        request_id = getattr(exc, "request_id", None)
        _log_error(
            kind="RateLimitError (429)",
            diagnosis="The API returned a rate-limit/quota response.",
            action="Check request rate, project quota/billing, and retry after a short delay.",
            exc=exc,
            request_id=request_id,
        )
        return LLMResult(
            text="The technical assistant is temporarily rate-limited. Please try again shortly.",
            ok=False,
            error_code="rate_limit",
            request_id=request_id,
        )

    except openai.BadRequestError as exc:
        request_id = getattr(exc, "request_id", None)
        _log_error(
            kind="BadRequestError (400)",
            diagnosis="The API rejected the request payload or one of its parameters.",
            action="Check OPENAI_MODEL and request options shown in modules/llm_handler.py.",
            exc=exc,
            request_id=request_id,
        )
        return LLMResult(
            text="The assistant request was rejected by the API. Check the server terminal for details.",
            ok=False,
            error_code="bad_request",
            request_id=request_id,
        )

    except openai.NotFoundError as exc:
        request_id = getattr(exc, "request_id", None)
        _log_error(
            kind="NotFoundError (404)",
            diagnosis="The configured model/resource was not found or is not available to this project.",
            action=f"Verify OPENAI_MODEL={MODEL!r} and confirm that your API project can use it.",
            exc=exc,
            request_id=request_id,
        )
        return LLMResult(
            text="The configured AI model is not available to this API project. Check the terminal.",
            ok=False,
            error_code="model_not_found",
            request_id=request_id,
        )

    except openai.InternalServerError as exc:
        request_id = getattr(exc, "request_id", None)
        _log_error(
            kind="InternalServerError (5xx)",
            diagnosis="The upstream API returned a server-side error.",
            action="Retry shortly. If it persists, keep the request ID for troubleshooting.",
            exc=exc,
            request_id=request_id,
        )
        return LLMResult(
            text="The technical assistant service returned a temporary server error. Please retry.",
            ok=False,
            error_code="server_error",
            request_id=request_id,
        )

    except openai.APIConnectionError as exc:
        diagnosis, action = _connection_diagnosis(exc)
        _log_error(
            kind="APIConnectionError",
            diagnosis=diagnosis,
            action=action,
            exc=exc,
        )
        return LLMResult(
            text="I could not reach the technical assistant. The terminal now shows the connection cause.",
            ok=False,
            error_code="connection",
        )

    except openai.APIStatusError as exc:
        request_id = getattr(exc, "request_id", None)
        _log_error(
            kind=f"APIStatusError ({getattr(exc, 'status_code', 'unknown')})",
            diagnosis="The API returned a non-success HTTP status.",
            action="Use the status code and request ID above to identify the failing API condition.",
            exc=exc,
            request_id=request_id,
        )
        return LLMResult(
            text="The technical assistant returned an API error. Check the terminal for the exact status.",
            ok=False,
            error_code="api_status",
            request_id=request_id,
        )

    except Exception as exc:
        _log_error(
            kind="UnexpectedError",
            diagnosis="An unexpected error occurred in the local LLM handler.",
            action="Inspect the exception above and verify package versions and configuration.",
            exc=exc,
        )
        return LLMResult(
            text="The technical assistant hit an unexpected local error. Check the server terminal.",
            ok=False,
            error_code="unexpected",
        )
