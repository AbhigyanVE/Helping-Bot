import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response

from modules.llm_handler import get_llm_response, print_llm_startup_status

load_dotenv()

# On Windows, the selector loop avoids noisy Proactor connection-reset callbacks
# that can appear when a browser refreshes/closes a WebSocket (WinError 10054).
if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

BASE_DIR = Path(__file__).resolve().parent
APP_VERSION = "0.5.0"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print_llm_startup_status()
    yield


app = FastAPI(title="VE Interview Copilot", version=APP_VERSION, lifespan=lifespan)


@app.get("/")
async def get() -> HTMLResponse:
    return HTMLResponse((BASE_DIR / "index.html").read_text(encoding="utf-8"))


@app.head("/")
async def head_root() -> Response:
    # Render may probe the root route with HEAD during deployment/health checks.
    return Response(status_code=200)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": APP_VERSION})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    print("[WebSocket] Client connected.")

    queue: asyncio.Queue[dict | None] = asyncio.Queue()
    meeting_transcript = ""

    async def response_worker() -> None:
        nonlocal meeting_transcript

        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break

            utterance_id = str(item.get("id") or "")
            user_text = str(item.get("text") or "").strip()
            received_at = float(item.get("received_at") or time.perf_counter())
            source_mode = str(item.get("source_mode") or "computer")
            source_label = {
                "computer": "Client Audio",
                "mic": "Microphone",
                "both": "Both/Mixed",
            }.get(source_mode, source_mode or "Unknown")

            try:
                if not user_text:
                    continue

                print(f"[{source_label}] id={utterance_id}: {user_text}")

                queue_wait_ms = (time.perf_counter() - received_at) * 1000
                generation_started_at = time.perf_counter()
                result = await get_llm_response(
                    latest_utterance=user_text,
                    meeting_transcript=meeting_transcript,
                )
                generation_ms = (time.perf_counter() - generation_started_at) * 1000

                print(
                    f"[Copilot] id={utterance_id} | ok={result.ok}"
                    f" | code={result.error_code or 'none'}"
                    f" | queue={queue_wait_ms:.0f}ms"
                    f" | generation={generation_ms:.0f}ms: {result.text}"
                )

                # Store both sides so later answers can follow the conversation.
                meeting_transcript += (
                    f"\nSpeaker [{source_label}]: {user_text}\n"
                    f"Copilot: {result.text}\n"
                )
                if len(meeting_transcript) > 12000:
                    meeting_transcript = meeting_transcript[-12000:]

                await websocket.send_json(
                    {
                        "type": "bot_response",
                        "id": utterance_id,
                        "text": result.text,
                        "ok": result.ok,
                        "error_code": result.error_code,
                        "request_id": result.request_id,
                        "queue_wait_ms": round(queue_wait_ms, 1),
                        "generation_ms": round(generation_ms, 1),
                    }
                )

            except WebSocketDisconnect:
                break
            except Exception as exc:
                print(f"[Worker Error] {type(exc).__name__}: {exc}")
                with suppress(Exception):
                    await websocket.send_json(
                        {
                            "type": "bot_response",
                            "id": utterance_id,
                            "text": "A local server error occurred. Check the terminal.",
                            "ok": False,
                            "error_code": "local_worker_error",
                        }
                    )
            finally:
                queue.task_done()

    worker = asyncio.create_task(response_worker())

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") != "transcript":
                continue

            user_text = str(data.get("text") or "").strip()
            if not user_text:
                continue

            # Receiving is intentionally decoupled from LLM processing. That way
            # the browser can keep sending finalized speech while one answer is
            # still being generated; the worker preserves conversational order.
            await queue.put(
                {
                    "id": str(data.get("id") or ""),
                    "text": user_text,
                    "received_at": time.perf_counter(),
                    "speech_to_text_ms": data.get("speech_to_text_ms"),
                    "source_mode": str(data.get("source_mode") or "computer"),
                }
            )

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")
    except ConnectionResetError as exc:
        print(f"[WebSocket] Connection reset by browser: {exc}")
    except Exception as exc:
        print(f"[WebSocket] Unexpected error: {type(exc).__name__}: {exc}")
    finally:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker


if __name__ == "__main__":
    # Local development still uses port 8000. Render provides PORT (normally
    # 10000) and requires the server to listen on 0.0.0.0.
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
