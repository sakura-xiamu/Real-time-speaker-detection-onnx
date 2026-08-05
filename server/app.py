"""FastAPI 入口：HTTP PCM 上传 + SSE 状态推送。"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from server import config  # noqa: E402
from server.engine.providers import decode_onnx_error, provider_summary  # noqa: E402
from server.session import get_session_manager  # noqa: E402

logger = logging.getLogger(__name__)


class InitializeRequest(BaseModel):
    embedding_model_key: str = Field(default=config.DEFAULT_EMBEDDING_MODEL_KEY)
    use_gpu: bool = True
    verify_threshold: float | None = Field(default=None)


class StartRequest(BaseModel):
    window_sec: float = Field(default=config.DEFAULT_WINDOW_SEC)
    hop_sec: float = Field(default=config.DEFAULT_HOP_SEC)
    embedding_model_key: str | None = None
    use_gpu: bool | None = None
    fixed_speaker_num: int = 0
    verify_threshold: float = Field(default=config.VERIFICATION_THRESHOLD)
    slow_inference_threshold_ms: float = 1000.0
    slow_inference_max_consecutive: int = 5
    low_frequency_interval_sec: float = 60.0


class SessionIdBody(BaseModel):
    session_id: str | None = None


def _install_thread_excepthook() -> None:
    """后台线程未捕获异常时推送 SSE，避免静默失败（无法拦截原生崩溃）。"""

    def _hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type in (SystemExit, KeyboardInterrupt):
            return
        msg = (
            f"后台线程异常({args.thread.name if args.thread else '?'}): "
            f"{decode_onnx_error(args.exc_value or Exception('unknown'))}"
        )
        logger.exception(
            msg, exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
        )
        try:
            get_session_manager().report_error(msg)
        except Exception:  # noqa: BLE001
            logger.exception("推送线程异常到 SSE 失败")

    threading.excepthook = _hook


@asynccontextmanager
async def lifespan(app: FastAPI):
    _install_thread_excepthook()
    mgr = get_session_manager()
    mgr.set_event_loop(asyncio.get_running_loop())
    yield
    mgr.stop()


app = FastAPI(title="Speak Local Demo — Speaker Diarization", lifespan=lifespan)

if config.STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"ok": False, "message": message})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """HTTP 请求级未捕获异常 → 可读 JSON，不让 uvicorn worker 因业务异常退出。"""
    detail = decode_onnx_error(exc)
    logger.exception("未处理请求异常 %s %s: %s", request.method, request.url.path, detail)
    try:
        get_session_manager().report_error(detail)
    except Exception:  # noqa: BLE001
        pass
    return _error(500, detail)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    ort = provider_summary()
    return {
        "status": "ok",
        "sample_rate": config.SAMPLE_RATE,
        "ort": ort,
        "segmentation_available": config.SEGMENTATION_MODEL.is_file(),
        "default_embedding_model_key": config.DEFAULT_EMBEDDING_MODEL_KEY,
    }


@app.get("/api/models")
async def models() -> dict[str, Any]:
    return {
        "embedding_models": config.list_embedding_models(),
        "segmentation": {
            "path": str(config.SEGMENTATION_MODEL),
            "available": config.SEGMENTATION_MODEL.is_file(),
        },
        "default_embedding_model_key": config.DEFAULT_EMBEDDING_MODEL_KEY,
    }


@app.post("/api/initialize")
async def initialize(req: InitializeRequest) -> JSONResponse:
    mgr = get_session_manager()
    try:
        result = await asyncio.to_thread(
            mgr.initialize,
            req.embedding_model_key,
            req.use_gpu,
            req.verify_threshold,
        )
        return JSONResponse({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        return _error(400, str(exc))


@app.post("/api/start")
async def start(req: StartRequest) -> JSONResponse:
    mgr = get_session_manager()
    try:
        result = await asyncio.to_thread(
            mgr.start,
            req.window_sec,
            req.hop_sec,
            req.embedding_model_key,
            req.use_gpu,
            req.fixed_speaker_num,
            req.verify_threshold,
            req.slow_inference_threshold_ms,
            req.slow_inference_max_consecutive,
            req.low_frequency_interval_sec,
        )
        return JSONResponse({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        return _error(400, str(exc))


@app.post("/api/stream/chunk")
async def stream_chunk(
    request: Request,
    session_id: str | None = Query(default=None),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> JSONResponse:
    sid = session_id or x_session_id
    if not sid:
        return _error(400, "缺少 session_id（query 或 X-Session-Id header）")
    body = await request.body()
    if not body:
        return _error(400, "空 PCM 数据")
    mgr = get_session_manager()
    try:
        result = mgr.push_chunk(sid, body)
        return JSONResponse(result)
    except Exception as exc:  # noqa: BLE001
        return _error(400, str(exc))


@app.get("/api/status")
async def status() -> dict[str, Any]:
    return get_session_manager().status()


@app.get("/api/sse/events")
async def sse_events(request: Request) -> EventSourceResponse:
    mgr = get_session_manager()
    queue = mgr.subscribe_sse()

    async def event_generator():
        try:
            yield {
                "event": "status",
                "data": json.dumps(mgr.status(), ensure_ascii=False),
            }
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
                    continue
                event = item.get("event", "message")
                data = item.get("data", {})
                yield {
                    "event": event,
                    "data": json.dumps(data, ensure_ascii=False),
                }
        finally:
            mgr.unsubscribe_sse(queue)

    return EventSourceResponse(event_generator())


@app.post("/api/voice-enroll/start")
async def voice_enroll_start(body: SessionIdBody | None = None) -> JSONResponse:
    mgr = get_session_manager()
    sid = body.session_id if body else None
    try:
        result = mgr.voice_enroll_start(sid)
        return JSONResponse({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        return _error(400, str(exc))


@app.post("/api/voice-enroll/finish")
async def voice_enroll_finish(body: SessionIdBody | None = None) -> JSONResponse:
    mgr = get_session_manager()
    sid = body.session_id if body else None
    try:
        result = await asyncio.to_thread(mgr.voice_enroll_finish, sid)
        return JSONResponse({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        return _error(400, str(exc))


@app.post("/api/voice-enroll/from-file")
async def voice_enroll_from_file(
    request: Request,
    sample_rate: int = Query(default=config.SAMPLE_RATE),
) -> JSONResponse:
    """接收浏览器解码后的 16kHz mono PCM int16 LE（或同格式原始字节）。"""
    body = await request.body()
    if not body:
        return _error(400, "空音频数据")
    mgr = get_session_manager()
    try:
        result = await asyncio.to_thread(mgr.voice_enroll_from_pcm, body, sample_rate)
        return JSONResponse({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        return _error(400, str(exc))


@app.post("/api/voice-enroll/cancel")
async def voice_enroll_cancel(body: SessionIdBody | None = None) -> JSONResponse:
    mgr = get_session_manager()
    sid = body.session_id if body else None
    try:
        result = mgr.voice_enroll_cancel(sid)
        return JSONResponse({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        return _error(400, str(exc))


@app.post("/api/voice-enroll/clear")
async def voice_enroll_clear() -> JSONResponse:
    mgr = get_session_manager()
    try:
        result = mgr.voice_enroll_clear()
        return JSONResponse({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        return _error(400, str(exc))


@app.post("/api/stop")
async def stop() -> JSONResponse:
    mgr = get_session_manager()
    result = mgr.stop()
    return JSONResponse(result)


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host=config.HOST, port=config.PORT, reload=False)


if __name__ == "__main__":
    main()
