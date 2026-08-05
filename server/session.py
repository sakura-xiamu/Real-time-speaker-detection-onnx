"""单机单活跃会话：PCM 缓冲、滑窗推理、SSE 广播。"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from server import config
from server.engine.diarization import DiarizationEngine
from server.engine.providers import (
    decode_onnx_error,
    is_gpu_runtime_error,
    provider_summary,
)
from server.voice_profile import VoiceProfileStore

logger = logging.getLogger(__name__)


def pcm16_to_float32(data: bytes) -> np.ndarray:
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return samples / 32768.0


def float32_to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return np.ascontiguousarray(audio.reshape(-1))


@dataclass
class SessionState:
    session_id: str
    window_sec: float = config.DEFAULT_WINDOW_SEC
    hop_sec: float = config.DEFAULT_HOP_SEC
    original_hop_sec: float = config.DEFAULT_HOP_SEC
    embedding_model_key: str = config.DEFAULT_EMBEDDING_MODEL_KEY
    use_gpu: bool = True
    fixed_speaker_num: int = 0
    verify_threshold: float = config.VERIFICATION_THRESHOLD
    slow_inference_threshold_ms: float = 1000.0
    slow_inference_max_consecutive: int = 5
    low_frequency_interval_sec: float = 60.0
    running: bool = False
    accept_audio: bool = False
    enrolling: bool = False
    stream_buffer: list[np.ndarray] = field(default_factory=list)
    enroll_buffer: list[np.ndarray] = field(default_factory=list)
    total_samples: int = 0
    window_index: int = 0
    analysis_busy: bool = False
    low_frequency_mode: bool = False
    slow_inference_count: int = 0
    low_freq_fast_count: int = 0
    last_result: dict[str, Any] | None = None
    last_error: str | None = None
    message: str = ""
    created_at: float = field(default_factory=time.time)
    providers: dict[str, Any] = field(default_factory=dict)

    def append_chunk(self, chunk: np.ndarray) -> None:
        self.stream_buffer.append(chunk)
        self.total_samples += len(chunk)
        if self.enrolling:
            self.enroll_buffer.append(chunk)
        self._trim()

    def _trim(self) -> None:
        max_samples = int(self.window_sec * config.SAMPLE_RATE)
        total = sum(len(c) for c in self.stream_buffer)
        while total > max_samples and self.stream_buffer:
            removed = self.stream_buffer.pop(0)
            total -= len(removed)

    def get_stream_audio(self) -> np.ndarray:
        if not self.stream_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.stream_buffer)

    def get_enroll_audio(self) -> np.ndarray:
        if not self.enroll_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.enroll_buffer)

    def enroll_buffered_sec(self) -> float:
        return sum(len(c) for c in self.enroll_buffer) / config.SAMPLE_RATE

    def clear_stream(self) -> None:
        self.stream_buffer.clear()
        self.total_samples = 0

    def reset_enroll(self) -> None:
        self.enroll_buffer.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "running": self.running,
            "accept_audio": self.accept_audio,
            "enrolling": self.enrolling,
            "window_sec": self.window_sec,
            "hop_sec": self.hop_sec,
            "original_hop_sec": self.original_hop_sec,
            "embedding_model_key": self.embedding_model_key,
            "use_gpu": self.use_gpu,
            "fixed_speaker_num": self.fixed_speaker_num,
            "verify_threshold": self.verify_threshold,
            "slow_inference_threshold_ms": self.slow_inference_threshold_ms,
            "slow_inference_max_consecutive": self.slow_inference_max_consecutive,
            "low_frequency_interval_sec": self.low_frequency_interval_sec,
            "low_frequency_mode": self.low_frequency_mode,
            "window_index": self.window_index,
            "audio_sec": round(self.total_samples / config.SAMPLE_RATE, 2),
            "analysis_busy": self.analysis_busy,
            "enroll_buffered_sec": round(self.enroll_buffered_sec(), 2),
            "message": self.message,
            "last_error": self.last_error,
            "last_result": self.last_result,
            "providers": self.providers,
            "has_enrollment": False,
        }


class SessionManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._session: SessionState | None = None
        self._engine: DiarizationEngine | None = None
        self._initialized = False
        self._embedding_model_key = config.DEFAULT_EMBEDDING_MODEL_KEY
        self._use_gpu = True
        self._voice = VoiceProfileStore()
        self._pending_audio: np.ndarray | None = None
        self._voice_enroll_last_error = ""
        self._sse_queues: list[asyncio.Queue] = []
        self._sse_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._analysis_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe_sse(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        with self._sse_lock:
            self._sse_queues.append(q)
        return q

    def unsubscribe_sse(self, q: asyncio.Queue) -> None:
        with self._sse_lock:
            if q in self._sse_queues:
                self._sse_queues.remove(q)

    def broadcast(self, event: str, data: dict[str, Any]) -> None:
        payload = {"event": event, "data": data}
        with self._sse_lock:
            queues = list(self._sse_queues)
        for q in queues:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except Exception:
                    pass
                try:
                    q.put_nowait(payload)
                except Exception:
                    pass

    def report_error(self, message: str) -> None:
        with self._lock:
            if self._session is not None:
                self._session.last_error = message
                self._session.message = message
        self.broadcast("error", {"message": message})
        self.broadcast("status", self.status())

    def _state_label(self) -> str:
        if self._session is not None and self._session.running:
            if self._session.analysis_busy:
                return "busy"
            return "running"
        if self._initialized:
            return "ready"
        return "idle"

    def _pending_sec(self) -> float:
        if self._pending_audio is None:
            return 0.0
        return float(len(self._pending_audio) / config.SAMPLE_RATE)

    def status(self) -> dict[str, Any]:
        with self._lock:
            base = {
                "sample_rate": config.SAMPLE_RATE,
                "has_enrollment": self._voice.has_enrollment(),
                "initialized": self._initialized,
                "state": self._state_label(),
                "engine_loaded": self._engine is not None,
                "embedding_model_key": self._embedding_model_key,
                "use_gpu": self._use_gpu,
                "voice_enroll_pending": self._pending_audio is not None,
                "voice_enroll_pending_sec": round(self._pending_sec(), 2),
                "voice_enroll_last_error": self._voice_enroll_last_error,
                "default_window_sec": config.DEFAULT_WINDOW_SEC,
                "default_hop_sec": config.DEFAULT_HOP_SEC,
                "default_embedding_model_key": config.DEFAULT_EMBEDDING_MODEL_KEY,
                "verification_threshold": config.VERIFICATION_THRESHOLD,
                "ort": provider_summary(),
                "session": None,
            }
            if self._session is not None:
                snap = self._session.snapshot()
                snap["has_enrollment"] = self._voice.has_enrollment()
                base["session"] = snap
            return base

    def initialize(
        self,
        embedding_model_key: str,
        use_gpu: bool,
        verify_threshold: float | None = None,
    ) -> dict[str, Any]:
        if embedding_model_key not in config.EMBEDDING_MODELS:
            raise ValueError(f"未知 embedding 模型: {embedding_model_key}")
        emb_path = config.embedding_model_path(embedding_model_key)
        if not emb_path.is_file():
            raise FileNotFoundError(
                f"模型文件不存在: {emb_path}，请先运行 python scripts/download_models.py"
            )
        if not config.SEGMENTATION_MODEL.is_file():
            raise FileNotFoundError(
                f"分割模型不存在: {config.SEGMENTATION_MODEL}，请先运行 python scripts/download_models.py"
            )

        with self._lock:
            if self._session is not None and self._session.enrolling:
                raise RuntimeError("正在录制声纹，请先完成或取消后再初始化")
            self._stop_detection_locked(keep_engine=False)
            self._embedding_model_key = embedding_model_key
            self._use_gpu = bool(use_gpu)
            if self._session is not None and verify_threshold is not None:
                self._session.verify_threshold = float(verify_threshold)

        engine = DiarizationEngine(
            segmentation_path=str(config.SEGMENTATION_MODEL),
            embedding_path=str(emb_path),
            use_gpu=bool(use_gpu),
        )
        with self._lock:
            self._engine = engine
            self._initialized = True
            message = f"模型已初始化（{embedding_model_key}）"
            if self._session is not None:
                self._session.embedding_model_key = embedding_model_key
                self._session.use_gpu = bool(use_gpu)
                self._session.providers = engine.providers
                self._session.message = message
                if verify_threshold is not None:
                    self._session.verify_threshold = float(verify_threshold)

        enroll_msg = self._commit_pending(trigger="initialize")
        if enroll_msg:
            message = f"{message}；{enroll_msg}"
            with self._lock:
                if self._session is not None:
                    self._session.message = message

        status = self.status()
        self.broadcast("status", status)
        return {
            "ok": True,
            "message": message,
            "providers": engine.providers,
            "initialized": True,
        }

    def start(
        self,
        window_sec: float,
        hop_sec: float,
        embedding_model_key: str | None = None,
        use_gpu: bool | None = None,
        fixed_speaker_num: int = 0,
        verify_threshold: float = config.VERIFICATION_THRESHOLD,
        slow_inference_threshold_ms: float = 1000.0,
        slow_inference_max_consecutive: int = 5,
        low_frequency_interval_sec: float = 60.0,
    ) -> dict[str, Any]:
        err = self._validate(
            window_sec,
            hop_sec,
            fixed_speaker_num,
            verify_threshold,
            slow_inference_threshold_ms,
            slow_inference_max_consecutive,
            low_frequency_interval_sec,
        )
        if err:
            raise ValueError(err)

        with self._lock:
            if not self._initialized or self._engine is None:
                raise RuntimeError("请先初始化模型（POST /api/initialize）")
            if self._session is not None and self._session.enrolling:
                raise RuntimeError("正在录制声纹，请先完成或取消后再开始检测")

            model_key = embedding_model_key or self._embedding_model_key
            gpu = self._use_gpu if use_gpu is None else bool(use_gpu)
            if model_key != self._embedding_model_key or gpu != self._use_gpu:
                raise RuntimeError("模型配置与已初始化不一致，请重新 initialize")

            self._stop_detection_locked(keep_engine=True)

            session_id = uuid.uuid4().hex
            self._session = SessionState(
                session_id=session_id,
                window_sec=float(window_sec),
                hop_sec=float(hop_sec),
                original_hop_sec=float(hop_sec),
                embedding_model_key=model_key,
                use_gpu=gpu,
                fixed_speaker_num=int(fixed_speaker_num),
                verify_threshold=float(verify_threshold),
                slow_inference_threshold_ms=float(slow_inference_threshold_ms),
                slow_inference_max_consecutive=int(slow_inference_max_consecutive),
                low_frequency_interval_sec=float(low_frequency_interval_sec),
                running=True,
                accept_audio=True,
                message=f"已开始检测（窗口 {window_sec}s / 步长 {hop_sec}s）",
                providers=dict(self._engine.providers) if self._engine else {},
            )
            self._stop_event.clear()
            engine = self._engine

        enroll_msg = self._commit_pending(trigger="start")
        with self._lock:
            if self._session is not None and enroll_msg:
                self._session.message = f"{self._session.message}；{enroll_msg}"

        self._analysis_thread = threading.Thread(
            target=self._analysis_loop, name="diarization-hop", daemon=True
        )
        self._analysis_thread.start()

        status = self.status()
        self.broadcast("status", status)
        msg = self._session.message if self._session else "ok"
        return {
            "session_id": session_id,
            "message": msg,
            "providers": engine.providers if engine else {},
        }

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._stop_detection_locked(keep_engine=True)
            if self._initialized:
                message = "已停止检测，模型仍就绪"
            else:
                message = "已停止"
        status = self.status()
        self.broadcast("status", status)
        return {"ok": True, "message": message, "initialized": self._initialized}

    def _stop_detection_locked(self, *, keep_engine: bool) -> None:
        self._stop_event.set()
        if self._session is not None:
            was_enrolling = self._session.enrolling
            self._session.running = False
            self._session.accept_audio = False
            self._session.enrolling = False
            self._session.analysis_busy = False
            self._session.reset_enroll()
            self._session.clear_stream()
            self._session.low_frequency_mode = False
            self._session.hop_sec = self._session.original_hop_sec
            self._session.slow_inference_count = 0
            self._session.low_freq_fast_count = 0
            self._session.message = "已停止检测" if not was_enrolling else "已停止"
        if not keep_engine:
            self._engine = None
            self._initialized = False

    def push_chunk(self, session_id: str, data: bytes) -> dict[str, Any]:
        with self._lock:
            if self._session is None or not self._session.accept_audio:
                raise RuntimeError("当前未接受音频（请先开始检测或开始录声纹）")
            if self._session.session_id != session_id:
                raise RuntimeError("session_id 不匹配")
            chunk = pcm16_to_float32(data)
            self._session.append_chunk(chunk)
            return {
                "ok": True,
                "bytes": len(data),
                "audio_sec": round(self._session.total_samples / config.SAMPLE_RATE, 2),
                "enroll_buffered_sec": round(self._session.enroll_buffered_sec(), 2),
            }

    def voice_enroll_start(self, session_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if self._session is not None and self._session.enrolling:
                raise RuntimeError("已在录制声纹")

            if (
                self._session is not None
                and self._session.running
                and self._session.accept_audio
            ):
                s = self._require_session(session_id)
                s.enrolling = True
                s.reset_enroll()
                s.message = "开始录制注册声纹，请清晰说话"
                sid = s.session_id
                msg = s.message
            else:
                # 采集-only：未检测时也可录制（可无模型 → finish 暂存）
                sid = uuid.uuid4().hex
                self._session = SessionState(
                    session_id=sid,
                    embedding_model_key=self._embedding_model_key,
                    use_gpu=self._use_gpu,
                    running=False,
                    accept_audio=True,
                    enrolling=True,
                    message="开始录制注册声纹（采集-only），请清晰说话",
                    providers=dict(self._engine.providers) if self._engine else {},
                )
                msg = self._session.message

        status = self.status()
        self.broadcast("status", status)
        return {"ok": True, "message": msg, "session_id": sid}

    def voice_enroll_finish(self, session_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            s = self._require_session(session_id)
            if not s.enrolling:
                raise RuntimeError("当前未在录制声纹")
            audio = float32_to_mono(s.get_enroll_audio())
            s.enrolling = False
            s.reset_enroll()
            was_running = s.running
            if not was_running:
                s.accept_audio = False
            engine = self._engine

        if engine is None:
            result = self._store_pending(audio)
        else:
            try:
                result = self._voice.enroll(engine, audio)
                self._clear_pending(clear_error=True)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._voice_enroll_last_error = str(exc)
                    if self._session is not None:
                        self._session.message = str(exc)
                status = self.status()
                self.broadcast("status", status)
                raise

        with self._lock:
            if self._session is not None:
                self._session.message = result.get("message", "声纹处理完成")

        status = self.status()
        self.broadcast("status", status)
        return result

    def voice_enroll_from_pcm(self, data: bytes, sample_rate: int = config.SAMPLE_RATE) -> dict[str, Any]:
        if not data:
            raise ValueError("空 PCM 数据")
        if sample_rate != config.SAMPLE_RATE:
            raise ValueError(f"仅支持 {config.SAMPLE_RATE} Hz PCM，收到 {sample_rate}")
        audio = float32_to_mono(pcm16_to_float32(data))
        if audio.size == 0:
            raise ValueError("PCM 无有效采样")

        with self._lock:
            if self._session is not None and self._session.enrolling:
                raise RuntimeError("正在录制声纹，请先完成或取消")
            engine = self._engine

        if engine is None:
            result = self._store_pending(audio)
        else:
            try:
                result = self._voice.enroll(engine, audio)
                self._clear_pending(clear_error=True)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._voice_enroll_last_error = str(exc)
                status = self.status()
                self.broadcast("status", status)
                raise

        with self._lock:
            if self._session is not None:
                self._session.message = result.get("message", "声纹处理完成")

        status = self.status()
        self.broadcast("status", status)
        return result

    def voice_enroll_cancel(self, session_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            s = self._require_session(session_id)
            s.enrolling = False
            s.reset_enroll()
            if not s.running:
                s.accept_audio = False
            s.message = "已取消声纹录制"
            msg = s.message
        status = self.status()
        self.broadcast("status", status)
        return {"ok": True, "message": msg}

    def voice_enroll_clear(self) -> dict[str, Any]:
        self._voice.clear()
        self._clear_pending(clear_error=True)
        with self._lock:
            if self._session is not None:
                self._session.message = "已清除注册声纹"
        status = self.status()
        self.broadcast("status", status)
        return {"ok": True, "message": "已清除注册声纹与暂存"}

    def _store_pending(self, audio: np.ndarray) -> dict[str, Any]:
        audio = float32_to_mono(audio)
        duration = len(audio) / config.SAMPLE_RATE
        if duration < config.MIN_ENROLLMENT_DURATION_SEC:
            raise ValueError(
                f"注册音频太短（{duration:.1f}s），请至少 "
                f"{config.MIN_ENROLLMENT_DURATION_SEC}s"
            )
        with self._lock:
            self._pending_audio = audio
            self._voice_enroll_last_error = ""
            msg = (
                f"声纹已暂存 {duration:.1f}s，初始化或开始检测时自动注册"
            )
            if self._session is not None:
                self._session.message = msg
        return {
            "ok": True,
            "pending": True,
            "enrolled": False,
            "duration_sec": round(duration, 2),
            "speech_sec": round(duration, 2),
            "message": msg,
        }

    def _clear_pending(self, *, clear_error: bool) -> None:
        with self._lock:
            self._pending_audio = None
            if clear_error:
                self._voice_enroll_last_error = ""

    def _commit_pending(self, *, trigger: str) -> str:
        with self._lock:
            audio = self._pending_audio
            engine = self._engine
            if audio is None:
                return ""
            if engine is None:
                return ""
            self._pending_audio = None

        try:
            result = self._voice.enroll(engine, audio)
            with self._lock:
                self._voice_enroll_last_error = ""
            msg = result.get("message", "声纹注册成功")
            logger.info("pending voice enroll committed via %s: %s", trigger, msg)
            return f"暂存声纹已注册（{trigger}）"
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            with self._lock:
                self._voice_enroll_last_error = err
            logger.warning("pending voice enroll commit failed via %s: %s", trigger, err)
            return f"暂存声纹注册失败: {err}"

    def _require_session(self, session_id: str | None) -> SessionState:
        if self._session is None:
            raise RuntimeError("无活跃会话")
        if session_id and self._session.session_id != session_id:
            raise RuntimeError("session_id 不匹配")
        return self._session

    @staticmethod
    def _validate(
        window_sec: float,
        hop_sec: float,
        fixed_speaker_num: int,
        verify_threshold: float,
        slow_inference_threshold_ms: float,
        slow_inference_max_consecutive: int,
        low_frequency_interval_sec: float,
    ) -> str | None:
        if not (3.0 <= window_sec <= 30.0):
            return "window_sec 须在 3–30 秒之间"
        if not (1.0 <= hop_sec <= window_sec):
            return "hop_sec 须在 1 秒到 window_sec 之间"
        if fixed_speaker_num < 0:
            return "fixed_speaker_num 不能为负数"
        if not (0.1 <= verify_threshold <= 0.99):
            return "verify_threshold 须在 0.1–0.99 之间"
        if slow_inference_threshold_ms < 0:
            return "slow_inference_threshold_ms 不能为负"
        if slow_inference_max_consecutive < 0:
            return "slow_inference_max_consecutive 不能为负"
        if low_frequency_interval_sec < 5 and slow_inference_max_consecutive > 0:
            # 0 threshold disables; when enabled, interval min 5
            if slow_inference_threshold_ms > 0:
                return "low_frequency_interval_sec 须 ≥ 5"
        return None

    def _apply_slow_inference(self, session_id: str, latency: float) -> str | None:
        """根据延迟更新慢推理状态，返回提示文案（若有）。"""
        note: str | None = None
        with self._lock:
            s = self._session
            if s is None or s.session_id != session_id:
                return None
            thr_ms = s.slow_inference_threshold_ms
            max_n = s.slow_inference_max_consecutive
            if thr_ms <= 0 or max_n <= 0:
                return None
            thr_sec = thr_ms / 1000.0
            if latency > thr_sec:
                s.slow_inference_count += 1
                s.low_freq_fast_count = 0
                if s.slow_inference_count >= max_n:
                    if not s.low_frequency_mode:
                        s.low_frequency_mode = True
                        s.hop_sec = s.low_frequency_interval_sec
                        s.slow_inference_count = 0
                        s.low_freq_fast_count = 0
                        note = (
                            f"连续 {max_n} 次推理超过 {thr_sec:.3f}s，"
                            f"已降级为每 {s.low_frequency_interval_sec:.0f}s 检测一次"
                        )
                        s.message = note
                    else:
                        s.slow_inference_count = 0
            else:
                s.slow_inference_count = 0
                if s.low_frequency_mode:
                    s.low_freq_fast_count += 1
                    if s.low_freq_fast_count >= max_n:
                        s.low_frequency_mode = False
                        s.hop_sec = s.original_hop_sec
                        s.low_freq_fast_count = 0
                        note = (
                            f"低频模式下连续 {max_n} 次低于 {thr_sec:.3f}s，"
                            f"已恢复正常检测"
                        )
                        s.message = note
        return note

    def _analysis_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                s = self._session
                if s is None or not s.running:
                    break
                hop = s.hop_sec
                session_id = s.session_id

            if self._stop_event.wait(hop):
                break

            with self._lock:
                s = self._session
                if s is None or not s.running or s.session_id != session_id:
                    break
                if s.analysis_busy:
                    continue
                min_samples = int(config.MIN_ANALYSIS_AUDIO_SEC * config.SAMPLE_RATE)
                audio = s.get_stream_audio()
                if len(audio) < min_samples:
                    continue
                s.analysis_busy = True
                engine = self._engine
                num_speakers = s.fixed_speaker_num
                thr = s.verify_threshold

            if engine is None:
                with self._lock:
                    if self._session is not None and self._session.session_id == session_id:
                        self._session.analysis_busy = False
                continue

            try:
                t0 = time.perf_counter()
                result = engine.process(audio, num_speakers=num_speakers)
                voice = self._voice.compare_window(
                    engine, audio, result.segments, threshold=thr
                )
                latency = time.perf_counter() - t0
                mode_note = self._apply_slow_inference(session_id, latency)
                fallback_note = ""
                if engine.last_fallback_reason:
                    fallback_note = "（已回退 CPU）"
                payload = {
                    "type": "diarization_result",
                    "window_index": 0,
                    "audio_sec": 0.0,
                    "num_speakers": result.num_speakers,
                    "latency_sec": round(latency, 3),
                    "active_ratio": round(result.active_ratio, 3),
                    "duration_sec": round(result.duration_sec, 3),
                    "segments": [seg.to_dict() for seg in result.segments],
                    "voice": voice,
                    "providers": dict(engine.providers),
                    "low_frequency_mode": False,
                }
                with self._lock:
                    if self._session is None or self._session.session_id != session_id:
                        continue
                    self._session.window_index += 1
                    payload["window_index"] = self._session.window_index
                    payload["audio_sec"] = round(
                        self._session.total_samples / config.SAMPLE_RATE, 2
                    )
                    payload["low_frequency_mode"] = self._session.low_frequency_mode
                    self._session.last_result = payload
                    self._session.last_error = None
                    self._session.providers = dict(engine.providers)
                    if mode_note:
                        self._session.message = mode_note
                    else:
                        self._session.message = (
                            f"窗口 #{payload['window_index']}：{result.num_speakers} 人"
                            f"{fallback_note}"
                        )
                self.broadcast("diarization_result", payload)
                self.broadcast("status", self.status())
            except Exception as exc:  # noqa: BLE001
                detail = decode_onnx_error(exc)
                err = f"窗口分析失败: {detail}"
                try:
                    if is_gpu_runtime_error(exc):
                        engine.force_all_cpu(detail)
                        err = f"窗口分析失败（已回退 CPU，下窗将重试）: {detail}"
                except Exception as fb_exc:  # noqa: BLE001
                    err = (
                        f"窗口分析失败: {detail}；"
                        f"回退 CPU 也失败: {decode_onnx_error(fb_exc)}"
                    )
                with self._lock:
                    if self._session is not None and self._session.session_id == session_id:
                        self._session.last_error = err
                        self._session.message = err
                        self._session.providers = dict(engine.providers)
                self.broadcast("error", {"message": err})
                self.broadcast("status", self.status())
            finally:
                with self._lock:
                    if self._session is not None and self._session.session_id == session_id:
                        self._session.analysis_busy = False


_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager
