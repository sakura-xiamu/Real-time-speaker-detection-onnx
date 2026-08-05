"""ONNX Runtime ExecutionProvider 选择（Windows DirectML → CPU）。"""
from __future__ import annotations

import logging
import platform
import threading
from typing import Any, Callable

DML_PROVIDER = "DmlExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
COREML_PROVIDER = "CoreMLExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"

EXCLUDED_PROVIDERS = {
    "AzureExecutionProvider",
}

logger = logging.getLogger(__name__)

_DXGI_ERROR_REMOVED_CODE = b"887A0005"
_GBK_DEVICE_BYTES = b"\xc9\xe8\xb1\xb8"
_GBK_PAUSE_BYTES = b"\xd4\xdd\xcd\xa3"
_DEVICE_REMOVED_KEYWORDS = (
    "887A0005",
    "DeviceRemoved",
    "DXGI",
    "DmlExecutionProvider",
    "ReadbackHeap",
    "GPU 设备",
    "设备实例",
    "已经暂停",
)


def decode_onnx_error(exc: BaseException) -> str:
    if isinstance(exc, UnicodeDecodeError):
        raw = getattr(exc, "object", b"")
        if isinstance(raw, (bytes, bytearray)):
            for encoding in ("gbk", "utf-8", "latin-1"):
                try:
                    return raw.decode(encoding, errors="replace")
                except Exception:
                    continue
            return raw.decode("utf-8", errors="replace")
        return f"UnicodeDecodeError: {exc}"
    return str(exc)


def is_device_removed_error(exc: BaseException) -> bool:
    if isinstance(exc, UnicodeDecodeError):
        raw = getattr(exc, "object", b"")
        if isinstance(raw, (bytes, bytearray)):
            if _DXGI_ERROR_REMOVED_CODE in raw:
                return True
            if _GBK_DEVICE_BYTES in raw or _GBK_PAUSE_BYTES in raw:
                return True
        return False
    text = decode_onnx_error(exc)
    return any(kw in text for kw in _DEVICE_REMOVED_KEYWORDS)


def is_gpu_runtime_error(exc: BaseException) -> bool:
    """判断是否为 GPU/DML 运行时失败（含 GBK 解码失败）。"""
    if isinstance(exc, UnicodeDecodeError):
        return True
    if is_device_removed_error(exc):
        return True
    text = decode_onnx_error(exc).lower()
    keys = (
        "dml",
        "directml",
        "cuda",
        "averagepool",
        "80070057",
        "e_invalidarg",
        "non-zero status",
        "onnxruntime",
        "fail",
        "invalid argument",
        "not implemented",
        "ep error",
        "execution provider",
    )
    return any(k in text for k in keys)


def _safe_close_session(session: Any) -> None:
    """尽力释放 ORT session，避免 DML 资源残留拖垮进程。"""
    if session is None:
        return
    try:
        del session
    except Exception:  # noqa: BLE001
        pass


def _available_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return [CPU_PROVIDER]


def _accel_providers(available: list[str]) -> list[str]:
    excluded = EXCLUDED_PROVIDERS | {CPU_PROVIDER}
    preferred_order = [DML_PROVIDER, CUDA_PROVIDER, COREML_PROVIDER]
    accel = [p for p in preferred_order if p in available and p not in excluded]
    for p in available:
        if p not in excluded and p not in accel:
            accel.append(p)
    return accel


def get_available_providers() -> list[str]:
    return _available_providers()


def get_onnx_providers(use_gpu: bool = True) -> list[str]:
    """返回 providers 列表。Windows 优先 DirectML。"""
    available = _available_providers()
    if not use_gpu:
        return [CPU_PROVIDER] if CPU_PROVIDER in available else available[:1] or [CPU_PROVIDER]

    providers: list[str] = []
    system = platform.system()
    if system == "Windows" and DML_PROVIDER in available:
        # 仅列 DML；部分模型算子 DML 不支持，由 OrtSessionHandle 运行时回退 CPU
        providers.append(DML_PROVIDER)
    else:
        providers.extend(_accel_providers(available))

    if not providers:
        providers = [CPU_PROVIDER]
    return providers


def _make_session(model_path: str, providers: list[str], num_threads: int = 2):
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = num_threads
    sess_opts.intra_op_num_threads = num_threads
    sess_opts.log_severity_level = 3
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_opts,
        providers=providers,
    )
    return session, list(session.get_providers())


def create_inference_session(
    model_path: str,
    use_gpu: bool = True,
    num_threads: int = 2,
):
    """创建 ORT session；创建失败时回退 CPU。"""
    providers = get_onnx_providers(use_gpu=use_gpu)
    last_error: BaseException | None = None

    attempts = [providers]
    if use_gpu and providers != [CPU_PROVIDER]:
        attempts.append([CPU_PROVIDER])

    for attempt in attempts:
        try:
            return _make_session(model_path, attempt, num_threads=num_threads)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    msg = decode_onnx_error(last_error) if last_error else "unknown"
    raise RuntimeError(f"无法创建 ONNX Runtime Session: {msg}")


class OrtSessionHandle:
    """带运行时 GPU→CPU 回退的 Session 包装。"""

    def __init__(
        self,
        model_path: str,
        use_gpu: bool = True,
        num_threads: int = 2,
        on_fallback: Callable[[str], None] | None = None,
    ) -> None:
        self.model_path = model_path
        self.use_gpu = bool(use_gpu)
        self.num_threads = num_threads
        self._on_fallback = on_fallback
        self._lock = threading.RLock()
        self.last_fallback_reason: str | None = None
        self.session, self.providers = create_inference_session(
            model_path, use_gpu=self.use_gpu, num_threads=num_threads
        )
        self._fell_back_to_cpu = self._providers_are_cpu(self.providers)
        if self._fell_back_to_cpu:
            self.use_gpu = False

    @staticmethod
    def _providers_are_cpu(providers: list[str]) -> bool:
        return (
            providers == [CPU_PROVIDER]
            or (len(providers) == 1 and providers[0] == CPU_PROVIDER)
            or (not providers)
        )

    @property
    def fell_back_to_cpu(self) -> bool:
        return self._fell_back_to_cpu

    def force_cpu(self, reason: str = "") -> bool:
        """销毁当前 session 并重建为 CPU。已是 CPU 则返回 False。"""
        with self._lock:
            if self._fell_back_to_cpu:
                return False
            reason = reason or "GPU/DML 不可用"
            self.last_fallback_reason = reason
            # 先标记，避免重建失败后继续打 DML
            self._fell_back_to_cpu = True
            self.use_gpu = False
            old = self.session
            self.session = None
            _safe_close_session(old)
            try:
                self.session, self.providers = _make_session(
                    self.model_path, [CPU_PROVIDER], num_threads=self.num_threads
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"回退 CPU 失败: {decode_onnx_error(exc)}（原因: {reason}）"
                ) from exc
            logger.warning(
                "ONNX session 已回退 CPU: model=%s reason=%s",
                self.model_path,
                reason,
            )
            return True

    def run(self, output_names, input_feed):
        notify: Callable[[str], None] | None = None
        reason = ""
        with self._lock:
            try:
                return self.session.run(output_names, input_feed)
            except Exception as exc:  # noqa: BLE001
                if self._fell_back_to_cpu or not is_gpu_runtime_error(exc):
                    raise RuntimeError(decode_onnx_error(exc)) from exc
                reason = decode_onnx_error(exc)
                # DML/GPU 算子失败 → 重建 CPU session 并重试一次
                self.force_cpu(reason)
                notify = self._on_fallback
                try:
                    out = self.session.run(output_names, input_feed)
                except Exception as exc2:  # noqa: BLE001
                    raise RuntimeError(decode_onnx_error(exc2)) from exc2
        # 锁外通知同伴模型一并回退，避免残留 DML 状态拖垮进程
        if notify is not None:
            try:
                notify(reason)
            except Exception:  # noqa: BLE001
                logger.exception("on_fallback 回调失败")
        return out

    def get_inputs(self):
        with self._lock:
            return self.session.get_inputs()

    def get_outputs(self):
        with self._lock:
            return self.session.get_outputs()

    def get_modelmeta(self):
        with self._lock:
            return self.session.get_modelmeta()


def provider_summary() -> dict[str, Any]:
    available = get_available_providers()
    return {
        "platform": platform.system(),
        "available_providers": available,
        "dml_available": DML_PROVIDER in available,
        "preferred_gpu_providers": get_onnx_providers(use_gpu=True),
        "cpu_providers": get_onnx_providers(use_gpu=False),
    }
