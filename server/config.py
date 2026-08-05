"""项目配置与 embedding 模型注册表。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
STATIC_DIR = ROOT_DIR / "static"
DATA_DIR = ROOT_DIR / "data"
ENROLLMENT_FILE = DATA_DIR / "enrollment.npy"

SAMPLE_RATE = 16000

SEGMENTATION_DIR = MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0"
SEGMENTATION_MODEL = SEGMENTATION_DIR / "model.onnx"

SPEAKER_RECOGNITION_RELEASE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models"
)
SPEAKER_SEGMENTATION_RELEASE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models"
)

# key -> {filename, display_name, framework hint, downloadable}
EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "campplus_zh_en_advanced": {
        "filename": "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        "display_name": "CAM++ 中英文 Advanced",
        "family": "3dspeaker",
        "downloadable": True,
    },
    "campplus_zh_cn_common": {
        "filename": "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        "display_name": "CAM++ 中文 Common",
        "family": "3dspeaker",
        "downloadable": True,
    },
    "campplus_en_voxceleb": {
        "filename": "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        "display_name": "CAM++ 英文 VoxCeleb",
        "family": "3dspeaker",
        "downloadable": True,
    },
    "eres2netv2_zh_cn": {
        "filename": "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx",
        "display_name": "ERes2NetV2 中文 Common",
        "family": "3dspeaker",
        "downloadable": True,
    },
    "eres2net_base_zh_cn_common": {
        "filename": "3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx",
        "display_name": "ERes2Net Base 200k 中文 Common",
        "family": "3dspeaker",
        "downloadable": True,
    },
    "eres2net_base_zh_cn_3dspeaker": {
        "filename": "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        "display_name": "ERes2Net Base 3D-Speaker",
        "family": "3dspeaker",
        "downloadable": True,
    },
    "eres2net_large_zh_cn_3dspeaker": {
        "filename": "3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx",
        "display_name": "ERes2Net Large 3D-Speaker",
        "family": "3dspeaker",
        "downloadable": True,
    },
    "eres2net_zh_cn_common": {
        "filename": "3dspeaker_speech_eres2net_sv_zh-cn_16k-common.onnx",
        "display_name": "ERes2Net 中文 Common",
        "family": "3dspeaker",
        "downloadable": True,
    },
    "eres2net_en_voxceleb": {
        "filename": "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
        "display_name": "ERes2Net 英文 VoxCeleb",
        "family": "3dspeaker",
        "downloadable": True,
    },
    "wespeaker_zh_cnceleb_resnet34": {
        "filename": "wespeaker_zh_cnceleb_resnet34.onnx",
        "display_name": "WeSpeaker ResNet34 中文 CN-Celeb",
        "family": "wespeaker",
        "downloadable": True,
    },
    "wespeaker_zh_cnceleb_resnet34_lm": {
        "filename": "wespeaker_zh_cnceleb_resnet34_LM.onnx",
        "display_name": "WeSpeaker ResNet34-LM 中文 CN-Celeb",
        "family": "wespeaker",
        "downloadable": True,
    },
    "wespeaker_en_voxceleb_resnet34": {
        "filename": "wespeaker_en_voxceleb_resnet34.onnx",
        "display_name": "WeSpeaker ResNet34 英文 VoxCeleb",
        "family": "wespeaker",
        "downloadable": True,
    },
    "wespeaker_en_voxceleb_resnet34_lm": {
        "filename": "wespeaker_en_voxceleb_resnet34_LM.onnx",
        "display_name": "WeSpeaker ResNet34-LM 英文 VoxCeleb",
        "family": "wespeaker",
        "downloadable": True,
    },
    "wespeaker_en_voxceleb_campplus": {
        "filename": "wespeaker_en_voxceleb_CAM++.onnx",
        "display_name": "WeSpeaker CAM++ 英文 VoxCeleb",
        "family": "wespeaker",
        "downloadable": True,
    },
    "wespeaker_en_voxceleb_campplus_lm": {
        "filename": "wespeaker_en_voxceleb_CAM++_LM.onnx",
        "display_name": "WeSpeaker CAM++-LM 英文 VoxCeleb",
        "family": "wespeaker",
        "downloadable": True,
    },
    "wespeaker_en_voxceleb_resnet152_lm": {
        "filename": "wespeaker_en_voxceleb_resnet152_LM.onnx",
        "display_name": "WeSpeaker ResNet152-LM 英文 VoxCeleb",
        "family": "wespeaker",
        "downloadable": True,
    },
    "wespeaker_en_voxceleb_resnet221_lm": {
        "filename": "wespeaker_en_voxceleb_resnet221_LM.onnx",
        "display_name": "WeSpeaker ResNet221-LM 英文 VoxCeleb",
        "family": "wespeaker",
        "downloadable": True,
    },
    "wespeaker_en_voxceleb_resnet293_lm": {
        "filename": "wespeaker_en_voxceleb_resnet293_LM.onnx",
        "display_name": "WeSpeaker ResNet293-LM 英文 VoxCeleb",
        "family": "wespeaker",
        "downloadable": True,
    },
}

DEFAULT_EMBEDDING_MODEL_KEY = "campplus_zh_en_advanced"

DIARIZATION_CLUSTER_THRESHOLD = 0.78
DIARIZATION_MIN_DURATION_ON = 0.3
DIARIZATION_MIN_DURATION_OFF = 0.5

VERIFICATION_THRESHOLD = 0.55

DEFAULT_WINDOW_SEC = 10.0
DEFAULT_HOP_SEC = 5.0

MIN_ANALYSIS_AUDIO_SEC = 1.5
MIN_ENROLLMENT_DURATION_SEC = 5.0
MIN_SPEAKER_EMBEDDING_SEC = 0.8

# 简单能量 VAD（注册用）
ENERGY_VAD_FRAME_MS = 30
ENERGY_VAD_HOP_MS = 10
ENERGY_VAD_THRESHOLD_RATIO = 0.15

HOST = "0.0.0.0"
PORT = 8764


def embedding_model_path(key: str) -> Path:
    info = EMBEDDING_MODELS[key]
    return MODELS_DIR / info["filename"]


def format_size_label(size_bytes: int) -> str:
    """将字节数格式化为可读大小，如 27.0 MB。"""
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size_bytes)
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    if unit_idx == 0:
        return f"{int(value)} {units[unit_idx]}"
    return f"{value:.1f} {units[unit_idx]}"


def model_label(info: dict[str, Any]) -> str:
    return str(info.get("label") or info.get("display_name") or info.get("filename") or "")


def list_embedding_models() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key, info in EMBEDDING_MODELS.items():
        path = embedding_model_path(key)
        available = path.is_file()
        size_bytes: int | None = path.stat().st_size if available else None
        label = model_label(info)
        items.append(
            {
                "key": key,
                "label": label,
                "display_name": label,
                "family": info["family"],
                "filename": info["filename"],
                "downloadable": bool(info.get("downloadable", True)),
                "available": available,
                "path": str(path),
                "size_bytes": size_bytes,
                "size_label": format_size_label(size_bytes) if size_bytes is not None else None,
                "default": key == DEFAULT_EMBEDDING_MODEL_KEY,
            }
        )
    return items
