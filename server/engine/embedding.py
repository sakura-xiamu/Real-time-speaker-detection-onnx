"""说话人 embedding ONNX 提取器（3d-speaker / wespeaker）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .fbank import compute_fbank, subtract_global_mean
from .providers import OrtSessionHandle


@dataclass
class EmbeddingMeta:
    output_dim: int
    sample_rate: int
    normalize_samples: bool
    feature_normalize_type: str
    framework: str
    language: str


class EmbeddingExtractor:
    def __init__(
        self,
        model_path: str,
        use_gpu: bool = True,
        on_fallback: Callable[[str], None] | None = None,
    ) -> None:
        self.model_path = model_path
        self.handle = OrtSessionHandle(
            model_path, use_gpu=use_gpu, on_fallback=on_fallback
        )
        self.providers = list(self.handle.providers)
        raw = self.handle.get_modelmeta().custom_metadata_map
        framework = (raw.get("framework") or "3d-speaker").strip()
        normalize_samples = True
        if "normalize_samples" in raw:
            normalize_samples = str(raw["normalize_samples"]).strip() not in (
                "0",
                "false",
                "False",
            )
        elif framework == "wespeaker":
            normalize_samples = False

        feature_normalize_type = (raw.get("feature_normalize_type") or "").strip()
        if not feature_normalize_type and framework in ("3d-speaker", "3dspeaker"):
            feature_normalize_type = "global-mean"

        sample_rate = int(raw.get("sample_rate") or 16000)
        output_dim = int(raw.get("output_dim") or 0)
        if output_dim <= 0:
            out_shape = self.handle.get_outputs()[0].shape
            for dim in reversed(list(out_shape)):
                if isinstance(dim, int) and dim > 1:
                    output_dim = dim
                    break
            if output_dim <= 0:
                output_dim = 192

        self.meta = EmbeddingMeta(
            output_dim=output_dim,
            sample_rate=sample_rate,
            normalize_samples=normalize_samples,
            feature_normalize_type=feature_normalize_type,
            framework=framework,
            language=(raw.get("language") or "").strip(),
        )
        self.input_name = self.handle.get_inputs()[0].name
        self.output_name = self.handle.get_outputs()[0].name

    def compute(self, waveform: np.ndarray) -> np.ndarray:
        audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if audio.size < self.meta.sample_rate // 10:
            raise ValueError("音频过短，无法提取声纹")

        feats = compute_fbank(
            audio,
            sample_rate=self.meta.sample_rate,
            normalize_samples=self.meta.normalize_samples,
        )
        if feats.shape[0] < 2:
            raise ValueError("有效帧过少，无法提取声纹")

        if self.meta.feature_normalize_type == "global-mean":
            feats = subtract_global_mean(feats)

        x = np.ascontiguousarray(feats[None, :, :], dtype=np.float32)
        (emb,) = self.handle.run([self.output_name], {self.input_name: x})
        self.providers = list(self.handle.providers)
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        return emb
