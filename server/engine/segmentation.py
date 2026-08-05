"""pyannote segmentation 3.0 ONNX 前后处理（对齐 sherpa-onnx 离线逻辑）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.lib.stride_tricks import as_strided

from .providers import OrtSessionHandle


@dataclass
class SegmentationMeta:
    window_size: int
    sample_rate: int
    window_shift: int
    receptive_field_size: int
    receptive_field_shift: int
    num_speakers: int
    powerset_max_classes: int
    num_classes: int


def get_powerset_mapping(
    num_classes: int, num_speakers: int, powerset_max_classes: int
) -> np.ndarray:
    mapping = np.zeros((num_classes, num_speakers), dtype=np.float32)
    k = 1
    for i in range(1, powerset_max_classes + 1):
        if i == 1:
            for j in range(num_speakers):
                mapping[k, j] = 1
                k += 1
        elif i == 2:
            for j in range(num_speakers):
                for m in range(j + 1, num_speakers):
                    mapping[k, j] = 1
                    mapping[k, m] = 1
                    k += 1
        else:
            raise RuntimeError(f"Unsupported powerset_max_classes={powerset_max_classes}")
    return mapping


def to_multi_label(y: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    y_idx = np.argmax(y, axis=-1)
    return mapping[y_idx.reshape(-1)].reshape(y_idx.shape[0], y_idx.shape[1], -1)


class SegmentationModel:
    def __init__(
        self,
        model_path: str,
        use_gpu: bool = True,
        on_fallback: Callable[[str], None] | None = None,
    ) -> None:
        self.handle = OrtSessionHandle(
            model_path, use_gpu=use_gpu, on_fallback=on_fallback
        )
        self.providers = list(self.handle.providers)
        meta = self.handle.get_modelmeta().custom_metadata_map
        self.meta = SegmentationMeta(
            window_size=int(meta["window_size"]),
            sample_rate=int(meta["sample_rate"]),
            window_shift=int(0.1 * int(meta["window_size"])),
            receptive_field_size=int(meta["receptive_field_size"]),
            receptive_field_shift=int(meta["receptive_field_shift"]),
            num_speakers=int(meta["num_speakers"]),
            powerset_max_classes=int(meta["powerset_max_classes"]),
            num_classes=int(meta["num_classes"]),
        )
        self.input_name = self.handle.get_inputs()[0].name
        self.output_name = self.handle.get_outputs()[0].name
        self.mapping = get_powerset_mapping(
            self.meta.num_classes,
            self.meta.num_speakers,
            self.meta.powerset_max_classes,
        )

    def _infer_batch(self, samples: np.ndarray) -> np.ndarray:
        """samples: (N, window_size) -> (N, num_frames, num_classes)"""
        x = np.expand_dims(np.ascontiguousarray(samples, dtype=np.float32), axis=1)
        (y,) = self.handle.run([self.output_name], {self.input_name: x})
        self.providers = list(self.handle.providers)
        return y

    def run(self, audio: np.ndarray, batch_size: int = 32) -> tuple[np.ndarray, bool]:
        """
        Returns:
            labels: (num_chunks, num_frames, num_speakers)
            has_last_chunk: bool
        """
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        m = self.meta
        if audio.shape[0] < m.window_size:
            pad = m.window_size - audio.shape[0]
            chunk = np.pad(audio, (0, pad))
            y = self._infer_batch(chunk[None, :])
            labels = to_multi_label(y, self.mapping)
            return labels, True

        num = (audio.shape[0] - m.window_size) // m.window_shift + 1
        samples = as_strided(
            audio,
            shape=(num, m.window_size),
            strides=(m.window_shift * audio.strides[0], audio.strides[0]),
        )
        has_last_chunk = (
            audio.shape[0] < m.window_size
            or (audio.shape[0] - m.window_size) % m.window_shift > 0
        )

        outputs = []
        for i in range(0, samples.shape[0], batch_size):
            outputs.append(self._infer_batch(np.ascontiguousarray(samples[i : i + batch_size])))

        if has_last_chunk:
            last = audio[num * m.window_shift :]
            pad_size = m.window_size - last.shape[0]
            last = np.pad(last, (0, pad_size))[None, :]
            outputs.append(self._infer_batch(last))

        y = np.vstack(outputs)
        labels = to_multi_label(y, self.mapping)
        return labels, has_last_chunk


def speaker_count_per_frame(labels: np.ndarray, meta: SegmentationMeta) -> np.ndarray:
    """labels: (num_chunks, num_frames, num_speakers) -> (num_total_frames,)"""
    counts = labels.sum(axis=-1)
    num_frames = (
        int(
            (meta.window_size + (labels.shape[0] - 1) * meta.window_shift)
            / meta.receptive_field_shift
        )
        + 1
    )
    ans = np.zeros((num_frames,), dtype=np.float64)
    weight = np.zeros((num_frames,), dtype=np.float64)
    for i in range(labels.shape[0]):
        this_chunk = counts[i]
        start = int(i * meta.window_shift / meta.receptive_field_shift + 0.5)
        end = start + this_chunk.shape[0]
        ans[start:end] += this_chunk
        weight[start:end] += 1
    ans /= np.maximum(weight, 1e-12)
    return (ans + 0.5).astype(np.int8)
