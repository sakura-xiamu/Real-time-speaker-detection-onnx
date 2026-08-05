"""离线说话人分离流水线：segmentation → embedding → FastClustering。"""
from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from server import config
from .clustering import fast_clustering
from .embedding import EmbeddingExtractor
from .providers import decode_onnx_error, is_gpu_runtime_error
from .segmentation import SegmentationModel, speaker_count_per_frame

logger = logging.getLogger(__name__)


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def merge(self, other: "DiarizationSegment", gap: float) -> "DiarizationSegment | None":
        if self.speaker != other.speaker:
            return None
        if self.end < other.start and self.end + gap >= other.start:
            return DiarizationSegment(self.start, other.end, self.speaker)
        if other.end < self.start and other.end + gap >= self.start:
            return DiarizationSegment(other.start, self.end, self.speaker)
        return None


@dataclass
class DiarizationResult:
    num_speakers: int
    segments: list[DiarizationSegment]
    duration_sec: float
    active_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_speakers": self.num_speakers,
            "duration_sec": self.duration_sec,
            "active_ratio": self.active_ratio,
            "segments": [s.to_dict() for s in self.segments],
        }


def _merge_segments(segments: list[DiarizationSegment], min_duration_off: float) -> None:
    changed = True
    while changed:
        changed = False
        for i in range(len(segments) - 1):
            merged = segments[i].merge(segments[i + 1], gap=min_duration_off)
            if merged is None:
                continue
            del segments[i + 1]
            segments[i] = merged
            changed = True
            break


def _extract_chunk_speaker_embeddings(
    extractor: EmbeddingExtractor,
    audio: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    window_shift: int,
    sample_rate: int,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """
    labels: (num_chunks, num_frames, num_speakers)
    Returns chunk_speaker_pairs and embeddings matrix.
    """
    num_chunks, num_frames, num_speakers = labels.shape
    pairs: list[tuple[int, int]] = []
    embeddings: list[np.ndarray] = []
    buffer = np.empty(window_size, dtype=np.float32)
    last_runtime_error: BaseException | None = None
    runtime_fail_count = 0

    for i in range(num_chunks):
        labels_t = labels[i].T  # (num_speakers, num_frames)
        sample_offset = i * window_shift
        for j in range(num_speakers):
            frames = labels_t[j]
            if frames.sum() < 10:
                continue

            idx = 0
            start = None
            for k in range(num_frames):
                if frames[k] != 0:
                    if start is None:
                        start = k
                elif start is not None:
                    start_samples = int(start / num_frames * window_size) + sample_offset
                    end_samples = int(k / num_frames * window_size) + sample_offset
                    end_samples = min(end_samples, len(audio))
                    start_samples = max(0, start_samples)
                    n = end_samples - start_samples
                    if n > 0 and idx + n <= window_size:
                        buffer[idx : idx + n] = audio[start_samples:end_samples]
                        idx += n
                    start = None
            if start is not None:
                start_samples = int(start / num_frames * window_size) + sample_offset
                end_samples = int(num_frames / num_frames * window_size) + sample_offset
                end_samples = min(end_samples, len(audio))
                start_samples = max(0, start_samples)
                n = end_samples - start_samples
                if n > 0 and idx + n <= window_size:
                    buffer[idx : idx + n] = audio[start_samples:end_samples]
                    idx += n

            if idx < int(0.2 * sample_rate):
                continue
            try:
                emb = extractor.compute(buffer[:idx])
            except ValueError:
                continue
            except Exception as exc:  # noqa: BLE001
                # GPU 错误向上抛，让引擎整模回退 CPU 后整窗重试
                if is_gpu_runtime_error(exc):
                    raise
                runtime_fail_count += 1
                last_runtime_error = exc
                continue
            pairs.append((i, j))
            embeddings.append(emb)

    if not embeddings:
        if last_runtime_error is not None and runtime_fail_count > 0:
            raise RuntimeError(
                f"声纹提取全部失败: {decode_onnx_error(last_runtime_error)}"
            ) from last_runtime_error
        return [], np.zeros((0, 0), dtype=np.float32)
    return pairs, np.stack(embeddings, axis=0)


class DiarizationEngine:
    def __init__(
        self,
        segmentation_path: str,
        embedding_path: str,
        use_gpu: bool = True,
        cluster_threshold: float = config.DIARIZATION_CLUSTER_THRESHOLD,
        min_duration_on: float = config.DIARIZATION_MIN_DURATION_ON,
        min_duration_off: float = config.DIARIZATION_MIN_DURATION_OFF,
    ) -> None:
        self._fallback_lock = threading.RLock()
        self._fallback_reason: str | None = None
        self.segmentation = SegmentationModel(
            segmentation_path,
            use_gpu=use_gpu,
            on_fallback=self._on_peer_fallback,
        )
        self.extractor = EmbeddingExtractor(
            embedding_path,
            use_gpu=use_gpu,
            on_fallback=self._on_peer_fallback,
        )
        self.cluster_threshold = float(cluster_threshold)
        self.min_duration_on = float(min_duration_on)
        self.min_duration_off = float(min_duration_off)
        self.sample_rate = self.segmentation.meta.sample_rate
        self.providers = {
            "segmentation": list(self.segmentation.providers),
            "embedding": list(self.extractor.providers),
        }

    def _on_peer_fallback(self, reason: str) -> None:
        self.force_all_cpu(reason)

    def force_all_cpu(self, reason: str = "") -> bool:
        """将 segmentation + embedding 全部重建为 CPU。返回是否发生了回退。"""
        with self._fallback_lock:
            reason = reason or "GPU/DML 运行时失败"
            changed = False
            for handle in (self.segmentation.handle, self.extractor.handle):
                if handle.force_cpu(reason):
                    changed = True
            self.segmentation.providers = list(self.segmentation.handle.providers)
            self.extractor.providers = list(self.extractor.handle.providers)
            self._sync_providers()
            if changed:
                self._fallback_reason = reason
                logger.warning("DiarizationEngine 已全部回退 CPU: %s", reason)
            return changed

    @property
    def last_fallback_reason(self) -> str | None:
        return self._fallback_reason

    def _sync_providers(self) -> None:
        self.providers = {
            "segmentation": list(self.segmentation.providers),
            "embedding": list(self.extractor.providers),
        }

    def compute_embedding(self, audio: np.ndarray) -> np.ndarray:
        try:
            emb = self.extractor.compute(audio)
        except Exception as exc:  # noqa: BLE001
            if is_gpu_runtime_error(exc):
                self.force_all_cpu(decode_onnx_error(exc))
                emb = self.extractor.compute(audio)
            else:
                raise
        self._sync_providers()
        return emb

    def process(
        self,
        audio: np.ndarray,
        num_speakers: int = 0,
    ) -> DiarizationResult:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = len(audio) / float(self.sample_rate)
        if duration < 1.0:
            raise ValueError("分析音频太短，请至少录制 1 秒")

        try:
            return self._process_impl(audio, duration, num_speakers)
        except Exception as exc:  # noqa: BLE001
            if not is_gpu_runtime_error(exc):
                raise
            reason = decode_onnx_error(exc)
            self.force_all_cpu(reason)
            return self._process_impl(audio, duration, num_speakers)

    def _process_impl(
        self,
        audio: np.ndarray,
        duration: float,
        num_speakers: int,
    ) -> DiarizationResult:
        labels, has_last_chunk = self.segmentation.run(audio)
        self._sync_providers()
        inactive = (labels.sum(axis=1) == 0).astype(np.int8)
        speakers_per_frame = speaker_count_per_frame(labels, self.segmentation.meta)
        if int(speakers_per_frame.max()) == 0:
            return DiarizationResult(
                num_speakers=0,
                segments=[],
                duration_sec=duration,
                active_ratio=0.0,
            )

        meta = self.segmentation.meta
        pairs, embeddings = _extract_chunk_speaker_embeddings(
            self.extractor,
            audio,
            labels,
            window_size=meta.window_size,
            window_shift=meta.window_shift,
            sample_rate=meta.sample_rate,
        )
        self._sync_providers()
        if embeddings.size == 0:
            return DiarizationResult(
                num_speakers=0,
                segments=[],
                duration_sec=duration,
                active_ratio=0.0,
            )

        n_clusters = int(num_speakers) if int(num_speakers) > 0 else -1
        cluster_labels = fast_clustering(
            embeddings,
            num_clusters=n_clusters,
            threshold=self.cluster_threshold,
        )

        chunk_speaker_to_cluster: dict[tuple[int, int], int] = {}
        for (chunk_idx, speaker_idx), cluster_idx in zip(pairs, cluster_labels):
            if inactive[chunk_idx, speaker_idx] == 1:
                continue
            chunk_speaker_to_cluster[(chunk_idx, speaker_idx)] = int(cluster_idx)

        if not chunk_speaker_to_cluster:
            return DiarizationResult(
                num_speakers=0,
                segments=[],
                duration_sec=duration,
                active_ratio=0.0,
            )

        num_spk = max(chunk_speaker_to_cluster.values()) + 1
        relabels = np.zeros((labels.shape[0], labels.shape[1], num_spk), dtype=np.float32)
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                for k in range(labels.shape[2]):
                    key = (i, k)
                    if key not in chunk_speaker_to_cluster:
                        continue
                    t = chunk_speaker_to_cluster[key]
                    if labels[i, j, k] == 1:
                        relabels[i, j, t] = 1

        num_frames = (
            int(
                (meta.window_size + (relabels.shape[0] - 1) * meta.window_shift)
                / meta.receptive_field_shift
            )
            + 1
        )
        count = np.zeros((num_frames, relabels.shape[-1]), dtype=np.float32)
        for i in range(relabels.shape[0]):
            this_chunk = relabels[i]
            start = int(i * meta.window_shift / meta.receptive_field_shift + 0.5)
            end = start + this_chunk.shape[0]
            count[start:end] += this_chunk

        if has_last_chunk:
            stop_frame = int(audio.shape[0] / meta.receptive_field_shift)
            count = count[:stop_frame]
            speakers_per_frame = speakers_per_frame[:stop_frame]

        sorted_count = np.argsort(-count, axis=-1)
        final = np.zeros_like(count)
        for i, (c, sc) in enumerate(zip(speakers_per_frame, sorted_count)):
            for k in range(int(c)):
                if k < sc.shape[0]:
                    final[i, sc[k]] = 1

        segments = self._frames_to_segments(final, meta)
        speaker_ids = {s.speaker for s in segments}
        active = sum(max(0.0, s.end - s.start) for s in segments)
        active_ratio = float(active / duration) if duration > 0 else 0.0
        return DiarizationResult(
            num_speakers=len(speaker_ids),
            segments=segments,
            duration_sec=duration,
            active_ratio=min(1.0, active_ratio),
        )

    def _frames_to_segments(
        self, final: np.ndarray, meta
    ) -> list[DiarizationSegment]:
        onset = 0.5
        offset = 0.5
        scale = meta.receptive_field_shift / meta.sample_rate
        scale_offset = meta.receptive_field_size / meta.sample_rate * 0.5
        result: list[DiarizationSegment] = []

        frames_t = final.T
        for kk in range(frames_t.shape[0]):
            frames = frames_t[kk]
            segment_list: list[DiarizationSegment] = []
            is_active = bool(frames[0] > onset) if len(frames) else False
            start = 0 if is_active else None
            for i in range(1, len(frames)):
                if is_active:
                    if frames[i] < offset:
                        segment_list.append(
                            DiarizationSegment(
                                start=start * scale + scale_offset,
                                end=i * scale + scale_offset,
                                speaker=kk,
                            )
                        )
                        is_active = False
                else:
                    if frames[i] > onset:
                        start = i
                        is_active = True
            if is_active and start is not None:
                segment_list.append(
                    DiarizationSegment(
                        start=start * scale + scale_offset,
                        end=(len(frames) - 1) * scale + scale_offset,
                        speaker=kk,
                    )
                )

            if len(segment_list) > 1:
                _merge_segments(segment_list, self.min_duration_off)
            for s in segment_list:
                if s.duration < self.min_duration_on:
                    continue
                result.append(s)

        result.sort(key=lambda s: s.start)
        return result
