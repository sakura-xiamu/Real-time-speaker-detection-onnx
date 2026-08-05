"""声纹注册与余弦相似度比对。"""
from __future__ import annotations

from typing import Any

import numpy as np

from server import config
from server.engine.diarization import DiarizationEngine, DiarizationSegment


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def energy_vad_filter(audio: np.ndarray, sample_rate: int = config.SAMPLE_RATE) -> np.ndarray:
    """基于帧能量的简易 VAD，过滤静音。"""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio

    frame = int(sample_rate * config.ENERGY_VAD_FRAME_MS / 1000)
    hop = int(sample_rate * config.ENERGY_VAD_HOP_MS / 1000)
    if frame < 1 or hop < 1 or len(audio) < frame:
        return np.ascontiguousarray(audio)

    energies = []
    for i in range(0, len(audio) - frame + 1, hop):
        chunk = audio[i : i + frame]
        energies.append(float(np.sqrt(np.mean(chunk * chunk))))
    if not energies:
        return np.ascontiguousarray(audio)

    energies_arr = np.asarray(energies, dtype=np.float32)
    thr = float(np.max(energies_arr) * config.ENERGY_VAD_THRESHOLD_RATIO)
    thr = max(thr, 1e-4)

    keep = np.zeros(len(audio), dtype=bool)
    for idx, e in enumerate(energies_arr):
        if e >= thr:
            start = idx * hop
            end = min(len(audio), start + frame)
            keep[start:end] = True

    filtered = audio[keep]
    if filtered.size < int(0.5 * sample_rate):
        return np.ascontiguousarray(audio)
    return np.ascontiguousarray(filtered)


class VoiceProfileStore:
    def __init__(self) -> None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.embedding: np.ndarray | None = self._load()

    def _load(self) -> np.ndarray | None:
        if not config.ENROLLMENT_FILE.is_file():
            return None
        emb = np.load(config.ENROLLMENT_FILE)
        return np.asarray(emb, dtype=np.float32).reshape(-1)

    def has_enrollment(self) -> bool:
        return self.embedding is not None

    def clear(self) -> None:
        self.embedding = None
        if config.ENROLLMENT_FILE.is_file():
            config.ENROLLMENT_FILE.unlink()

    def enroll(self, engine: DiarizationEngine, audio: np.ndarray) -> dict[str, Any]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = len(audio) / config.SAMPLE_RATE
        if duration < config.MIN_ENROLLMENT_DURATION_SEC:
            raise ValueError(
                f"注册音频太短（{duration:.1f}s），请至少录制 "
                f"{config.MIN_ENROLLMENT_DURATION_SEC}s"
            )
        filtered = energy_vad_filter(audio)
        speech_sec = len(filtered) / config.SAMPLE_RATE
        if speech_sec < config.MIN_ENROLLMENT_DURATION_SEC:
            raise ValueError(
                f"有效语音仅 {speech_sec:.1f}s，请清晰说话至少 "
                f"{config.MIN_ENROLLMENT_DURATION_SEC}s"
            )
        emb = engine.compute_embedding(filtered)
        self.embedding = emb
        np.save(config.ENROLLMENT_FILE, emb)
        return {
            "enrolled": True,
            "duration_sec": round(duration, 2),
            "speech_sec": round(speech_sec, 2),
            "message": "声纹注册成功",
        }

    def compare_window(
        self,
        engine: DiarizationEngine,
        audio: np.ndarray,
        segments: list[DiarizationSegment],
        threshold: float,
    ) -> dict[str, Any]:
        thr = float(threshold)
        if self.embedding is None:
            return {
                "enrolled": False,
                "similarity": None,
                "is_me": None,
                "threshold": thr,
                "speakers": [],
            }

        sr = config.SAMPLE_RATE
        per_speaker: dict[int, list[tuple[float, float]]] = {}
        for seg in segments:
            per_speaker.setdefault(seg.speaker, []).append((seg.start, seg.end))

        speakers_result: list[dict[str, Any]] = []
        best_score = -1.0
        best_is_me = False

        for spk, ranges in per_speaker.items():
            chunks: list[np.ndarray] = []
            total = 0.0
            for start, end in ranges:
                s = max(0, int(start * sr))
                e = min(len(audio), int(end * sr))
                if e > s:
                    chunks.append(audio[s:e])
                    total += (e - s) / sr

            if total < config.MIN_SPEAKER_EMBEDDING_SEC:
                speakers_result.append(
                    {
                        "speaker": spk,
                        "speech_sec": round(total, 2),
                        "similarity": None,
                        "is_me": None,
                        "skipped": True,
                    }
                )
                continue

            try:
                spk_audio = np.concatenate(chunks)
                emb = engine.compute_embedding(spk_audio)
                score = cosine_similarity(emb, self.embedding)
                is_me = score >= thr
                speakers_result.append(
                    {
                        "speaker": spk,
                        "speech_sec": round(total, 2),
                        "similarity": float(score),
                        "is_me": bool(is_me),
                        "skipped": False,
                    }
                )
                if score > best_score:
                    best_score = score
                    best_is_me = is_me
            except Exception:
                speakers_result.append(
                    {
                        "speaker": spk,
                        "speech_sec": round(total, 2),
                        "similarity": None,
                        "is_me": None,
                        "skipped": False,
                    }
                )

        if best_score < 0:
            return {
                "enrolled": True,
                "similarity": None,
                "is_me": None,
                "threshold": thr,
                "speakers": speakers_result,
            }
        return {
            "enrolled": True,
            "similarity": float(best_score),
            "is_me": bool(best_is_me),
            "threshold": thr,
            "speakers": speakers_result,
        }
