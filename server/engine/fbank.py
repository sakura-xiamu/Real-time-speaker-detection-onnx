"""Kaldi-style 80-dim fbank，与 sherpa-onnx / 3d-speaker 约定对齐。"""
from __future__ import annotations

import numpy as np

try:
    import kaldi_native_fbank as knf
except ImportError:  # pragma: no cover
    knf = None


def compute_fbank(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    dither: float = 0.0,
    normalize_samples: bool = True,
) -> np.ndarray:
    """
    计算 fbank 特征。

    Args:
        waveform: float32 一维音频。normalize_samples=True 时期望 [-1, 1]；
                  False（wespeaker）时期望接近 int16 量级（×32768）。
        sample_rate: 采样率
        num_mel_bins: mel 滤波器数量（默认 80）
        dither: 抖动（说话人模型通常为 0）
        normalize_samples: 是否按 float 波形处理

    Returns:
        (num_frames, num_mel_bins) float32
    """
    if knf is None:
        raise ImportError("请安装 kaldi-native-fbank: pip install kaldi-native-fbank")

    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return np.zeros((0, num_mel_bins), dtype=np.float32)

    if not normalize_samples:
        # wespeaker: sherpa 在 normalize_samples=0 时接受 int16 量级输入
        peak = float(np.max(np.abs(audio)))
        if peak <= 1.5:
            audio = audio * 32768.0

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = float(sample_rate)
    opts.frame_opts.dither = float(dither)
    opts.frame_opts.snip_edges = True
    opts.frame_opts.window_type = "hamming"
    opts.mel_opts.num_bins = int(num_mel_bins)
    opts.mel_opts.low_freq = 20
    opts.mel_opts.high_freq = -400
    opts.use_energy = False

    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sample_rate, audio.tolist())
    fbank.input_finished()

    frames = []
    for i in range(fbank.num_frames_ready):
        frames.append(fbank.get_frame(i))
    if not frames:
        return np.zeros((0, num_mel_bins), dtype=np.float32)
    return np.asarray(frames, dtype=np.float32)


def subtract_global_mean(features: np.ndarray) -> np.ndarray:
    """按列减去全局均值（3d-speaker feature_normalize_type=global-mean）。"""
    if features.size == 0:
        return features
    mean = np.mean(features, axis=0, keepdims=True)
    return (features - mean).astype(np.float32)
