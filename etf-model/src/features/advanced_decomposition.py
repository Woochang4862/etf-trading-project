"""
Advanced Time Series Decomposition Features for ETF Stock Prediction

Includes:
- Fourier Transform features (frequency domain analysis)
- Wavelet Transform features (multi-scale analysis)
- Hilbert Transform features (instantaneous frequency/phase)
- Singular Spectrum Analysis (SSA) inspired features
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


ADVANCED_DECOMPOSITION_FEATURES: List[str] = [
    "fft_dominant_period",
    "fft_spectral_entropy",
    "fft_power_ratio_short",
    "fft_power_ratio_medium",
    "fft_power_ratio_long",
    "fft_spectral_centroid",
    "fft_spectral_spread",
    "fft_spectral_rolloff",
    "wavelet_energy_d1",
    "wavelet_energy_d2",
    "wavelet_energy_d3",
    "wavelet_energy_ratio",
    "wavelet_detail_trend",
    "hilbert_instant_amplitude",
    "hilbert_instant_phase",
    "hilbert_phase_velocity",
    "ssa_trend_strength",
    "ssa_cycle_strength",
    "ssa_noise_ratio",
    "trend_cycle_ratio",
    "spectral_flatness",
    "autocorr_decay_rate",
    "periodicity_score",
]


def _compute_fft_features(series: np.ndarray, window: int = 63) -> dict:
    """Compute FFT-based features for a price series"""
    if len(series) < window:
        return {
            "dominant_period": np.nan,
            "spectral_entropy": np.nan,
            "power_ratio_short": np.nan,
            "power_ratio_medium": np.nan,
            "power_ratio_long": np.nan,
            "spectral_centroid": np.nan,
            "spectral_spread": np.nan,
            "spectral_rolloff": np.nan,
        }

    data = series[-window:]
    data = data - np.mean(data)

    fft_vals = np.fft.rfft(data)
    power_spectrum = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(window)

    total_power = np.sum(power_spectrum[1:])
    if total_power == 0:
        return {
            "dominant_period": np.nan,
            "spectral_entropy": np.nan,
            "power_ratio_short": np.nan,
            "power_ratio_medium": np.nan,
            "power_ratio_long": np.nan,
            "spectral_centroid": np.nan,
            "spectral_spread": np.nan,
            "spectral_rolloff": np.nan,
        }

    dominant_idx = np.argmax(power_spectrum[1:]) + 1
    dominant_freq = freqs[dominant_idx]
    dominant_period = 1 / dominant_freq if dominant_freq > 0 else window

    norm_power = power_spectrum[1:] / total_power
    norm_power = np.clip(norm_power, 1e-10, 1)
    spectral_entropy = -np.sum(norm_power * np.log(norm_power))

    short_term_idx = (freqs >= 1 / 10) & (freqs < 1 / 2)
    medium_term_idx = (freqs >= 1 / 30) & (freqs < 1 / 10)
    long_term_idx = (freqs >= 1 / window) & (freqs < 1 / 30)

    power_ratio_short = (
        np.sum(power_spectrum[short_term_idx]) / total_power
        if np.any(short_term_idx)
        else 0
    )
    power_ratio_medium = (
        np.sum(power_spectrum[medium_term_idx]) / total_power
        if np.any(medium_term_idx)
        else 0
    )
    power_ratio_long = (
        np.sum(power_spectrum[long_term_idx]) / total_power
        if np.any(long_term_idx)
        else 0
    )

    spectral_centroid = np.sum(freqs[1:] * power_spectrum[1:]) / total_power
    spectral_spread = np.sqrt(
        np.sum(((freqs[1:] - spectral_centroid) ** 2) * power_spectrum[1:])
        / total_power
    )

    cumsum_power = np.cumsum(power_spectrum[1:])
    rolloff_idx = np.searchsorted(cumsum_power, 0.85 * total_power)
    spectral_rolloff = (
        freqs[rolloff_idx + 1] if rolloff_idx + 1 < len(freqs) else freqs[-1]
    )

    return {
        "dominant_period": dominant_period,
        "spectral_entropy": spectral_entropy,
        "power_ratio_short": power_ratio_short,
        "power_ratio_medium": power_ratio_medium,
        "power_ratio_long": power_ratio_long,
        "spectral_centroid": spectral_centroid,
        "spectral_spread": spectral_spread,
        "spectral_rolloff": spectral_rolloff,
    }


def _haar_wavelet_decompose(series: np.ndarray, levels: int = 3) -> List[np.ndarray]:
    """Simple Haar wavelet decomposition without external dependencies"""
    if len(series) < 2**levels:
        return [np.array([np.nan])] * (levels + 1)

    approximation = series.copy()
    details = []

    for _ in range(levels):
        n = len(approximation)
        if n < 2:
            details.append(np.array([np.nan]))
            continue

        n = n - (n % 2)
        approximation = approximation[:n]

        new_approx = (approximation[::2] + approximation[1::2]) / 2
        detail = (approximation[::2] - approximation[1::2]) / 2

        details.append(detail)
        approximation = new_approx

    return [approximation] + details


def _compute_wavelet_features(series: np.ndarray, window: int = 64) -> dict:
    """Compute wavelet-based features"""
    if len(series) < window:
        return {
            "energy_d1": np.nan,
            "energy_d2": np.nan,
            "energy_d3": np.nan,
            "energy_ratio": np.nan,
            "detail_trend": np.nan,
        }

    data = series[-window:]
    coeffs = _haar_wavelet_decompose(data, levels=3)

    energies = [
        np.sum(c**2) if len(c) > 0 and not np.isnan(c).all() else 0 for c in coeffs
    ]
    total_energy = sum(energies)

    if total_energy == 0:
        return {
            "energy_d1": np.nan,
            "energy_d2": np.nan,
            "energy_d3": np.nan,
            "energy_ratio": np.nan,
            "detail_trend": np.nan,
        }

    detail_trend = 0
    if len(coeffs) >= 3 and len(coeffs[1]) > 1 and len(coeffs[2]) > 1:
        d1_mean = np.nanmean(np.abs(coeffs[1]))
        d2_mean = np.nanmean(np.abs(coeffs[2]))
        if d2_mean > 0:
            detail_trend = d1_mean / d2_mean

    return {
        "energy_d1": energies[1] / total_energy if len(energies) > 1 else np.nan,
        "energy_d2": energies[2] / total_energy if len(energies) > 2 else np.nan,
        "energy_d3": energies[3] / total_energy if len(energies) > 3 else np.nan,
        "energy_ratio": energies[0] / total_energy if len(energies) > 0 else np.nan,
        "detail_trend": detail_trend,
    }


def _hilbert_transform(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Hilbert transform using FFT (analytic signal)"""
    n = len(series)
    if n < 4:
        return np.full(n, np.nan), np.full(n, np.nan)

    fft_vals = np.fft.fft(series)

    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2

    analytic = np.fft.ifft(fft_vals * h)
    amplitude = np.abs(analytic)
    phase = np.angle(analytic)

    return amplitude, phase


def _compute_hilbert_features(series: np.ndarray, window: int = 63) -> dict:
    """Compute Hilbert transform features"""
    if len(series) < window:
        return {
            "instant_amplitude": np.nan,
            "instant_phase": np.nan,
            "phase_velocity": np.nan,
        }

    data = series[-window:]
    detrended = data - np.linspace(data[0], data[-1], len(data))

    amplitude, phase = _hilbert_transform(detrended)

    return {
        "instant_amplitude": amplitude[-1] / (np.std(data) + 1e-10),
        "instant_phase": phase[-1],
        "phase_velocity": np.mean(np.diff(np.unwrap(phase))),
    }


def _compute_ssa_features(
    series: np.ndarray, window: int = 20, num_components: int = 5
) -> dict:
    """Compute SSA-inspired features using SVD decomposition"""
    if len(series) < window + num_components:
        return {
            "trend_strength": np.nan,
            "cycle_strength": np.nan,
            "noise_ratio": np.nan,
        }

    data = series[-(window + num_components) :]

    n = len(data)
    k = n - window + 1

    if k < 2:
        return {
            "trend_strength": np.nan,
            "cycle_strength": np.nan,
            "noise_ratio": np.nan,
        }

    trajectory = np.zeros((window, k))
    for i in range(k):
        trajectory[:, i] = data[i : i + window]

    try:
        U, s, Vt = np.linalg.svd(trajectory, full_matrices=False)
        total_variance = np.sum(s**2)

        if total_variance == 0:
            return {
                "trend_strength": np.nan,
                "cycle_strength": np.nan,
                "noise_ratio": np.nan,
            }

        trend_strength = s[0] ** 2 / total_variance
        cycle_strength = np.sum(s[1:3] ** 2) / total_variance if len(s) > 2 else 0
        noise_ratio = np.sum(s[3:] ** 2) / total_variance if len(s) > 3 else 0

        return {
            "trend_strength": trend_strength,
            "cycle_strength": cycle_strength,
            "noise_ratio": noise_ratio,
        }
    except Exception:
        return {
            "trend_strength": np.nan,
            "cycle_strength": np.nan,
            "noise_ratio": np.nan,
        }


def _compute_additional_spectral_features(series: np.ndarray, window: int = 63) -> dict:
    """Compute additional spectral and autocorrelation features"""
    if len(series) < window:
        return {
            "spectral_flatness": np.nan,
            "autocorr_decay_rate": np.nan,
            "periodicity_score": np.nan,
            "trend_cycle_ratio": np.nan,
        }

    data = series[-window:]

    fft_vals = np.fft.rfft(data - np.mean(data))
    power_spectrum = np.abs(fft_vals) ** 2

    geometric_mean = np.exp(np.mean(np.log(power_spectrum[1:] + 1e-10)))
    arithmetic_mean = np.mean(power_spectrum[1:])
    spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)

    autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    autocorr = autocorr / (autocorr[0] + 1e-10)

    decay_threshold = 0.5
    decay_idx = np.where(np.abs(autocorr) < decay_threshold)[0]
    autocorr_decay_rate = decay_idx[0] / window if len(decay_idx) > 0 else 1.0

    peaks = []
    for i in range(1, len(autocorr) - 1):
        if (
            autocorr[i] > autocorr[i - 1]
            and autocorr[i] > autocorr[i + 1]
            and autocorr[i] > 0.3
        ):
            peaks.append(i)
    periodicity_score = len(peaks) / (window / 2)

    sma_short = np.mean(data[-10:])
    sma_long = np.mean(data)
    cycle_var = np.var(data - np.linspace(data[0], data[-1], len(data)))
    trend_var = np.var(np.linspace(data[0], data[-1], len(data)))
    trend_cycle_ratio = trend_var / (cycle_var + 1e-10)

    return {
        "spectral_flatness": spectral_flatness,
        "autocorr_decay_rate": autocorr_decay_rate,
        "periodicity_score": periodicity_score,
        "trend_cycle_ratio": trend_cycle_ratio,
    }


def add_advanced_decomposition_features(
    df: pd.DataFrame, window: int = 63
) -> pd.DataFrame:
    """
    Add all advanced decomposition features

    Args:
        df: DataFrame with 'close' price column
        window: Window size for feature computation

    Returns:
        DataFrame with additional decomposition features
    """
    close = df["close"].values
    n = len(close)

    fft_features = {
        k: np.full(n, np.nan)
        for k in [
            "fft_dominant_period",
            "fft_spectral_entropy",
            "fft_power_ratio_short",
            "fft_power_ratio_medium",
            "fft_power_ratio_long",
            "fft_spectral_centroid",
            "fft_spectral_spread",
            "fft_spectral_rolloff",
        ]
    }

    wavelet_features = {
        k: np.full(n, np.nan)
        for k in [
            "wavelet_energy_d1",
            "wavelet_energy_d2",
            "wavelet_energy_d3",
            "wavelet_energy_ratio",
            "wavelet_detail_trend",
        ]
    }

    hilbert_features = {
        k: np.full(n, np.nan)
        for k in [
            "hilbert_instant_amplitude",
            "hilbert_instant_phase",
            "hilbert_phase_velocity",
        ]
    }

    ssa_features = {
        k: np.full(n, np.nan)
        for k in ["ssa_trend_strength", "ssa_cycle_strength", "ssa_noise_ratio"]
    }

    spectral_features = {
        k: np.full(n, np.nan)
        for k in [
            "spectral_flatness",
            "autocorr_decay_rate",
            "periodicity_score",
            "trend_cycle_ratio",
        ]
    }

    step = max(1, window // 4)
    compute_indices = list(range(window, n, step)) + [n - 1]

    for i in compute_indices:
        series = close[: i + 1]

        fft_result = _compute_fft_features(series, window)
        for k, v in fft_result.items():
            fft_features[f"fft_{k}"][i] = v

        wavelet_result = _compute_wavelet_features(series, min(64, window))
        for k, v in wavelet_result.items():
            wavelet_features[f"wavelet_{k}"][i] = v

        hilbert_result = _compute_hilbert_features(series, window)
        for k, v in hilbert_result.items():
            hilbert_features[f"hilbert_{k}"][i] = v

        ssa_result = _compute_ssa_features(series, min(20, window // 3), 5)
        for k, v in ssa_result.items():
            ssa_features[f"ssa_{k}"][i] = v

        spectral_result = _compute_additional_spectral_features(series, window)
        for k, v in spectral_result.items():
            spectral_features[k][i] = v

    for feature_dict in [
        fft_features,
        wavelet_features,
        hilbert_features,
        ssa_features,
        spectral_features,
    ]:
        for k, v in feature_dict.items():
            series = pd.Series(v)
            df[k] = series.interpolate(method="linear", limit_direction="both").values

    return df
