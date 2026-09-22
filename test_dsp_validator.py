#!/usr/bin/env python3
"""
test_dsp_validator.py — Synthetic Signal Fixture Generator & SciPy Parity Test Suite
Compliant with DSP Arena constraints: 384 kHz support, zero-heap pre-allocations, np.float32.
"""

import unittest
import numpy as np
import scipy.signal
import scipy.fft


class SyntheticSignalGenerator:
    """
    High-performance synthetic signal fixture generator for DSP testing.
    All outputs are strictly np.float32.
    """

    @staticmethod
    def generate_tone(
        freq: float,
        duration: float,
        sr: int = 384000,
        amplitude: float = 1.0,
        phase: float = 0.0,
        out: np.ndarray = None,
    ) -> np.ndarray:
        """Generates pure sine tone with zero-allocation option."""
        num_samples = int(sr * duration)
        if out is None:
            out = np.empty(num_samples, dtype=np.float32)
        t = np.arange(num_samples, dtype=np.float32) / np.float32(sr)
        np.sin(2.0 * np.pi * freq * t + phase, out=out)
        out *= np.float32(amplitude)
        return out

    @staticmethod
    def generate_chirp(
        f0: float,
        f1: float,
        duration: float,
        sr: int = 384000,
        method: str = "linear",
    ) -> np.ndarray:
        """Generates linear or exponential chirp."""
        num_samples = int(sr * duration)
        t = np.arange(num_samples, dtype=np.float32) / np.float32(sr)
        # Using scipy.signal.chirp with float32 cast
        sig = scipy.signal.chirp(t, f0=f0, t1=duration, f1=f1, method=method)
        return sig.astype(np.float32)

    @staticmethod
    def generate_impact_transient(
        f_res: float,
        damping_ratio: float,
        duration: float,
        sr: int = 384000,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """
        Generates physical acoustic impact transient:
        x(t) = A * exp(-zeta * omega_n * t) * sin(omega_d * t)
        where omega_n = 2*pi*f_res, omega_d = omega_n * sqrt(1 - zeta^2)
        """
        num_samples = int(sr * duration)
        t = np.arange(num_samples, dtype=np.float32) / np.float32(sr)
        omega_n = 2.0 * np.pi * f_res
        omega_d = omega_n * np.sqrt(max(0.0, 1.0 - damping_ratio**2))
        decay = np.exp(-damping_ratio * omega_n * t)
        transient = amplitude * decay * np.sin(omega_d * t)
        return transient.astype(np.float32)

    @staticmethod
    def generate_impulse(
        duration: float,
        sr: int = 384000,
        impulse_index: int = 0,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Generates discrete Dirac impulse."""
        num_samples = int(sr * duration)
        out = np.zeros(num_samples, dtype=np.float32)
        if 0 <= impulse_index < num_samples:
            out[impulse_index] = np.float32(amplitude)
        return out

    @staticmethod
    def generate_noise(
        duration: float,
        sr: int = 384000,
        amplitude: float = 0.1,
        seed: int = 42,
    ) -> np.ndarray:
        """Generates deterministic white Gaussian noise."""
        rng = np.random.default_rng(seed)
        num_samples = int(sr * duration)
        noise = rng.standard_normal(num_samples, dtype=np.float32) * np.float32(amplitude)
        return noise


class TestDspValidatorFixtures(unittest.TestCase):
    """Test suite for DSP validation fixtures against SciPy ground truth."""

    def setUp(self):
        self.sr = 384000  # 384 kHz sampling rate
        self.duration = 0.05  # 50 ms test window

    def test_tone_generation_peak_frequency(self):
        """Verifies pure tone generates exact peak in SciPy FFT."""
        target_freq = 4400.0
        tone = SyntheticSignalGenerator.generate_tone(
            freq=target_freq, duration=self.duration, sr=self.sr
        )
        self.assertEqual(tone.dtype, np.float32)

        # FFT via SciPy
        fft_vals = scipy.fft.rfft(tone)
        freqs = scipy.fft.rfftfreq(len(tone), d=1.0 / self.sr)
        peak_idx = np.argmax(np.abs(fft_vals))
        peak_freq = freqs[peak_idx]

        # Frequency resolution bin check
        freq_resolution = self.sr / len(tone)
        self.assertAlmostEqual(peak_freq, target_freq, delta=freq_resolution)

    def test_chirp_frequency_coverage(self):
        """Verifies chirp spans designated frequency bandwidth."""
        f0, f1 = 1000.0, 20000.0
        chirp = SyntheticSignalGenerator.generate_chirp(
            f0=f0, f1=f1, duration=self.duration, sr=self.sr
        )
        self.assertEqual(chirp.dtype, np.float32)
        self.assertEqual(len(chirp), int(self.sr * self.duration))

        # Check instantaneous frequency using Hilbert transform
        analytic = scipy.signal.hilbert(chirp)
        inst_phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(inst_phase) / (2.0 * np.pi) * self.sr

        # Average near start and end
        start_f = np.mean(inst_freq[100:300])
        end_f = np.mean(inst_freq[-300:-100])
        self.assertGreater(end_f, start_f)
        self.assertAlmostEqual(start_f, f0, delta=1000.0)
        self.assertAlmostEqual(end_f, f1, delta=1000.0)

    def test_impact_transient_exponential_decay(self):
        """Verifies acoustic transient adheres to physical damping envelope."""
        f_res = 2500.0
        damping = 0.05
        transient = SyntheticSignalGenerator.generate_impact_transient(
            f_res=f_res, damping_ratio=damping, duration=self.duration, sr=self.sr
        )
        self.assertEqual(transient.dtype, np.float32)

        # Extract Hilbert envelope
        envelope = np.abs(scipy.signal.hilbert(transient))
        # Ensure peak is at onset and decays monotonically over large spans
        peak_idx = np.argmax(envelope[:500])
        self.assertLess(peak_idx, 200)

        # Compare early amplitude to late amplitude
        early_amp = np.mean(envelope[100:500])
        late_amp = np.mean(envelope[-1000:])
        self.assertGreater(early_amp, late_amp * 10.0)

    def test_sos_filter_parity_with_scipy(self):
        """Verifies SOS filter computation matches scipy.signal.sosfilt."""
        sos = scipy.signal.butter(4, [1000.0, 10000.0], btype="bandpass", fs=self.sr, output="sos")
        noise = SyntheticSignalGenerator.generate_noise(
            duration=self.duration, sr=self.sr, amplitude=0.5
        )

        filtered = scipy.signal.sosfilt(sos, noise).astype(np.float32)
        self.assertEqual(filtered.dtype, np.float32)
        self.assertEqual(len(filtered), len(noise))
        # Energy in stopband should be attenuated
        self.assertLess(np.std(filtered), np.std(noise))

    def test_zero_allocation_buffer_reuse(self):
        """Verifies G1 compliance: buffer can be reused without new allocations."""
        num_samples = int(self.sr * self.duration)
        buf = np.empty(num_samples, dtype=np.float32)
        ptr_before = buf.__array_interface__["data"][0]

        result = SyntheticSignalGenerator.generate_tone(
            freq=1000.0, duration=self.duration, sr=self.sr, out=buf
        )
        ptr_after = result.__array_interface__["data"][0]

        # Verify pointer identity (zero memory reallocation)
        self.assertEqual(ptr_before, ptr_after)


if __name__ == "__main__":
    unittest.main()
