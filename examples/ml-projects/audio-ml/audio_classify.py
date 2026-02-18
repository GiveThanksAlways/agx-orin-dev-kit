#!/usr/bin/env python3
"""
Audio ML — Mel Spectrogram + Audio Classification on Jetson Orin AGX.

Demonstrates audio machine learning entirely in tinygrad:
  1. Mel spectrogram computation (STFT → mel filterbank) on GPU
  2. Audio classification CNN (mel spectrogram → class label)
  3. Real-time audio classification from microphone

The mel spectrogram is the standard audio representation for ML:
  - Short-Time Fourier Transform (STFT) → frequency content over time
  - Mel filterbank → compress frequencies to perceptual (mel) scale
  - Log scaling → compress dynamic range

This is what speech recognition (Whisper), music classification,
and sound event detection all use as input features.

Usage:
    NV=1 python3 audio_classify.py --bench
    NV=1 python3 audio_classify.py --file audio.wav
"""
import argparse, os, sys, time
import numpy as np

# Audio parameters
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
DURATION = 2.0  # seconds per clip

# Classes for environmental sound classification
SOUND_CLASSES = [
    "silence", "speech", "music", "dog_bark", "car_horn",
    "siren", "bird_chirp", "footsteps", "clapping", "typing"
]

def mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS):
    """
    Create mel-scale triangular filterbank matrix.

    Maps linear frequency bins (from FFT) to mel-scale bins.
    The mel scale approximates human frequency perception:
      mel = 2595 * log10(1 + f/700)
    """
    # Frequency range
    fmin, fmax = 0.0, sr / 2.0

    # Mel scale points
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    # FFT bin centers
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2.0, n_freqs)

    # Build triangular filters
    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        low, center, high = hz_points[i], hz_points[i+1], hz_points[i+2]
        # Rising slope
        mask_up = (fft_freqs >= low) & (fft_freqs <= center)
        if center != low:
            filterbank[i, mask_up] = (fft_freqs[mask_up] - low) / (center - low)
        # Falling slope
        mask_down = (fft_freqs > center) & (fft_freqs <= high)
        if high != center:
            filterbank[i, mask_down] = (high - fft_freqs[mask_down]) / (high - center)

    return filterbank

def compute_mel_spectrogram_numpy(audio, sr=SAMPLE_RATE, n_fft=N_FFT,
                                   hop_length=HOP_LENGTH, n_mels=N_MELS):
    """Compute mel spectrogram using NumPy (baseline)."""
    # Hann window
    window = np.hanning(n_fft).astype(np.float32)

    # STFT
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        stft[:, i] = spectrum

    # Power spectrum
    power = np.abs(stft) ** 2

    # Mel filterbank
    fb = mel_filterbank(sr, n_fft, n_mels)
    mel = fb @ power

    # Log scale (floor to avoid log(0))
    mel_log = np.log(np.maximum(mel, 1e-10))
    return mel_log

def compute_mel_spectrogram_tinygrad(audio, sr=SAMPLE_RATE, n_fft=N_FFT,
                                      hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    Compute mel spectrogram using tinygrad — GPU-accelerated.

    The key insight: STFT is a batched matmul of windowed audio frames
    against the DFT basis, which maps perfectly to GPU tensor ops.
    """
    from tinygrad import Tensor, dtypes

    # Precompute DFT basis matrix (real part only for magnitude)
    n_freqs = n_fft // 2 + 1
    k = np.arange(n_freqs).reshape(-1, 1)  # [n_freqs, 1]
    n = np.arange(n_fft).reshape(1, -1)     # [1, n_fft]
    # DFT: X[k] = sum_n x[n] * exp(-j*2*pi*k*n/N)
    angles = -2.0 * np.pi * k * n / n_fft
    dft_real = np.cos(angles).astype(np.float32)  # [n_freqs, n_fft]
    dft_imag = np.sin(angles).astype(np.float32)

    # Hann window
    window = np.hanning(n_fft).astype(np.float32)

    # Frame the audio
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    frames = np.zeros((n_frames, n_fft), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frames[i] = audio[start:start + n_fft] * window

    # Upload to GPU
    frames_t = Tensor(frames)                    # [n_frames, n_fft]
    dft_real_t = Tensor(dft_real)                # [n_freqs, n_fft]
    dft_imag_t = Tensor(dft_imag)

    # STFT via matmul: X = frames @ DFT^T → [n_frames, n_freqs]
    real_part = frames_t.matmul(dft_real_t.permute(1, 0))
    imag_part = frames_t.matmul(dft_imag_t.permute(1, 0))

    # Power spectrum: |X|^2 = real^2 + imag^2
    power = (real_part * real_part + imag_part * imag_part).permute(1, 0)  # [n_freqs, n_frames]

    # Mel filterbank (matmul)
    fb_t = Tensor(mel_filterbank(sr, n_fft, n_mels))  # [n_mels, n_freqs]
    mel = fb_t.matmul(power)                            # [n_mels, n_frames]

    # Log scale
    mel_log = (mel + 1e-10).log()

    return mel_log.numpy()

def build_audio_classifier(n_mels=N_MELS, n_classes=10):
    """
    Build an audio classification CNN.

    Input: mel spectrogram [1, 1, n_mels, n_frames]
    Output: class probabilities [1, n_classes]

    Architecture:
      - 4 conv layers (increasing channels, stride-2 downsampling)
      - Global average pooling
      - FC → softmax
    """
    from tinygrad import Tensor
    from tinygrad.nn import Conv2d, Linear

    class AudioClassifier:
        def __init__(self):
            self.conv1 = Conv2d(1, 16, (3, 3), stride=(2, 2), padding=1)
            self.conv2 = Conv2d(16, 32, (3, 3), stride=(2, 2), padding=1)
            self.conv3 = Conv2d(32, 64, (3, 3), stride=(2, 2), padding=1)
            self.conv4 = Conv2d(64, 128, (3, 3), stride=(2, 2), padding=1)
            self.fc = Linear(128, n_classes)

        def __call__(self, x):
            """x: [B, 1, n_mels, n_frames] → [B, n_classes]"""
            x = self.conv1(x).relu()
            x = self.conv2(x).relu()
            x = self.conv3(x).relu()
            x = self.conv4(x).relu()
            # Global average pooling
            x = x.mean(axis=(2, 3))  # [B, 128]
            x = self.fc(x)           # [B, n_classes]
            return x

        def predict(self, mel_spectrogram):
            """Convenience: mel_spec numpy → class name + confidence."""
            from tinygrad import Tensor
            # Ensure correct shape [1, 1, n_mels, n_frames]
            if mel_spectrogram.ndim == 2:
                mel_spectrogram = mel_spectrogram[np.newaxis, np.newaxis, :, :]
            x = Tensor(mel_spectrogram.astype(np.float32))
            logits = self(x)
            probs = logits.softmax(axis=1).numpy()[0]
            cls_id = np.argmax(probs)
            return SOUND_CLASSES[cls_id], float(probs[cls_id])

    return AudioClassifier()

def generate_test_audio(duration=DURATION, sr=SAMPLE_RATE, kind="tone"):
    """Generate synthetic test audio."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    if kind == "tone":
        # 440 Hz sine tone
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    elif kind == "chirp":
        # Frequency sweep 200 Hz → 4000 Hz
        freq = 200 + (4000 - 200) * t / duration
        audio = 0.5 * np.sin(2 * np.pi * np.cumsum(freq) / sr)
    elif kind == "noise":
        audio = np.random.randn(len(t)).astype(np.float32) * 0.3
    elif kind == "silence":
        audio = np.zeros(len(t), dtype=np.float32)
    else:
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    return audio

def main():
    parser = argparse.ArgumentParser(description="Audio ML on Jetson")
    parser.add_argument("--file", help="Audio file to classify")
    parser.add_argument("--bench", action="store_true", help="Benchmark mel spectrogram + CNN")
    parser.add_argument("--visualize", action="store_true", help="Save mel spectrogram as image")
    args = parser.parse_args()

    backend = "NV" if os.environ.get("NV") == "1" else \
              "CUDA" if os.environ.get("CUDA") == "1" else "CPU"

    if args.bench:
        print(f"╔══════════════════════════════════════════╗")
        print(f"║  Audio ML Benchmark — Backend: {backend:>4}      ║")
        print(f"╚══════════════════════════════════════════╝\n")

        audio = generate_test_audio(DURATION, SAMPLE_RATE, "chirp")
        n_samples = len(audio)
        print(f"Audio: {DURATION}s @ {SAMPLE_RATE}Hz = {n_samples} samples")
        print(f"STFT: n_fft={N_FFT}, hop={HOP_LENGTH}")
        print(f"Mel: {N_MELS} bands\n")

        # NumPy baseline
        print("=== Mel Spectrogram (NumPy CPU) ===")
        times = []
        for _ in range(10):
            t0 = time.time()
            mel_np = compute_mel_spectrogram_numpy(audio)
            times.append(time.time() - t0)
        print(f"  Shape: {mel_np.shape}")
        print(f"  Time: {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms")

        # Tinygrad GPU
        print(f"\n=== Mel Spectrogram (tinygrad {backend}) ===")
        # Warmup
        _ = compute_mel_spectrogram_tinygrad(audio)
        times = []
        for _ in range(10):
            t0 = time.time()
            mel_tg = compute_mel_spectrogram_tinygrad(audio)
            times.append(time.time() - t0)
        print(f"  Shape: {mel_tg.shape}")
        print(f"  Time: {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms")

        # Verify correctness
        if mel_np.shape == mel_tg.shape:
            corr = np.corrcoef(mel_np.flatten(), mel_tg.flatten())[0, 1]
            print(f"  Correlation with NumPy: {corr:.6f}")

        # Audio classifier CNN
        print(f"\n=== Audio Classifier CNN ({backend}) ===")
        model = build_audio_classifier()
        from tinygrad import Tensor
        x = Tensor.randn(1, 1, N_MELS, 125)  # ~2s at hop_length=256

        # Warmup
        for _ in range(3):
            model(x).numpy()

        times = []
        for _ in range(20):
            t0 = time.time()
            model(x).numpy()
            times.append(time.time() - t0)
        print(f"  Input: [1, 1, {N_MELS}, 125]")
        print(f"  Time: {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms")
        print(f"  FPS: {1.0/np.mean(times):.1f}")

        # End-to-end pipeline
        print(f"\n=== End-to-End Pipeline (mel + classify) ===")
        times = []
        for _ in range(10):
            t0 = time.time()
            mel = compute_mel_spectrogram_tinygrad(audio)
            cls_name, conf = model.predict(mel)
            times.append(time.time() - t0)
        print(f"  Time: {np.mean(times)*1000:.2f}ms")
        print(f"  Class: {cls_name} (conf={conf:.3f})")
        print(f"  Real-time factor: {DURATION / np.mean(times):.1f}x")

        return

    if args.file:
        try:
            import soundfile as sf
            audio, sr = sf.read(args.file, dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # Mono
            if sr != SAMPLE_RATE:
                # Simple resampling via linear interpolation
                old_len = len(audio)
                new_len = int(old_len * SAMPLE_RATE / sr)
                audio = np.interp(np.linspace(0, old_len - 1, new_len),
                                  np.arange(old_len), audio).astype(np.float32)
        except ImportError:
            print("Install soundfile: pip install soundfile")
            sys.exit(1)

        print(f"Audio: {len(audio)/SAMPLE_RATE:.2f}s @ {SAMPLE_RATE}Hz")

        mel = compute_mel_spectrogram_tinygrad(audio[:int(DURATION * SAMPLE_RATE)])
        model = build_audio_classifier()
        cls_name, conf = model.predict(mel)
        print(f"Classification: {cls_name} (confidence: {conf:.3f})")
        print("(Note: Using random weights — train the model for real predictions)")
    else:
        print("Usage:")
        print("  python3 audio_classify.py --bench          # Benchmark")
        print("  python3 audio_classify.py --file audio.wav  # Classify")

if __name__ == "__main__":
    main()
