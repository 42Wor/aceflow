//! Whisper-compatible WAV loading and log-Mel DSP.
//!
//! The default path matches OpenAI Whisper / `torch.stft` with a periodic
//! Hann window, center reflect padding, a dropped final frame, power
//! spectrum, Slaney-normalized Mel filters, and Whisper log scaling.

use hound::{SampleFormat, WavReader};
use rustfft::{num_complex::Complex, FftPlanner};
use std::path::Path;

use crate::device::Device;
use crate::tensor::Tensor;
use crate::{Error, Result};

pub const SAMPLE_RATE: u32 = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 80;
pub const N_SAMPLES: usize = 30 * SAMPLE_RATE as usize;
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH;
pub const N_FREQS: usize = N_FFT / 2 + 1;

/// Load a WAV file, mix down to mono, resample to 16 kHz, and return PCM.
pub fn load_wav(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let path = path.as_ref();
    let mut reader = WavReader::open(path)
        .map_err(|e| Error(format!("failed to open {}: {e}", path.display())))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;

    let mut mono: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, _) => {
            let raw: Vec<f32> = reader
                .samples::<f32>()
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| Error(format!("wav read: {e}")))?;
            mix_mono(&raw, channels)
        }
        (SampleFormat::Int, 8) => {
            let raw: Vec<i8> = reader
                .samples::<i8>()
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| Error(format!("wav read: {e}")))?;
            mix_mono(
                &raw.iter().map(|s| *s as f32 / 128.0).collect::<Vec<_>>(),
                channels,
            )
        }
        (SampleFormat::Int, 16) => {
            let raw: Vec<i16> = reader
                .samples::<i16>()
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| Error(format!("wav read: {e}")))?;
            mix_mono(
                &raw.iter().map(|s| *s as f32 / 32768.0).collect::<Vec<_>>(),
                channels,
            )
        }
        (SampleFormat::Int, 24 | 32) => {
            let raw: Vec<i32> = reader
                .samples::<i32>()
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| Error(format!("wav read: {e}")))?;
            let denom = if spec.bits_per_sample == 24 {
                8_388_608.0
            } else {
                2_147_483_648.0
            };
            mix_mono(
                &raw.iter().map(|s| *s as f32 / denom).collect::<Vec<_>>(),
                channels,
            )
        }
        (fmt, bits) => {
            return Err(Error(format!(
                "unsupported wav format {fmt:?} / {bits} bits"
            )))
        }
    };

    if spec.sample_rate != SAMPLE_RATE {
        mono = resample_linear(&mono, spec.sample_rate, SAMPLE_RATE);
    }
    Ok(mono)
}

fn mix_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    samples
        .chunks(channels)
        .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Linear resampling with rounded output length and endpoint clamping.
pub fn resample_linear(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || samples.is_empty() {
        return samples.to_vec();
    }
    assert!(from > 0 && to > 0, "sample rates must be non-zero");
    let ratio = from as f64 / to as f64;
    let new_len = ((samples.len() as f64) * to as f64 / from as f64).round() as usize;
    let last = samples.len() - 1;
    (0..new_len)
        .map(|i| {
            let src = i as f64 * ratio;
            let i0 = (src.floor() as usize).min(last);
            let i1 = (i0 + 1).min(last);
            let t = (src - i0 as f64) as f32;
            samples[i0] * (1.0 - t) + samples[i1] * t
        })
        .collect()
}

/// Pad or trim PCM to a 30-second Whisper chunk.
pub fn pad_or_trim(audio: &[f32]) -> Vec<f32> {
    pad_or_trim_to(audio, N_SAMPLES)
}

pub fn pad_or_trim_to(audio: &[f32], n_samples: usize) -> Vec<f32> {
    if audio.len() >= n_samples {
        audio[..n_samples].to_vec()
    } else {
        let mut out = vec![0.0f32; n_samples];
        out[..audio.len()].copy_from_slice(audio);
        out
    }
}

/// Periodic Hann window matching `torch.hann_window(n, periodic=True)`.
pub fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        (hz - f_min) / f_sp
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_min + f_sp * mel
    }
}

/// `(n_mels, n_fft/2+1)` Slaney-normalized triangular Mel filterbank.
pub fn mel_filters(n_mels: usize, n_fft: usize, sr: u32) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let fmin = 0.0f32;
    let fmax = sr as f32 / 2.0;
    let mut hz_points = vec![0.0f32; n_mels + 2];
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    for i in 0..n_mels + 2 {
        let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
        hz_points[i] = mel_to_hz(mel);
    }

    let mut fftfreqs = vec![0.0f32; n_freqs];
    for i in 0..n_freqs {
        fftfreqs[i] = fmax * i as f32 / (n_freqs - 1) as f32;
    }

    let mut weights = vec![0.0f32; n_mels * n_freqs];
    for i in 0..n_mels {
        let f_left = hz_points[i];
        let f_center = hz_points[i + 1];
        let f_right = hz_points[i + 2];
        let down = f_center - f_left;
        let up = f_right - f_center;
        for j in 0..n_freqs {
            let f = fftfreqs[j];
            let lower = (f - f_left) / down;
            let upper = (f_right - f) / up;
            weights[i * n_freqs + j] = lower.min(upper).max(0.0);
        }
        let enorm = 2.0 / (hz_points[i + 2] - hz_points[i]);
        for j in 0..n_freqs {
            weights[i * n_freqs + j] *= enorm;
        }
    }
    weights
}

/// Reflect-pad `n` samples on each side, excluding each edge sample.
pub fn reflect_pad(x: &[f32], n: usize) -> Vec<f32> {
    let len = x.len();
    if len == 0 {
        return vec![0.0; 2 * n];
    }
    let mut out = vec![0.0f32; len + 2 * n];
    out[n..n + len].copy_from_slice(x);
    for i in 0..n {
        let left = (i + 1).min(len - 1);
        out[n - 1 - i] = x[left];
        let right = len.saturating_sub(2 + i);
        out[n + len + i] = x[right];
    }
    out
}

/// Center-reflected STFT power in frequency-major row-major layout.
///
/// The final frame is deliberately dropped to match Whisper preprocessing.
pub fn stft_power(audio: &[f32], n_fft: usize, hop_length: usize) -> Vec<f32> {
    let (_, power) = stft_power_with_frames(audio, n_fft, hop_length);
    power
}

/// Return `(kept_frames, power)` where power has shape `(n_fft/2+1, frames)`.
pub fn stft_power_with_frames(audio: &[f32], n_fft: usize, hop_length: usize) -> (usize, Vec<f32>) {
    assert!(n_fft > 0, "n_fft must be positive");
    assert!(hop_length > 0, "hop_length must be positive");
    let window = hann_window(n_fft);
    let padded = reflect_pad(audio, n_fft / 2);
    if padded.len() < n_fft {
        return (0, Vec::new());
    }
    let n_stft_frames = (padded.len() - n_fft) / hop_length + 1;
    let n_keep = n_stft_frames.saturating_sub(1);
    let n_freqs = n_fft / 2 + 1;

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut power = vec![0.0f32; n_freqs * n_keep];
    let mut scratch = vec![Complex::new(0.0, 0.0); n_fft];
    for t in 0..n_keep {
        let start = t * hop_length;
        for i in 0..n_fft {
            scratch[i] = Complex::new(padded[start + i] * window[i], 0.0);
        }
        fft.process(&mut scratch);
        for f in 0..n_freqs {
            let value = scratch[f];
            power[f * n_keep + t] = value.re * value.re + value.im * value.im;
        }
    }
    (n_keep, power)
}

#[derive(Clone, Debug)]
pub struct MelSpectrogram {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub n_samples: usize,
    pub n_frames: usize,
    window: Vec<f32>,
    filters: Vec<f32>,
}

impl Default for MelSpectrogram {
    fn default() -> Self {
        Self::whisper()
    }
}

impl MelSpectrogram {
    pub fn whisper() -> Self {
        Self::new(SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, N_SAMPLES)
            .expect("Whisper Mel configuration is valid")
    }

    pub fn new(
        sample_rate: u32,
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
        n_samples: usize,
    ) -> Result<Self> {
        if sample_rate == 0 || n_fft == 0 || hop_length == 0 || n_mels == 0 || n_samples == 0 {
            return Err(Error(
                "sample_rate, n_fft, hop_length, n_mels, and n_samples must be positive".into(),
            ));
        }
        let n_frames = n_samples / hop_length;
        Ok(Self {
            sample_rate,
            n_fft,
            hop_length,
            n_mels,
            n_samples,
            n_frames,
            window: hann_window(n_fft),
            filters: mel_filters(n_mels, n_fft, sample_rate),
        })
    }

    pub fn window(&self) -> &[f32] {
        &self.window
    }

    pub fn filters(&self) -> &[f32] {
        &self.filters
    }

    /// Compute row-major `(n_mels, n_frames)` Whisper-scaled log-Mel values.
    pub fn transform(&self, audio: &[f32]) -> Vec<f32> {
        let audio = pad_or_trim_to(audio, self.n_samples);
        let padded = reflect_pad(&audio, self.n_fft / 2);
        let n_stft_frames = (padded.len() - self.n_fft) / self.hop_length + 1;
        let n_keep = n_stft_frames.saturating_sub(1).min(self.n_frames);
        let n_freqs = self.n_fft / 2 + 1;

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.n_fft);
        let mut magnitudes = vec![0.0f32; n_freqs * n_keep];
        let mut scratch = vec![Complex::new(0.0, 0.0); self.n_fft];
        for t in 0..n_keep {
            let start = t * self.hop_length;
            for i in 0..self.n_fft {
                scratch[i] = Complex::new(padded[start + i] * self.window[i], 0.0);
            }
            fft.process(&mut scratch);
            for f in 0..n_freqs {
                let value = scratch[f];
                magnitudes[f * n_keep + t] = value.re * value.re + value.im * value.im;
            }
        }

        let mut mel = vec![0.0f32; self.n_mels * n_keep];
        for m in 0..self.n_mels {
            for f in 0..n_freqs {
                let weight = self.filters[m * n_freqs + f];
                if weight == 0.0 {
                    continue;
                }
                let src = f * n_keep;
                let dst = m * n_keep;
                for t in 0..n_keep {
                    mel[dst + t] += weight * magnitudes[src + t];
                }
            }
        }
        apply_whisper_log_mel(&mut mel);

        if n_keep < self.n_frames {
            let mut full = vec![0.0f32; self.n_mels * self.n_frames];
            for m in 0..self.n_mels {
                full[m * self.n_frames..m * self.n_frames + n_keep]
                    .copy_from_slice(&mel[m * n_keep..(m + 1) * n_keep]);
            }
            return full;
        }
        mel
    }

    /// Whisper log-Mel as a tensor on `device`.
    ///
    /// The 400-point STFT stays on CPU (rustfft). The Mel filterbank is a
    /// GEMM, so when `device` is WGPU it runs through the pipeline cache.
    pub fn transform_to_tensor(&self, audio: &[f32], device: Device) -> Tensor {
        let audio = pad_or_trim_to(audio, self.n_samples);
        let (n_keep, power) = stft_power_with_frames(&audio, self.n_fft, self.hop_length);
        let n_freqs = self.n_fft / 2 + 1;
        let mut mel = if n_keep == 0 {
            vec![0.0f32; self.n_mels * self.n_frames]
        } else if device.is_wgpu() {
            let filters =
                Tensor::from_vec_on(self.filters.clone(), &[self.n_mels, n_freqs], device);
            let spectrum = Tensor::from_vec_on(power, &[n_freqs, n_keep], device);
            filters.matmul(&spectrum).data
        } else {
            let mut cpu = vec![0.0f32; self.n_mels * n_keep];
            for m in 0..self.n_mels {
                for f in 0..n_freqs {
                    let weight = self.filters[m * n_freqs + f];
                    if weight == 0.0 {
                        continue;
                    }
                    let src = f * n_keep;
                    let dst = m * n_keep;
                    for t in 0..n_keep {
                        cpu[dst + t] += weight * power[src + t];
                    }
                }
            }
            cpu
        };
        apply_whisper_log_mel(&mut mel);
        if n_keep < self.n_frames {
            let mut full = vec![0.0f32; self.n_mels * self.n_frames];
            if n_keep > 0 {
                for m in 0..self.n_mels {
                    full[m * self.n_frames..m * self.n_frames + n_keep]
                        .copy_from_slice(&mel[m * n_keep..(m + 1) * n_keep]);
                }
            }
            mel = full;
        }
        Tensor::from_vec_on(mel, &[self.n_mels, self.n_frames], device)
    }
}

fn apply_whisper_log_mel(mel: &mut [f32]) {
    let mut max_v = f32::NEG_INFINITY;
    for value in mel.iter_mut() {
        *value = (*value).max(1e-10).log10();
        if *value > max_v {
            max_v = *value;
        }
    }
    let floor = max_v - 8.0;
    for value in mel.iter_mut() {
        *value = (value.max(floor) + 4.0) / 4.0;
    }
}

/// Compute Whisper's row-major `(80, 3000)` log-Mel spectrogram.
pub fn log_mel_spectrogram(audio: &[f32]) -> Vec<f32> {
    MelSpectrogram::whisper().transform(audio)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_resampling_matches_reference_points() {
        let output = resample_linear(&[0.0, 2.0, 4.0], 3, 6);
        assert_eq!(output, vec![0.0, 1.0, 2.0, 3.0, 4.0, 4.0]);
    }

    #[test]
    fn periodic_hann_has_expected_samples() {
        let window = hann_window(4);
        assert!((window[0] - 0.0).abs() < 1e-6);
        assert!((window[1] - 0.5).abs() < 1e-6);
        assert!((window[2] - 1.0).abs() < 1e-6);
        assert!((window[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn stft_drops_the_final_frame() {
        let (frames, power) = stft_power_with_frames(&[1.0; 8], 4, 2);
        assert_eq!(frames, 4);
        assert_eq!(power.len(), 3 * frames);
    }

    #[test]
    fn whisper_log_mel_has_fixed_shape_and_silence_value() {
        let mel = log_mel_spectrogram(&[]);
        assert_eq!(mel.len(), N_MELS * N_FRAMES);
        assert!(mel.iter().all(|x| (*x + 1.5).abs() < 1e-6));
    }

    #[test]
    fn transform_to_tensor_cpu_matches_transform() {
        let spec = MelSpectrogram::whisper();
        let audio = vec![0.05; 8_000];
        let cpu = spec.transform(&audio);
        let tensor = spec.transform_to_tensor(&audio, Device::Cpu);
        assert_eq!(tensor.shape, vec![N_MELS, N_FRAMES]);
        assert_eq!(cpu, tensor.data);
    }

    #[test]
    fn mel_filters_are_slaney_normalized_and_nonempty() {
        let filters = mel_filters(80, 400, 16_000);
        assert_eq!(filters.len(), 80 * 201);
        assert!(filters.iter().sum::<f32>() > 0.0);
    }
}
