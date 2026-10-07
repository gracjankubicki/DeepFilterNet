//! `transforms::resample` was ported to rubato 5 (`Fft` + `process_all`). Checks length,
//! delay compensation, channel order and amplitude for short and multi-channel input.

use df::transforms::resample;
use ndarray::prelude::*;

fn sine(sr: usize, len: usize, freq: f32) -> Array1<f32> {
    Array1::from_shape_fn(len, |i| {
        (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin()
    })
}

fn rms(x: ArrayView1<f32>) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

/// Dominant frequency estimated from rising zero crossings.
fn freq(x: ArrayView1<f32>, sr: usize) -> f32 {
    let v: Vec<f32> = x.to_vec();
    let crossings = v.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
    crossings as f32 * sr as f32 / v.len() as f32
}

#[test]
fn output_length_matches_ratio() {
    for (sr, new_sr, len) in [
        (48_000, 16_000, 48_000),
        (16_000, 48_000, 16_000),
        (44_100, 48_000, 44_100),
        (48_000, 44_100, 1_000),
        (24_000, 48_000, 17),
    ] {
        let x = Array2::<f32>::zeros((1, len));
        let y = resample(x.view(), sr, new_sr, None).unwrap();
        let expected = (len as f64 * new_sr as f64 / sr as f64).ceil() as usize;
        assert_eq!(
            y.len_of(Axis(1)),
            expected,
            "{sr} → {new_sr}, {len} samples"
        );
    }
}

#[test]
fn keeps_channel_order_frequency_and_level() {
    let (sr, new_sr, len) = (44_100, 48_000, 44_100);
    let mut x = Array2::<f32>::zeros((2, len));
    x.row_mut(0).assign(&sine(sr, len, 440.0));
    x.row_mut(1).assign(&(sine(sr, len, 1_000.0) * 0.25));
    let y = resample(x.view(), sr, new_sr, None).unwrap();
    assert_eq!(y.shape(), &[2, 48_000]);
    let (c0, c1) = (y.slice(s![0, 4_000..44_000]), y.slice(s![1, 4_000..44_000]));
    assert!(
        (freq(c0, new_sr) - 440.0).abs() < 3.0,
        "ch0 {}",
        freq(c0, new_sr)
    );
    assert!(
        (freq(c1, new_sr) - 1_000.0).abs() < 3.0,
        "ch1 {}",
        freq(c1, new_sr)
    );
    assert!((rms(c0) - 0.7071).abs() < 0.01, "ch0 rms {}", rms(c0));
    assert!(
        (rms(c1) - 0.25 * 0.7071).abs() < 0.01,
        "ch1 rms {}",
        rms(c1)
    );
}

#[test]
fn delay_is_compensated() {
    let (sr, new_sr, len) = (16_000, 48_000, 16_000);
    let mut x = Array2::<f32>::zeros((1, len));
    x[[0, 5_000]] = 1.0;
    let y = resample(x.view(), sr, new_sr, None).unwrap();
    let peak = y
        .row(0)
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert!(
        (peak as i64 - 15_000).abs() <= 2,
        "impulse at {peak}, expected 15000"
    );
}
