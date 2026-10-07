//! Guards against audible regressions when upgrading the inference backend (tract) or
//! ndarray. The reference output was produced with the original backend (tract 0.21.4) by
//! running this test with `DF_WRITE_REFERENCE=1`.
//!
//! History: an earlier tract upgrade had to be reverted because of audio artifacts
//! (upstream PR #405), so every backend bump must keep this test green.

use std::path::PathBuf;

use df::tract::{DfParams, DfTract, RuntimeParams};
use df::wav_utils::ReadWav;
use ndarray::prelude::*;

const REFERENCE: &str = "tests/data/noisy_snr0_dfn3_reference.f32";
/// Minimum similarity between new and reference output (signal-to-difference ratio).
/// Bit-exactness is not expected across backend versions, but anything audible is
/// far below this threshold.
const MIN_SDR_DB: f64 = 40.0;

fn manifest_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn new_model() -> DfTract {
    DfTract::new(DfParams::default(), &RuntimeParams::default_with_ch(1)).expect("model init")
}

fn enhance(noisy: ArrayView2<f32>) -> Array2<f32> {
    enhance_with(&mut new_model(), noisy)
}

fn enhance_with(model: &mut DfTract, noisy: ArrayView2<f32>) -> Array2<f32> {
    let mut enh = Array2::<f32>::zeros(noisy.raw_dim());
    for (ns_f, enh_f) in noisy
        .axis_chunks_iter(Axis(1), model.hop_size)
        .zip(enh.axis_chunks_iter_mut(Axis(1), model.hop_size))
    {
        if ns_f.len_of(Axis(1)) < model.hop_size {
            break;
        }
        model.process(ns_f, enh_f).expect("process frame");
    }
    enh
}

fn noisy_sample() -> Array2<f32> {
    ReadWav::new(manifest_path("../assets/noisy_snr0.wav").to_str().unwrap())
        .expect("read noisy sample")
        .samples_arr2()
        .expect("samples")
}

fn read_f32(path: &PathBuf) -> Vec<f32> {
    std::fs::read(path)
        .expect("reference file (generate with DF_WRITE_REFERENCE=1)")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn enhanced_output_matches_reference_backend() {
    let reader = ReadWav::new(manifest_path("../assets/noisy_snr0.wav").to_str().unwrap())
        .expect("read noisy sample");
    assert_eq!(reader.sr, 48000);
    let noisy = reader.samples_arr2().expect("samples");
    let enh = enhance(noisy.view());
    let out: Vec<f32> = enh.row(0).to_vec();

    let reference_path = manifest_path(REFERENCE);
    if std::env::var_os("DF_WRITE_REFERENCE").is_some() {
        std::fs::create_dir_all(reference_path.parent().unwrap()).unwrap();
        let bytes: Vec<u8> = out.iter().flat_map(|s| s.to_le_bytes()).collect();
        std::fs::write(&reference_path, bytes).unwrap();
        return;
    }

    let reference = read_f32(&reference_path);
    assert_eq!(reference.len(), out.len(), "output length changed");
    let signal: f64 = reference.iter().map(|&r| (r as f64).powi(2)).sum();
    let diff: f64 = reference.iter().zip(&out).map(|(&r, &o)| (r as f64 - o as f64).powi(2)).sum();
    let sdr_db = 10.0 * (signal / diff.max(f64::MIN_POSITIVE)).log10();
    let max_abs = reference.iter().zip(&out).map(|(&r, &o)| (r - o).abs()).fold(0f32, f32::max);
    eprintln!("signal-to-difference vs reference: {sdr_db:.1} dB, max |diff| = {max_abs:e}");
    assert!(
        sdr_db >= MIN_SDR_DB,
        "output deviates from reference: {sdr_db:.1} dB < {MIN_SDR_DB} dB"
    );
}

/// Clones must not share feature buffers. Two clones of a warmed-up model continue on
/// different audio, interleaved frame by frame and also on two threads at once; each result
/// must equal that of an untouched clone running alone.
#[test]
fn cloned_models_are_independent() {
    let noisy = noisy_sample();
    let other: Array2<f32> =
        noisy.mapv(|v| -0.5 * v).slice(s![.., ..;-1]).as_standard_layout().into_owned();
    let warm = {
        let mut m = new_model();
        enhance_with(&mut m, noisy.slice(s![.., ..48_000]));
        m
    };
    // References: each continuation computed by its own untouched clone.
    let expected_a = enhance_with(&mut warm.clone(), noisy.view());
    let expected_b = enhance_with(&mut warm.clone(), other.view());

    // Interleaved on one thread.
    let (mut a, mut b) = (warm.clone(), warm.clone());
    let hop = a.hop_size;
    let mut out_a = Array2::<f32>::zeros(noisy.raw_dim());
    let mut out_b = Array2::<f32>::zeros(noisy.raw_dim());
    for i in 0..noisy.len_of(Axis(1)) / hop {
        let r = i * hop..(i + 1) * hop;
        a.process(
            noisy.slice(s![.., r.clone()]),
            out_a.slice_mut(s![.., r.clone()]),
        )
        .unwrap();
        b.process(other.slice(s![.., r.clone()]), out_b.slice_mut(s![.., r])).unwrap();
    }
    assert_eq!(out_a, expected_a, "clone a changed by clone b");
    assert_eq!(out_b, expected_b, "clone b changed by clone a");

    // Concurrently on two threads.
    let (mut a, mut b) = (warm.clone(), warm);
    let (na, nb) = (noisy.clone(), other.clone());
    let ta = std::thread::spawn(move || enhance_with(&mut a, na.view()));
    let tb = std::thread::spawn(move || enhance_with(&mut b, nb.view()));
    assert_eq!(ta.join().unwrap(), expected_a);
    assert_eq!(tb.join().unwrap(), expected_b);
}

/// `reset` forgets previous audio: the output equals that of a fresh model.
#[test]
fn reset_matches_a_fresh_model() {
    let noisy = noisy_sample();
    let expected = enhance(noisy.view());
    let mut m = new_model();
    enhance_with(&mut m, noisy.mapv(|v| v * 0.7).view());
    m.reset().unwrap();
    assert_eq!(enhance_with(&mut m, noisy.view()), expected);
}
