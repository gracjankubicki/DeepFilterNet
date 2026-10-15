//! A personalized encoder (extra inputs `spk` and `sim`, research/voice002 export) running
//! in tract must sound like the torch model it was exported from.
//!
//! Fixtures come from `python -m voice002.export_onnx ... --reference DIR`. Set
//! `DF_PERSONAL_MODEL` to the archive and `DF_PERSONAL_REFERENCE` to DIR; without them the
//! test is skipped (the archive is not part of this repository).
//!
//! Plain DFN3 is compared the same way (bundled model vs torch DFN3): torch and libDF differ
//! a little in features, so the personal model must stay as close as DFN3 does where the
//! output is audible. In strongly suppressed parts (the other talker) the same tiny feature
//! differences give a lower ratio against the quiet output, so there the difference is
//! judged against the input instead.
//!
//! The personal model is more sensitive than DFN3 by itself: in torch, -75 dBFS of added
//! noise changes its audible output by 46.2 dB SDR against 55.1 dB for DFN3 (2026-10-15).
//! Its tract port measured 41.2 dB vs 45.7 dB for DFN3 on audible frames, so the margin
//! below is set under that intrinsic 8.9 dB gap.

use std::path::{Path, PathBuf};

use df::tract::{DfParams, DfTract, RuntimeParams};
use ndarray::prelude::*;

/// One cue per 200 ms = 20 frames of 10 ms (research/voice002 dfnet.SIM_HOP_FRAMES).
const SIM_HOP_FRAMES: usize = 20;
/// Allowed loss against the DFN3 baseline where the output is audible (dB).
const MARGIN_DB: f64 = 6.0;
/// 10 ms frames of the torch output at least this loud count as audible (dBFS).
const AUDIBLE_DBFS: f64 = -40.0;
/// The whole difference must stay this far below the input (dB).
const MIN_INPUT_TO_DIFF_DB: f64 = 35.0;

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn params() -> RuntimeParams {
    // Always run both stages, like the torch model.
    RuntimeParams::default_with_ch(1).with_thresholds(-1000.0, 1000.0, 1000.0)
}

/// Enhance frame by frame; `cue(frame)` sets the conditioning before each frame.
fn enhance(model: &mut DfTract, x: &[f32], mut cue: impl FnMut(&mut DfTract, usize)) -> Vec<f32> {
    let hop = model.hop_size;
    let mut out = vec![0f32; x.len()];
    for (i, (inp, o)) in x.chunks_exact(hop).zip(out.chunks_exact_mut(hop)).enumerate() {
        cue(model, i);
        let noisy = ArrayView2::from_shape((1, hop), inp).unwrap();
        let enh = ArrayViewMut2::from_shape((1, hop), o).unwrap();
        model.process(noisy, enh).unwrap();
    }
    out
}

fn energy(x: impl Iterator<Item = f64>) -> f64 {
    x.map(|v| v * v).sum::<f64>().max(f64::MIN_POSITIVE)
}

struct Parity {
    /// Signal-to-difference ratio over audible frames of the torch output (dB).
    audible_db: f64,
    /// Input energy over difference energy, whole signal (dB).
    input_db: f64,
}

/// `rust` is delayed by the model latency relative to `torch` and `input`.
fn parity(model: &DfTract, rust: &[f32], torch: &[f32], input: &[f32]) -> Parity {
    let lat = model.fft_size - model.hop_size + model.lookahead * model.hop_size;
    let n = rust.len() - lat;
    let (r, t) = (&rust[lat..], &torch[..n]);
    let hop = model.hop_size;
    let (mut sig, mut diff) = (0f64, 0f64);
    for (rf, tf) in r.chunks_exact(hop).zip(t.chunks_exact(hop)) {
        let level = 10.0 * (energy(tf.iter().map(|&v| v as f64)) / hop as f64).log10();
        if level >= AUDIBLE_DBFS {
            sig += energy(tf.iter().map(|&v| v as f64));
            diff += energy(rf.iter().zip(tf).map(|(&a, &b)| a as f64 - b as f64));
        }
    }
    let all_diff = energy(r.iter().zip(t).map(|(&a, &b)| a as f64 - b as f64));
    Parity {
        audible_db: 10.0 * (sig / diff.max(f64::MIN_POSITIVE)).log10(),
        input_db: 10.0 * (energy(input[..n].iter().map(|&v| v as f64)) / all_diff).log10(),
    }
}

#[test]
fn personal_model_matches_torch() {
    let (Some(model), Some(dir)) = (
        std::env::var_os("DF_PERSONAL_MODEL"),
        std::env::var_os("DF_PERSONAL_REFERENCE"),
    ) else {
        eprintln!("skipped: set DF_PERSONAL_MODEL and DF_PERSONAL_REFERENCE");
        return;
    };
    let dir = PathBuf::from(dir);
    let x = read_f32(&dir.join("input.f32"));
    let spk = read_f32(&dir.join("spk.f32"));
    let sim = read_f32(&dir.join("sim.f32"));

    let mut dfn3 = DfTract::new(DfParams::default(), &params()).unwrap();
    assert!(!dfn3.is_conditioned());
    let y0 = enhance(&mut dfn3, &x, |_, _| {});
    let base = parity(&dfn3, &y0, &read_f32(&dir.join("dfn3.f32")), &x);

    let mut personal = DfTract::new(DfParams::new(PathBuf::from(model)).unwrap(), &params()).unwrap();
    assert_eq!(personal.speaker_dim(), Some(spk.len()));
    personal.set_speaker(&spk).unwrap();
    let hops = sim.len() / 2;
    // torch shifts the features by the encoder's conv lookahead (DfNet.pad_feat): the
    // encoder step that sees frame n uses the cue of frame n - conv_lookahead.
    let shift = personal.conv_lookahead;
    let y = enhance(&mut personal, &x, |m, frame| {
        let h = (frame.saturating_sub(shift) / SIM_HOP_FRAMES).min(hops - 1);
        m.set_sim(&sim[2 * h..2 * h + 2]).unwrap();
    });
    let got = parity(&personal, &y, &read_f32(&dir.join("personal.f32")), &x);
    eprintln!(
        "vs torch, audible frames: DFN3 {:.1} dB, personal {:.1} dB; input/difference: DFN3 {:.1} dB, personal {:.1} dB",
        base.audible_db, got.audible_db, base.input_db, got.input_db
    );
    assert!(
        got.audible_db >= base.audible_db - MARGIN_DB,
        "audible: personal {:.1} dB < DFN3 {:.1} dB - {MARGIN_DB}",
        got.audible_db,
        base.audible_db
    );
    assert!(got.input_db >= MIN_INPUT_TO_DIFF_DB, "difference {:.1} dB below the input", got.input_db);
}

#[test]
fn plain_model_rejects_conditioning() {
    let mut dfn3 = DfTract::new(DfParams::default(), &params()).unwrap();
    assert_eq!(dfn3.speaker_dim(), None);
    assert!(dfn3.set_speaker(&[0.0; 192]).is_err());
    assert!(dfn3.set_sim(&[0.0; 2]).is_err());
}
