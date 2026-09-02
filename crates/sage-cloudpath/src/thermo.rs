//! Native Thermo `.raw` reading, via [`thernio`].
//!
//! Reads Thermo RAW files directly into [`RawSpectrum`], removing the mzML conversion step for
//! Thermo instruments. The field mapping follows `mzdb-rs`'s `thermo2mzdb` converter, which drives
//! the same `thernio` API for a different target.
//!
//! # Local files only
//!
//! Unlike mzML and MGF, this does not go through `read_and_execute`, so `s3://`, `gs://` and `az://`
//! paths are not supported. The Bruker TDF reader has the same limitation for the same reason: both
//! underlying libraries seek within a file rather than consuming a byte stream.
//!
//! # Centroids are required
//!
//! A scan with no centroid stream is an error, not a silent empty spectrum and not a fall back to
//! profile data. Sage searches centroided peaks; handing it an empty peak list would drop the scan
//! from the search with no indication that anything was lost. On Orbitrap data the centroid stream
//! (the FT label data) is essentially always present, so this fires rarely -- and when it does, the
//! error names the scans and their MS levels, because the consequence differs: missing MS2 centroids
//! block identification, while missing MS1 centroids block only label-free quantification.
//!
//! # Lock mass peaks are retained
//!
//! Thermo's own library filters reference/lock-mass peaks by default; `thernio` exposes them and
//! this reader keeps them, matching `mzdb-rs`. They sit at known fixed m/z and do not match peptide
//! fragments in any meaningful way, so the cost is a handful of extra peaks per spectrum.

use sage_core::spectrum::{Precursor, RawSpectrum, Representation};
use sage_core::mass::Tolerance;

use thernio::raw::{RawFile, ScanMode};

/// How many offending scans to name before truncating the error message.
const MAX_REPORTED_SCANS: usize = 10;

#[derive(thiserror::Error, Debug)]
pub enum ThermoError {
    #[error("Thermo RAW error: {0}")]
    Raw(#[from] thernio::raw::RawError),

    /// Scans carrying no centroid stream. Reported together rather than one at a time: a file
    /// acquired without label data has the problem on every scan of that type, and failing on the
    /// first would say nothing about the scale of it.
    #[error(
        "{count} scan(s) have no centroid stream and cannot be searched \
(MS levels: {ms_levels}; first scans: {scans}). \
Sage searches centroided peaks, so these spectra would otherwise be silently dropped. \
Re-acquire with label data enabled, or convert this file to centroided mzML."
    )]
    MissingCentroids {
        count: usize,
        ms_levels: String,
        scans: String,
    },
}

/// Read every scan of a Thermo RAW file.
///
/// `file_id` is Sage's index of this file within the run, carried onto each spectrum.
pub fn read_thermo_raw<P: AsRef<std::path::Path>>(
    path: P,
    file_id: usize,
) -> Result<Vec<RawSpectrum>, ThermoError> {
    let mut raw = RawFile::open(path.as_ref())?;

    let first = raw.first_scan_number() as usize;
    let last = raw.last_scan_number() as usize;

    // Scan events are indexed from the first scan, and carry polarity and centroid/profile mode --
    // neither of which is on the scan itself.
    let scan_events: Vec<_> = raw.scan_events().to_vec();

    let mut spectra = Vec::with_capacity(last.saturating_sub(first) + 1);
    let mut missing: Vec<(usize, u8)> = Vec::new();

    for scan_number in first..=last {
        // The trailer record borrows `raw` immutably while `scan()` needs it mutably, so every
        // value needed from it is copied out here and the borrow released before the scan is read.
        let trailer = {
            let record = raw.trailer_extra_record(scan_number);
            TrailerValues {
                ion_injection_time: record.as_ref().and_then(|t| t.ion_injection_time()),
                charge_state: record.as_ref().and_then(|t| t.charge_state()),
                monoisotopic_mz: record.as_ref().and_then(|t| t.monoisotopic_mz()),
            }
        };
        let ion_injection_time = trailer.ion_injection_time.unwrap_or(0.0) as f32;

        let scan = raw.scan(scan_number)?;

        let centroids = match scan.centroids() {
            Some(peaks) if !peaks.is_empty() => peaks,
            _ => {
                missing.push((scan_number, scan.ms_level));
                continue;
            }
        };

        // `LabelPeak` already carries f32 m/z and intensity, which is exactly what `RawSpectrum`
        // holds -- no conversion, no precision loss.
        let mut mz = Vec::with_capacity(centroids.len());
        let mut intensity = Vec::with_capacity(centroids.len());
        for peak in centroids {
            mz.push(peak.mz);
            intensity.push(peak.intensity);
        }

        let event = scan_events.get(scan_number.saturating_sub(first));

        // `scan_mode` lives on the scan event, not the scan. Falling back to Centroid is safe here
        // because a scan without a centroid stream has already been rejected above.
        let representation = match event.map(|e| e.scan_mode) {
            Some(ScanMode::Profile) => Representation::Profile,
            _ => Representation::Centroid,
        };

        let precursors = if scan.ms_level > 1 {
            build_precursors(&scan, &trailer)
        } else {
            Vec::new()
        };

        spectra.push(RawSpectrum {
            file_id,
            ms_level: scan.ms_level,
            id: scan_number.to_string(),
            precursors,
            representation,
            // `thernio` reports retention time in minutes and `RawSpectrum::scan_start_time` is
            // documented in minutes, so this passes through unscaled. Note that `mzdb-rs` multiplies
            // by 60 here -- mzDB stores seconds -- and copying that would distort the time axis
            // silently, leaving RT prediction and LFQ to run against wrong values rather than fail.
            scan_start_time: scan.retention_time as f32,
            ion_injection_time,
            total_ion_current: scan.total_ion_current as f32,
            mz,
            intensity,
            // Thermo instruments in this path carry no ion mobility dimension.
            mobility: None,
        });
    }

    if !missing.is_empty() {
        return Err(missing_centroids_error(missing));
    }

    Ok(spectra)
}

/// The per-scan trailer values this reader needs, copied out of the borrowed record.
///
/// Exists purely to release the immutable borrow of `RawFile` before `scan()` takes a mutable one.
/// All three are range-validated inside `thernio`, so `None` means genuinely absent rather than
/// present-but-implausible.
#[derive(Debug, Default, Clone, Copy)]
struct TrailerValues {
    ion_injection_time: Option<f64>,
    charge_state: Option<i64>,
    monoisotopic_mz: Option<f64>,
}

/// Build the precursor list for an MSn scan.
fn build_precursors(scan: &thernio::raw::Scan, trailer: &TrailerValues) -> Vec<Precursor> {
    // The isolation target is the centre of the quadrupole selection window.
    let isolation_target = match scan.precursor_mz_values().first().copied() {
        Some(mz) => mz,
        None => return Vec::new(),
    };

    // Prefer the instrument's own monoisotopic assignment over the isolation target: the latter is
    // the centre of the selection window, which is not the precursor mass. `precursor_tol` is
    // applied to this value, so at tight ppm tolerances the difference costs identifications.
    // `thernio` range-validates the trailer value, so `None` here means genuinely absent.
    let mz = trailer.monoisotopic_mz.unwrap_or(isolation_target);

    let charge = trailer.charge_state.and_then(|c| u8::try_from(c).ok());

    // Isolation windows are not necessarily centred on the target: `isolation_offset` shifts the
    // window, which matters for the wide-window and DIA paths that use it to assign precursors.
    let isolation_window = scan.reactions.first().and_then(|reaction| {
        let width = reaction.isolation_width;
        if width <= 0.0 {
            return None;
        }
        // The window is centred on the *isolation target* -- the quadrupole centre the method
        // selected -- not on the precursor mass. Those differ: on this fixture the target is 476.2
        // while the monoisotopic precursor is 475.87238, a third of a dalton apart. Centring on the
        // precursor instead shifts the whole window by that gap, which the mzDB cross-check caught.
        let offset = reaction.isolation_offset.unwrap_or(0.0);
        let half = width / 2.0;
        Some(Tolerance::Da(
            (isolation_target + offset - half) as f32,
            (isolation_target + offset + half) as f32,
        ))
    });

    vec![Precursor {
        mz: mz as f32,
        intensity: None,
        charge,
        spectrum_ref: None,
        isolation_window,
        inverse_ion_mobility: None,
    }]
}

/// Summarise the scans that carried no centroid stream.
fn missing_centroids_error(missing: Vec<(usize, u8)>) -> ThermoError {
    let mut levels: Vec<u8> = missing.iter().map(|(_, level)| *level).collect();
    levels.sort_unstable();
    levels.dedup();

    let ms_levels = levels
        .iter()
        .map(|l| l.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let mut scans = missing
        .iter()
        .take(MAX_REPORTED_SCANS)
        .map(|(scan, _)| scan.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    if missing.len() > MAX_REPORTED_SCANS {
        scans.push_str(", ...");
    }

    ThermoError::MissingCentroids {
        count: missing.len(),
        ms_levels,
        scans,
    }
}
