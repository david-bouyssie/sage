//! Native mzDB reading, via [`mzdb`] (the `mzdb-rs` crate).
//!
//! Reads mzDB files directly into [`RawSpectrum`], the same shape as the Thermo reader in
//! `crate::thermo`. mzDB is SQLite-backed and already fully decoded by the `mzdb` crate -- peak
//! arrays, precursor fields and isolation windows all come back as typed Rust values rather than
//! packed blobs or hand-parsed XML, so this reader is thinner than the Thermo one.
//!
//! # MS2-only by default
//!
//! Sage only needs MS1 spectra for optional stages -- `predict_rt`'s alignment step and LFQ's MS1
//! trace integration -- so this reader takes an `ms1` flag and skips the MS1 table entirely when it
//! is `false`, mirroring `requires_ms1` in the runner. Skipping is a real saving here, not just a
//! filter: mzDB's `for_each_spectrum`/`iter_spectra` accept an MS-level filter that becomes a SQL
//! `WHERE ms_level = ?`, so MS1 rows are never read off disk rather than read and discarded.
//!
//! # Local files only
//!
//! Same restriction as Thermo and Bruker: `rusqlite` opens a local path, not a stream, so there is
//! no `object_store` route for `s3://`/`gs://`/`az://` mzDB files.
//!
//! # Ion injection time comes from `scan_list`, not the header
//!
//! mzDB's `SpectrumHeader` has no dedicated field for it, but the value is not actually missing --
//! it is written as `MS:1000927` on the `<scan>` element inside the `scan_list` XML column (the same
//! `[Thermo Trailer Extra]` value the Thermo reader in `crate::thermo` reads directly from the
//! instrument), and `mzdb::xml::parse_scan_list` already extracts it as `Scan::ion_injection_time`
//! in milliseconds. An earlier version of this reader looked only at `param_tree`, where the value
//! never appears, and defaulted to zero as a result; that was a real gap, not an mzDB limitation.
//!
//! # Isolation windows and precursor m/z come from typed XML parsing, not regex
//!
//! `mzdb::xml::extract_isolation_window` and `extract_selected_ion_mz` parse the `precursor_list`
//! XML column with `roxmltree`, already resolving `isolation window target m/z` against its lower
//! and upper offsets. This reader calls those directly rather than re-deriving bounds from raw
//! fields, so it can't repeat the centering bug the Thermo reader had before its own fix.

use sage_core::spectrum::{Precursor, RawSpectrum, Representation};
use sage_core::mass::Tolerance;

use mzdb::model::Spectrum as MzDbSpectrum;
use mzdb::queries::SpectrumHeaderLoadOptions;
use mzdb::MzDbReader;

#[derive(thiserror::Error, Debug)]
pub enum MzDbError {
    // `mzdb-rs`'s own API returns `anyhow_ext::Result`, which carries a `Display` impl but not a
    // `std::error::Error` we can `#[from]` cleanly without pulling `anyhow_ext` itself into this
    // crate's dependency graph -- `sage-cloudpath` otherwise has no `anyhow` dependency anywhere,
    // so the error is captured by its rendered message instead of by source-chaining.
    #[error("mzDB error: {0}")]
    MzDb(String),
    #[error("mzDB file has a non-UTF8 path: {0}")]
    InvalidPath(String),
}

/// Read spectra from an mzDB file.
///
/// `file_id` is Sage's index of this file within the run, carried onto each spectrum. `ms1`
/// controls whether MS1 rows are read at all -- pass `false` when nothing downstream needs them
/// (no `predict_rt`, no LFQ), which skips that table rather than reading and discarding it.
pub fn read_mzdb<P: AsRef<std::path::Path>>(
    path: P,
    file_id: usize,
    ms1: bool,
) -> Result<Vec<RawSpectrum>, MzDbError> {
    let path_str = path
        .as_ref()
        .to_str()
        .ok_or_else(|| MzDbError::InvalidPath(path.as_ref().display().to_string()))?;
    // `precursor_list` carries mz/charge/isolation-window and must stay on; `scan_list` carries ion
    // injection time and is off by default, so both are requested explicitly here rather than
    // spreading from `SpectrumHeaderLoadOptions::default()` or `::none()` -- neither on its own
    // gives the combination this reader actually needs, and getting that spread wrong silently
    // drops one of the two columns rather than failing to compile.
    let load_options = SpectrumHeaderLoadOptions {
        load_param_tree: false,
        load_scan_list: true,
        load_precursor_list: true,
    };
    let reader = MzDbReader::open_with_options(path_str, load_options)
        .map_err(|e| MzDbError::MzDb(e.to_string()))?;

    let mut spectra = Vec::with_capacity(reader.get_spectrum_count());

    // `ms1 = false` requests MS2-only directly from `for_each_spectrum`'s own filter, rather than
    // reading everything and discarding MS1 rows here -- the filter becomes a SQL predicate inside
    // `mzdb`, so this is a real read-avoidance, not just a downstream `if`.
    let ms_level_filter: Option<u8> = if ms1 { None } else { Some(2) };

    reader
        .for_each_spectrum(ms_level_filter, |spectrum| {
            spectra.push(to_raw_spectrum(spectrum, file_id));
            Ok(())
        })
        .map_err(|e| MzDbError::MzDb(e.to_string()))?;

    Ok(spectra)
}

/// Convert one `mzdb` spectrum into Sage's `RawSpectrum`.
fn to_raw_spectrum(spectrum: &MzDbSpectrum, file_id: usize) -> RawSpectrum {
    let header = &spectrum.header;
    let data = &spectrum.data;

    let representation = match data.data_encoding.mode {
        mzdb::model::DataMode::Centroid => Representation::Centroid,
        mzdb::model::DataMode::Profile => Representation::Profile,
        // `Fitted` peaks (mzDB's third mode, carrying lwhm/rwhm alongside mz/intensity) are already
        // centroid-shaped -- one m/z, one intensity per peak -- so they are handled the same way as
        // Sage's own scorer, which only distinguishes centroid from profile.
        mzdb::model::DataMode::Fitted => Representation::Centroid,
    };

    let ms_level = header.ms_level as u8;
    let precursors = if ms_level > 1 {
        build_precursors(header)
    } else {
        Vec::new()
    };

    let ion_injection_time = header
        .scan_list_str
        .as_deref()
        .and_then(extract_ion_injection_time)
        .unwrap_or(0.0);

    RawSpectrum {
        file_id,
        ms_level,
        id: header.id.to_string(),
        precursors,
        representation,
        // mzDB's `time` column is *seconds* (confirmed directly against this crate's own writer
        // convention, and against the raw SQL contents of a real fixture file), while
        // `RawSpectrum::scan_start_time` is documented in minutes -- the same convention Sage's
        // mzML and Thermo readers already follow. An earlier version of this reader passed
        // `header.time` straight through with a comment claiming the units already matched; that
        // claim was never actually verified and was wrong by exactly a factor of 60, caught only
        // by an exhaustive scan-by-scan comparison against the reference mzDB's own SQL contents,
        // not by the single-scan probe run first.
        scan_start_time: header.time / 60.0,
        ion_injection_time,
        total_ion_current: header.tic,
        mz: data.mz_array.clone(),
        intensity: data.intensity_array.clone(),
        // mzDB does not currently carry an ion mobility dimension in this crate's model.
        mobility: None,
    }
}

/// Pull `MS:1000927` (ion injection time, milliseconds) from a spectrum's `scan_list` XML.
///
/// Falls back to `None` -- not `0.0` -- when the scan has no injection-time CV param at all, so the
/// caller's `unwrap_or(0.0)` is a deliberate default rather than this function silently manufacturing
/// a value. `parse_scan_list` returns one `Scan` per combination entry; only the first is used, same
/// as `extract_scan_time` does for the RT case.
fn extract_ion_injection_time(scan_list_xml: &str) -> Option<f32> {
    let scan_list = mzdb::xml::parse_scan_list(scan_list_xml).ok()?;
    scan_list
        .scans
        .first()?
        .ion_injection_time
        .map(|v| v as f32)
}

/// Build the precursor list for an MSn spectrum from its header's stored XML fields.
fn build_precursors(header: &mzdb::model::SpectrumHeader) -> Vec<Precursor> {
    let Some(precursor_list_xml) = header.precursor_list_str.as_deref() else {
        return Vec::new();
    };

    // Prefer the header's own `main_precursor_mz`/`main_precursor_charge` columns when present --
    // these are mzDB's already-resolved values, equivalent to Thermo's monoisotopic trailer field --
    // falling back to the XML's selected-ion m/z only if the header column is absent.
    let mz = header
        .precursor_mz
        .or_else(|| mzdb::xml::extract_selected_ion_mz(precursor_list_xml));

    let Some(mz) = mz else {
        return Vec::new();
    };

    let charge = header
        .precursor_charge
        .and_then(|c| u8::try_from(c).ok());

    // Already resolved against target +/- offsets by `mzdb::xml`, so no re-derivation of window
    // bounds from a target/offset pair happens here -- unlike the Thermo reader, which has to
    // combine `isolation_width`/`isolation_offset` itself because `thernio` exposes the raw reaction
    // fields rather than a finished window.
    let isolation_window = mzdb::xml::extract_isolation_window(precursor_list_xml)
        .map(|(lo, hi)| Tolerance::Da(lo as f32, hi as f32));

    vec![Precursor {
        mz: mz as f32,
        intensity: None,
        charge,
        spectrum_ref: None,
        isolation_window,
        inverse_ion_mobility: None,
    }]
}
