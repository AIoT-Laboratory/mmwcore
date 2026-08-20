use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::{AdcArchiveFileError, MAX_FRAME_BYTES, MAX_METADATA_BYTES, error};

const RADAR_CAPTURE_SCHEMA: &str = "mmwcore.radar_capture_spec.v1";

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CaptureRecord {
    pub(super) schema: String,
    pub(super) profile: ProfileRecord,
    pub(super) adc: AdcRecord,
    pub(super) tx_order: Vec<u64>,
    pub(super) frame_periodicity_s: Option<f64>,
    pub(super) num_frames: Option<u64>,
    pub(super) expected_size_bytes: Option<u64>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ProfileRecord {
    pub(super) start_frequency_hz: f64,
    pub(super) frequency_slope_hz_per_s: f64,
    pub(super) adc_sample_rate_hz: f64,
    pub(super) adc_start_time_s: f64,
    pub(super) ramp_end_time_s: f64,
    pub(super) idle_time_s: f64,
    pub(super) num_adc_samples: u64,
    pub(super) num_chirps_per_tx: u64,
    pub(super) num_tx: u64,
    pub(super) num_rx: u64,
    pub(super) speed_of_light_mps: f64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AdcRecord {
    pub(super) num_chirps: u64,
    pub(super) num_rx: u64,
    pub(super) num_samples: u64,
    pub(super) layout: String,
}

pub(super) fn validate_capture_json(bytes: &[u8]) -> Result<CaptureRecord, AdcArchiveFileError> {
    if bytes.is_empty() || bytes.len() as u64 > MAX_METADATA_BYTES {
        return Err(error(
            "ADC archive capture metadata size is outside v2 bounds.",
        ));
    }
    let capture: CaptureRecord = serde_json::from_slice(bytes)
        .map_err(|value| error(format!("ADC archive capture metadata is invalid: {value}")))?;
    validate_capture(&capture)?;
    Ok(capture)
}

pub(super) fn canonical_capture_json(
    capture: &CaptureRecord,
) -> Result<Vec<u8>, AdcArchiveFileError> {
    serde_json::to_vec(capture).map_err(|value| {
        error(format!(
            "Cannot serialize ADC archive capture metadata: {value}"
        ))
    })
}

fn validate_capture(capture: &CaptureRecord) -> Result<(), AdcArchiveFileError> {
    if capture.schema != RADAR_CAPTURE_SCHEMA {
        return Err(error("ADC archive capture metadata schema is unsupported."));
    }
    let profile = &capture.profile;
    for (name, value) in [
        ("start_frequency_hz", profile.start_frequency_hz),
        ("frequency_slope_hz_per_s", profile.frequency_slope_hz_per_s),
        ("adc_sample_rate_hz", profile.adc_sample_rate_hz),
        ("adc_start_time_s", profile.adc_start_time_s),
        ("ramp_end_time_s", profile.ramp_end_time_s),
        ("idle_time_s", profile.idle_time_s),
        ("speed_of_light_mps", profile.speed_of_light_mps),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(error(format!("ADC archive capture {name} is invalid.")));
        }
    }
    if profile.adc_start_time_s >= profile.ramp_end_time_s {
        return Err(error(
            "ADC archive capture adc_start_time_s must precede ramp_end_time_s.",
        ));
    }
    for (name, value) in [
        ("num_adc_samples", profile.num_adc_samples),
        ("num_chirps_per_tx", profile.num_chirps_per_tx),
        ("num_tx", profile.num_tx),
        ("num_rx", profile.num_rx),
        ("adc.num_chirps", capture.adc.num_chirps),
        ("adc.num_rx", capture.adc.num_rx),
        ("adc.num_samples", capture.adc.num_samples),
    ] {
        if value == 0 {
            return Err(error(format!(
                "ADC archive capture {name} must be positive."
            )));
        }
    }
    if !matches!(
        capture.adc.layout.as_str(),
        "iq_interleaved" | "sample_i_then_q" | "group2_i_then_q" | "group4_i_then_q"
    ) {
        return Err(error("ADC archive capture ADC layout is unsupported."));
    }
    if capture.adc.num_rx != profile.num_rx
        || capture.adc.num_samples != profile.num_adc_samples
        || capture.tx_order.len() as u64 != profile.num_tx
        || capture.adc.num_chirps
            != profile
                .num_chirps_per_tx
                .checked_mul(profile.num_tx)
                .ok_or_else(|| error("ADC archive chirp count overflows u64."))?
    {
        return Err(error("ADC archive capture dimensions are inconsistent."));
    }
    let tx_order: HashSet<_> = capture.tx_order.iter().copied().collect();
    if tx_order.len() != capture.tx_order.len() {
        return Err(error("ADC archive capture tx_order contains duplicates."));
    }
    if let Some(periodicity) = capture.frame_periodicity_s {
        let active =
            (profile.idle_time_s + profile.ramp_end_time_s) * capture.adc.num_chirps as f64;
        if !periodicity.is_finite() || periodicity <= 0.0 || periodicity < active {
            return Err(error("ADC archive capture frame periodicity is invalid."));
        }
    }
    if capture.num_frames.is_none_or(|value| value == 0)
        || capture.expected_size_bytes.is_none_or(|value| value == 0)
    {
        return Err(error(
            "ADC archive capture must declare positive num_frames and expected_size_bytes.",
        ));
    }
    Ok(())
}

pub(super) fn capture_frame_bytes(capture: &CaptureRecord) -> Result<u64, AdcArchiveFileError> {
    capture
        .adc
        .num_chirps
        .checked_mul(capture.adc.num_rx)
        .and_then(|value| value.checked_mul(capture.adc.num_samples))
        .and_then(|value| value.checked_mul(4))
        .filter(|value| *value <= MAX_FRAME_BYTES)
        .ok_or_else(|| error("ADC archive capture frame size is invalid."))
}
