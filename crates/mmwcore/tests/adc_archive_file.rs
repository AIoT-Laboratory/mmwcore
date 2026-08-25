use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use mmwcore::{open_adc_archive_file, write_adc_archive_file};
use sha2::{Digest, Sha256};

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("current time")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "mmwcore-adc-archive-v3-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn capture_json(frame_count: usize) -> String {
    format!(
        concat!(
            "{{\"adc\":{{\"layout\":\"iq_interleaved\",\"num_chirps\":1,",
            "\"num_rx\":1,\"num_samples\":4}},\"expected_size_bytes\":{},",
            "\"frame_periodicity_s\":0.1,\"num_frames\":{},\"profile\":{{",
            "\"adc_sample_rate_hz\":4400000.0,\"adc_start_time_s\":0.000006,",
            "\"frequency_slope_hz_per_s\":60012000000000.0,\"idle_time_s\":0.00036,",
            "\"num_adc_samples\":4,\"num_chirps_per_tx\":1,\"num_rx\":1,",
            "\"num_tx\":1,\"ramp_end_time_s\":0.000065,",
            "\"speed_of_light_mps\":299792458.0,\"start_frequency_hz\":60000000000.0}},",
            "\"schema\":\"mmwcore.radar_capture_spec.v1\",\"tx_order\":[0]}}"
        ),
        frame_count * 16,
        frame_count,
    )
}

#[test]
fn v3_round_trip_is_self_describing_and_random_accessible() {
    let directory = TestDirectory::new();
    let source = directory.path().join("adc.bin");
    let destination = directory.path().join("adc.mmwa");
    let raw: Vec<u8> = (0..48).map(|value| (value * 17) as u8).collect();
    fs::write(&source, &raw).expect("write source");
    let digest: [u8; 32] = Sha256::digest(&raw).into();
    let metadata = capture_json(3);
    let supplied_metadata = format!("\n  {metadata}\n");

    let mut archive =
        write_adc_archive_file(&source, &destination, &supplied_metadata, Some(digest))
            .expect("write archive");

    assert_eq!(
        &fs::read(&destination).expect("read archive")[..8],
        b"MMWADCA3"
    );
    assert_ne!(archive.capture_json(), supplied_metadata);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(archive.capture_json()).unwrap(),
        serde_json::from_str::<serde_json::Value>(&metadata).unwrap()
    );
    let stored_metadata = archive.capture_json().to_owned();
    assert_eq!(archive.frame_bytes(), 16);
    assert_eq!(archive.frame_count(), 3);
    assert_eq!(archive.block_samples(), 512);
    assert_eq!(archive.restart_frames(), 4);
    assert_eq!(archive.read_frames(1, 3, true).unwrap(), raw[16..]);
    archive.verify_all().unwrap();
    assert_eq!(archive.read_frames(0, 1, false).unwrap(), raw[..16]);

    let reopened = open_adc_archive_file(&destination).expect("open archive");
    assert_eq!(reopened.capture_json(), stored_metadata);
    assert_eq!(reopened.adc_sha256(), digest);
}

#[test]
fn v3_batch_windows_preserve_order_across_shared_chunks() {
    let directory = TestDirectory::new();
    let source = directory.path().join("adc.bin");
    let destination = directory.path().join("adc.mmwa");
    let frame_bytes = 16;
    let frame_count = 12;
    let raw: Vec<u8> = (0..frame_bytes * frame_count)
        .map(|value| (value * 17) as u8)
        .collect();
    fs::write(&source, &raw).expect("write source");
    let mut archive =
        write_adc_archive_file(&source, &destination, &capture_json(frame_count), None)
            .expect("write archive");
    let starts = [5_u64, 0, 5, 3, 8];
    let mut expected = Vec::new();
    for start in starts {
        let start = usize::try_from(start).unwrap();
        expected.extend_from_slice(&raw[start * frame_bytes..(start + 4) * frame_bytes]);
    }

    assert_eq!(archive.read_windows(&starts, 4, true).unwrap(), expected);
    assert!(archive.read_windows(&[], 4, true).unwrap().is_empty());
    assert!(archive.read_windows(&[0], 0, true).is_err());
    assert!(archive.read_windows(&[9], 4, true).is_err());

    archive.verify_all().unwrap();
    assert_eq!(archive.read_windows(&starts, 4, false).unwrap(), expected);
}

#[test]
fn v3_writer_rejects_incomplete_or_inconsistent_capture_metadata() {
    let directory = TestDirectory::new();
    let source = directory.path().join("adc.bin");
    fs::write(&source, vec![0_u8; 32]).expect("write source");
    let valid: serde_json::Value = serde_json::from_str(&capture_json(2)).unwrap();

    for (index, mutation) in [
        ("unknown", serde_json::json!(true)),
        ("num_frames", serde_json::Value::Null),
        ("expected_size_bytes", serde_json::json!(16)),
    ]
    .into_iter()
    .enumerate()
    {
        let mut invalid = valid.clone();
        invalid
            .as_object_mut()
            .expect("capture object")
            .insert(mutation.0.to_owned(), mutation.1);
        let destination = directory.path().join(format!("invalid-{index}.mmwa"));
        assert!(write_adc_archive_file(&source, &destination, &invalid.to_string(), None).is_err());
        assert!(!destination.exists());
    }
}

#[test]
fn v3_rejects_metadata_tampering_truncation_and_overwrite() {
    let directory = TestDirectory::new();
    let source = directory.path().join("adc.bin");
    let destination = directory.path().join("adc.mmwa");
    fs::write(&source, vec![7_u8; 32]).expect("write source");
    write_adc_archive_file(&source, &destination, &capture_json(2), None).expect("write archive");

    assert!(write_adc_archive_file(&source, &destination, &capture_json(2), None).is_err());

    let original = fs::read(&destination).expect("read archive");
    let mut tampered = original.clone();
    tampered[112] ^= 1;
    let metadata_path = directory.path().join("metadata.mmwa");
    fs::write(&metadata_path, tampered).expect("write tampered archive");
    assert!(open_adc_archive_file(&metadata_path).is_err());

    let truncated_path = directory.path().join("truncated.mmwa");
    fs::write(&truncated_path, &original[..original.len() - 1]).expect("write truncated archive");
    assert!(open_adc_archive_file(&truncated_path).is_err());
}

#[test]
fn missing_archive_preserves_io_error_kind_and_source() {
    let directory = TestDirectory::new();
    let error = open_adc_archive_file(&directory.path().join("missing.mmwa")).unwrap_err();

    assert_eq!(error.io_kind(), Some(ErrorKind::NotFound));
    assert!(std::error::Error::source(&error).is_some());
}
