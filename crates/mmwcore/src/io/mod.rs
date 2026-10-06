//! Standardized lossless ADC compression and file I/O.
pub mod adc_compression;

// Compatibility module paths for earlier releases.
pub use adc_compression::codec as adc_archive;
pub use adc_compression::container as adc_archive_file;
