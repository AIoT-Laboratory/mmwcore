//! Byte-exact ADC compression: frame-group codec and self-describing .mmwa container.
pub mod codec;
pub mod container;
pub use codec::{
    AdcCompressionError, compress_adc_frames, decompress_adc_frames, maximum_compressed_adc_bytes,
};
pub use container::{
    CompressedAdcFile, CompressedAdcFileError, compress_adc_file, decompress_adc_file,
    open_compressed_adc,
};
