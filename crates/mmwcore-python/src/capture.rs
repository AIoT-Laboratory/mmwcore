//! PyO3 boundary for native ADC and capture-format parsing.

use numpy::ndarray::{Array1, Array4};
use numpy::{Complex32, IntoPyArray, PyArray1, PyArray4, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyAny};

use super::{
    AdcComplexLayout, AdcFrameSpec, Dca1000AssemblyResult, Dca1000Packet,
    assemble_native_dca1000_frame_bytes, dca1000_assembly_result, dca1000_error, decode_error,
    decode_native_adc_i16, parse_native_dca1000_packet, reorder_native_dca1000_packets,
};

#[pyfunction]
fn decode_adc_i16<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray1<'py, i16>,
    num_chirps: usize,
    num_rx: usize,
    num_samples: usize,
    layout: u8,
    drop_incomplete: bool,
) -> PyResult<Bound<'py, PyArray4<Complex32>>> {
    let samples = samples
        .as_slice()
        .map_err(|_| PyValueError::new_err("ADC samples must be a contiguous int16 array."))?
        .to_vec();
    let layout = AdcComplexLayout::try_from(layout).map_err(decode_error)?;
    let spec = AdcFrameSpec::new(num_chirps, num_rx, num_samples, layout).map_err(decode_error)?;
    let cube = py
        .detach(move || decode_native_adc_i16(&samples, spec, drop_incomplete))
        .map_err(decode_error)?;
    let [frames, chirps, receivers, samples] = cube.shape();
    let array = Array4::from_shape_vec((frames, chirps, receivers, samples), cube.into_data())
        .map_err(|_| PyValueError::new_err("Native ADC cube shape is invalid."))?;

    Ok(array.into_pyarray(py))
}

#[pyfunction]
fn parse_dca1000_packet(
    py: Python<'_>,
    data: Vec<u8>,
) -> PyResult<(u32, u64, Bound<'_, PyArray1<i16>>)> {
    let packet = py
        .detach(move || parse_native_dca1000_packet(&data))
        .map_err(dca1000_error)?;
    let packet_number = packet.packet_number();
    let byte_count = packet.byte_count();
    let payload = Array1::from_vec(packet.into_payload()).into_pyarray(py);

    Ok((packet_number, byte_count, payload))
}

#[pyfunction]
fn reorder_dca1000_packets<'py>(
    py: Python<'py>,
    packet_numbers: PyReadonlyArray1<'py, u32>,
    payloads: Vec<PyReadonlyArray1<'py, i16>>,
    frame_start_packet_number: u32,
    packets_per_frame: usize,
    payload_values_per_packet: Option<usize>,
    fill_value: i16,
) -> PyResult<Dca1000AssemblyResult<'py>> {
    let packet_numbers = packet_numbers
        .as_slice()
        .map_err(|_| PyValueError::new_err("DCA1000 packet_numbers must be contiguous uint32."))?;
    if packet_numbers.len() != payloads.len() {
        return Err(PyValueError::new_err(
            "DCA1000 packet_numbers and payloads must have the same length.",
        ));
    }
    let mut packets = Vec::with_capacity(packet_numbers.len());
    for (index, (packet_number, payload)) in packet_numbers.iter().zip(payloads).enumerate() {
        let payload = payload.as_slice().map_err(|_| {
            PyValueError::new_err(format!(
                "DCA1000 payload {index} must be a contiguous int16 array."
            ))
        })?;
        packets
            .push(Dca1000Packet::new(*packet_number, 0, payload.to_vec()).map_err(dca1000_error)?);
    }

    let assembly = py
        .detach(move || {
            reorder_native_dca1000_packets(
                &packets,
                frame_start_packet_number,
                packets_per_frame,
                payload_values_per_packet,
                fill_value,
            )
        })
        .map_err(dca1000_error)?;
    dca1000_assembly_result(py, assembly)
}

#[pyfunction]
fn assemble_dca1000_frame_bytes<'py>(
    py: Python<'py>,
    packets: &Bound<'py, PyAny>,
    raw_values_per_frame: usize,
    payload_values_per_packet: usize,
    frame_start_byte_count: u64,
) -> PyResult<Dca1000AssemblyResult<'py>> {
    const BYTE_COUNT_MODULUS: u64 = 1_u64 << 48;
    const PACKET_HEADER_BYTES: usize = 10;
    if raw_values_per_frame == 0 {
        return Err(PyValueError::new_err(
            "raw_values_per_frame must be positive.",
        ));
    }
    if payload_values_per_packet == 0 {
        return Err(PyValueError::new_err(
            "payload_values_per_packet must be positive.",
        ));
    }
    if !raw_values_per_frame.is_multiple_of(payload_values_per_packet) {
        return Err(PyValueError::new_err(
            "raw_values_per_frame must be divisible by payload_values_per_packet.",
        ));
    }
    if frame_start_byte_count >= BYTE_COUNT_MODULUS {
        return Err(PyValueError::new_err(
            "frame_start_byte_count must fit the unsigned 48-bit wire counter.",
        ));
    }
    let packets_per_frame = raw_values_per_frame / payload_values_per_packet;
    let packet_count = u64::try_from(packets_per_frame)
        .map_err(|_| PyValueError::new_err("packets_per_frame exceeds u64."))?;
    if packet_count > u64::from(u32::MAX) + 1 {
        return Err(PyValueError::new_err(
            "packets_per_frame exceeds the unsigned 32-bit sequence space.",
        ));
    }
    let frame_bytes = raw_values_per_frame
        .checked_mul(size_of::<i16>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| PyValueError::new_err("DCA1000 frame byte span overflows."))?;
    if frame_bytes > BYTE_COUNT_MODULUS {
        return Err(PyValueError::new_err(
            "DCA1000 frame byte span exceeds the unsigned 48-bit wire counter.",
        ));
    }
    let actual_packet_count = packets.len()?;
    if actual_packet_count != packets_per_frame {
        return Err(PyValueError::new_err(format!(
            "Stateless DCA1000 frame assembly requires exactly {packets_per_frame} packet(s); got {actual_packet_count}."
        )));
    }
    let expected_packet_bytes = payload_values_per_packet
        .checked_mul(size_of::<i16>())
        .and_then(|bytes| PACKET_HEADER_BYTES.checked_add(bytes))
        .ok_or_else(|| PyValueError::new_err("DCA1000 packet byte size overflows."))?;
    for index in 0..actual_packet_count {
        let item = packets.get_item(index)?;
        let packet = item
            .cast::<pyo3::types::PyBytes>()
            .map_err(|_| PyValueError::new_err(format!("DCA1000 packet {index} must be bytes.")))?;
        if packet.as_bytes().len() != expected_packet_bytes {
            return Err(PyValueError::new_err(format!(
                "DCA1000 packet {index} contains {} bytes; expected exactly {expected_packet_bytes}.",
                packet.as_bytes().len()
            )));
        }
    }

    let mut owned_packets = Vec::new();
    owned_packets
        .try_reserve_exact(actual_packet_count)
        .map_err(|_| pyo3::exceptions::PyMemoryError::new_err("Cannot reserve DCA1000 packets."))?;
    for index in 0..actual_packet_count {
        let item = packets.get_item(index)?;
        let packet = item
            .cast::<pyo3::types::PyBytes>()
            .map_err(|_| PyValueError::new_err(format!("DCA1000 packet {index} must be bytes.")))?;
        let bytes = packet.as_bytes();
        if bytes.len() != expected_packet_bytes {
            return Err(PyValueError::new_err(format!(
                "DCA1000 packet {index} changed size during validation."
            )));
        }
        let mut owned_packet = Vec::new();
        owned_packet.try_reserve_exact(bytes.len()).map_err(|_| {
            pyo3::exceptions::PyMemoryError::new_err(format!(
                "Cannot reserve DCA1000 packet {index}."
            ))
        })?;
        owned_packet.extend_from_slice(bytes);
        owned_packets.push(owned_packet);
    }
    let assembly = py
        .detach(move || {
            assemble_native_dca1000_frame_bytes(
                &owned_packets,
                raw_values_per_frame,
                payload_values_per_packet,
                frame_start_byte_count,
            )
        })
        .map_err(dca1000_error)?;
    dca1000_assembly_result(py, assembly)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(decode_adc_i16, module)?)?;
    module.add_function(wrap_pyfunction!(parse_dca1000_packet, module)?)?;
    module.add_function(wrap_pyfunction!(reorder_dca1000_packets, module)?)?;
    module.add_function(wrap_pyfunction!(assemble_dca1000_frame_bytes, module)?)?;
    Ok(())
}
