//! PyO3 boundary for native ADC and capture-format parsing.

use super::*;

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
) -> PyResult<(i64, u64, Bound<'_, PyArray1<i16>>)> {
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
    packet_numbers: PyReadonlyArray1<'py, i64>,
    payloads: Vec<PyReadonlyArray1<'py, i16>>,
    packets_per_frame: usize,
    payload_values_per_packet: Option<usize>,
    fill_value: i16,
) -> PyResult<Dca1000AssemblyResult<'py>> {
    let packet_numbers = packet_numbers
        .as_slice()
        .map_err(|_| PyValueError::new_err("DCA1000 packet_numbers must be contiguous int64."))?;
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
                packets_per_frame,
                payload_values_per_packet,
                fill_value,
            )
        })
        .map_err(dca1000_error)?;
    dca1000_assembly_result(py, assembly)
}

#[pyfunction]
fn assemble_dca1000_frame_bytes(
    py: Python<'_>,
    packets: Vec<Vec<u8>>,
    raw_values_per_frame: usize,
    payload_values_per_packet: usize,
    fill_value: i16,
) -> PyResult<Dca1000AssemblyResult<'_>> {
    let assembly = py
        .detach(move || {
            assemble_native_dca1000_frame_bytes(
                &packets,
                raw_values_per_frame,
                payload_values_per_packet,
                fill_value,
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
