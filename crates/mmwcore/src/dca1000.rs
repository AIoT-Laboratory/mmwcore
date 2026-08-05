//! Deterministic DCA1000 packet parsing and frame assembly.

use std::collections::BTreeSet;
use std::fmt;

pub const DCA1000_PACKET_HEADER_BYTES: usize = 10;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Dca1000Packet {
    packet_number: i64,
    byte_count: u64,
    payload: Vec<i16>,
}

impl Dca1000Packet {
    pub fn new(
        packet_number: i64,
        byte_count: u64,
        payload: Vec<i16>,
    ) -> Result<Self, Dca1000Error> {
        if packet_number <= 0 {
            return Err(Dca1000Error::NonPositivePacketNumber(packet_number));
        }
        Ok(Self {
            packet_number,
            byte_count,
            payload,
        })
    }

    pub const fn packet_number(&self) -> i64 {
        self.packet_number
    }

    pub const fn byte_count(&self) -> u64 {
        self.byte_count
    }

    pub fn payload(&self) -> &[i16] {
        &self.payload
    }

    pub fn into_payload(self) -> Vec<i16> {
        self.payload
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PacketLossStats {
    pub expected_packets: usize,
    pub received_packets: usize,
    pub missing_packet_numbers: Vec<i64>,
    pub duplicate_packet_numbers: Vec<i64>,
    pub out_of_frame_packet_numbers: Vec<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Dca1000FrameAssembly {
    samples: Vec<i16>,
    stats: PacketLossStats,
}

impl Dca1000FrameAssembly {
    pub fn samples(&self) -> &[i16] {
        &self.samples
    }

    pub fn stats(&self) -> &PacketLossStats {
        &self.stats
    }

    pub fn into_parts(self) -> (Vec<i16>, PacketLossStats) {
        (self.samples, self.stats)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Dca1000Error {
    PacketTooShort { bytes: usize },
    OddPayloadByteCount { bytes: usize },
    NonPositivePacketNumber(i64),
    EmptyPackets,
    InvalidPacketsPerFrame { packets_per_frame: usize },
    InvalidPayloadValuesPerPacket { payload_values_per_packet: usize },
    InvalidRawValuesPerFrame { raw_values_per_frame: usize },
    FrameBufferOverflow,
    PacketNumberRangeOverflow,
}

impl fmt::Display for Dca1000Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PacketTooShort { bytes } => write!(
                formatter,
                "DCA1000 packet is shorter than the 10-byte header; got {bytes} bytes."
            ),
            Self::OddPayloadByteCount { bytes } => write!(
                formatter,
                "DCA1000 packet payload must contain whole int16 values; got {bytes} payload byte(s)."
            ),
            Self::NonPositivePacketNumber(packet_number) => write!(
                formatter,
                "DCA1000Packet.packet_number must be positive; got {packet_number}."
            ),
            Self::EmptyPackets => write!(formatter, "packets must not be empty."),
            Self::InvalidPacketsPerFrame { packets_per_frame } => write!(
                formatter,
                "packets_per_frame must be positive; got {packets_per_frame}."
            ),
            Self::InvalidPayloadValuesPerPacket {
                payload_values_per_packet,
            } => write!(
                formatter,
                "payload_values_per_packet must be positive; got {payload_values_per_packet}."
            ),
            Self::InvalidRawValuesPerFrame {
                raw_values_per_frame,
            } => write!(
                formatter,
                "raw_values_per_frame must be positive; got {raw_values_per_frame}."
            ),
            Self::FrameBufferOverflow => {
                write!(formatter, "DCA1000 frame buffer size overflows usize.")
            }
            Self::PacketNumberRangeOverflow => {
                write!(formatter, "DCA1000 packet number range overflows i64.")
            }
        }
    }
}

impl std::error::Error for Dca1000Error {}

pub fn parse_dca1000_packet(data: &[u8]) -> Result<Dca1000Packet, Dca1000Error> {
    if data.len() < DCA1000_PACKET_HEADER_BYTES {
        return Err(Dca1000Error::PacketTooShort { bytes: data.len() });
    }

    let packet_number = i64::from(i32::from_le_bytes(
        data[..4]
            .try_into()
            .expect("DCA1000 header length was validated"),
    ));
    let byte_count =
        u64::from_le_bytes([data[4], data[5], data[6], data[7], data[8], data[9], 0, 0]);
    let payload_bytes = &data[DCA1000_PACKET_HEADER_BYTES..];
    if payload_bytes.len() % size_of::<i16>() != 0 {
        return Err(Dca1000Error::OddPayloadByteCount {
            bytes: payload_bytes.len(),
        });
    }
    let payload = payload_bytes
        .chunks_exact(size_of::<i16>())
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .collect();

    Dca1000Packet::new(packet_number, byte_count, payload)
}

pub fn reorder_dca1000_packets(
    packets: &[Dca1000Packet],
    packets_per_frame: usize,
    payload_values_per_packet: Option<usize>,
    fill_value: i16,
) -> Result<Dca1000FrameAssembly, Dca1000Error> {
    if packets_per_frame == 0 {
        return Err(Dca1000Error::InvalidPacketsPerFrame { packets_per_frame });
    }
    if packets.is_empty() {
        return Err(Dca1000Error::EmptyPackets);
    }

    let values_per_packet = payload_values_per_packet.unwrap_or_else(|| {
        packets
            .iter()
            .map(|packet| packet.payload().len())
            .max()
            .unwrap_or_default()
    });
    if values_per_packet == 0 {
        return Err(Dca1000Error::InvalidPayloadValuesPerPacket {
            payload_values_per_packet: values_per_packet,
        });
    }

    let frame_len = packets_per_frame
        .checked_mul(values_per_packet)
        .ok_or(Dca1000Error::FrameBufferOverflow)?;
    let frame_start = packets
        .iter()
        .map(Dca1000Packet::packet_number)
        .min()
        .expect("non-empty packets were validated");
    let frame_end = frame_start
        .checked_add(
            i64::try_from(packets_per_frame - 1)
                .map_err(|_| Dca1000Error::PacketNumberRangeOverflow)?,
        )
        .ok_or(Dca1000Error::PacketNumberRangeOverflow)?;

    let mut samples = vec![fill_value; frame_len];
    let mut seen = BTreeSet::new();
    let mut duplicates = Vec::new();
    let mut out_of_frame = Vec::new();

    for packet in packets {
        let packet_number = packet.packet_number();
        if packet_number < frame_start || packet_number > frame_end {
            out_of_frame.push(packet_number);
            continue;
        }
        if !seen.insert(packet_number) {
            duplicates.push(packet_number);
            continue;
        }

        let relative_index = usize::try_from(packet_number - frame_start)
            .map_err(|_| Dca1000Error::FrameBufferOverflow)?;
        let start = relative_index
            .checked_mul(values_per_packet)
            .ok_or(Dca1000Error::FrameBufferOverflow)?;
        let payload = packet.payload();
        let copy_len = payload.len().min(values_per_packet);
        samples[start..start + copy_len].copy_from_slice(&payload[..copy_len]);
    }

    let mut missing = Vec::new();
    for offset in 0..packets_per_frame {
        let expected_packet = frame_start
            .checked_add(
                i64::try_from(offset).map_err(|_| Dca1000Error::PacketNumberRangeOverflow)?,
            )
            .ok_or(Dca1000Error::PacketNumberRangeOverflow)?;
        if !seen.contains(&expected_packet) {
            missing.push(expected_packet);
        }
    }

    Ok(Dca1000FrameAssembly {
        samples,
        stats: PacketLossStats {
            expected_packets: packets_per_frame,
            received_packets: packets.len(),
            missing_packet_numbers: missing,
            duplicate_packet_numbers: duplicates,
            out_of_frame_packet_numbers: out_of_frame,
        },
    })
}

pub fn assemble_dca1000_frame(
    packets: &[Dca1000Packet],
    raw_values_per_frame: usize,
    payload_values_per_packet: usize,
    fill_value: i16,
) -> Result<Dca1000FrameAssembly, Dca1000Error> {
    if raw_values_per_frame == 0 {
        return Err(Dca1000Error::InvalidRawValuesPerFrame {
            raw_values_per_frame,
        });
    }
    if payload_values_per_packet == 0 {
        return Err(Dca1000Error::InvalidPayloadValuesPerPacket {
            payload_values_per_packet,
        });
    }

    let packets_per_frame = raw_values_per_frame / payload_values_per_packet
        + usize::from(raw_values_per_frame % payload_values_per_packet != 0);
    let mut assembly = reorder_dca1000_packets(
        packets,
        packets_per_frame,
        Some(payload_values_per_packet),
        fill_value,
    )?;
    assembly.samples.truncate(raw_values_per_frame);
    Ok(assembly)
}

pub fn assemble_dca1000_frame_bytes(
    packets: &[Vec<u8>],
    raw_values_per_frame: usize,
    payload_values_per_packet: usize,
    fill_value: i16,
) -> Result<Dca1000FrameAssembly, Dca1000Error> {
    let parsed_packets = packets
        .iter()
        .map(|packet| parse_dca1000_packet(packet))
        .collect::<Result<Vec<_>, _>>()?;
    assemble_dca1000_frame(
        &parsed_packets,
        raw_values_per_frame,
        payload_values_per_packet,
        fill_value,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        Dca1000Error, Dca1000Packet, assemble_dca1000_frame, assemble_dca1000_frame_bytes,
        parse_dca1000_packet, reorder_dca1000_packets,
    };

    #[test]
    fn parses_little_endian_header_and_payload() {
        let packet =
            parse_dca1000_packet(&[7, 0, 0, 0, 176, 5, 0, 0, 0, 0, 1, 0, 254, 255, 3, 0]).unwrap();

        assert_eq!(packet.packet_number(), 7);
        assert_eq!(packet.byte_count(), 1456);
        assert_eq!(packet.payload(), [1, -2, 3]);
    }

    #[test]
    fn rejects_short_or_misaligned_packets() {
        assert_eq!(
            parse_dca1000_packet(&[1, 0]),
            Err(Dca1000Error::PacketTooShort { bytes: 2 })
        );
        assert_eq!(
            parse_dca1000_packet(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            Err(Dca1000Error::OddPayloadByteCount { bytes: 1 })
        );
    }

    #[test]
    fn reorders_packets_and_records_packet_loss() {
        let packets = [
            Dca1000Packet::new(5, 0, vec![50]).unwrap(),
            Dca1000Packet::new(7, 0, vec![70]).unwrap(),
            Dca1000Packet::new(7, 0, vec![71]).unwrap(),
            Dca1000Packet::new(9, 0, vec![90]).unwrap(),
        ];

        let assembly = reorder_dca1000_packets(&packets, 3, Some(1), -1).unwrap();

        assert_eq!(assembly.samples(), [50, -1, 70]);
        assert_eq!(assembly.stats().missing_packet_numbers, [6]);
        assert_eq!(assembly.stats().duplicate_packet_numbers, [7]);
        assert_eq!(assembly.stats().out_of_frame_packet_numbers, [9]);
    }

    #[test]
    fn assembles_and_truncates_to_adc_frame_size() {
        let packets = [
            Dca1000Packet::new(2, 4, vec![3, 4]).unwrap(),
            Dca1000Packet::new(1, 0, vec![1, 2]).unwrap(),
        ];

        let assembly = assemble_dca1000_frame(&packets, 4, 2, 0).unwrap();

        assert_eq!(assembly.samples(), [1, 2, 3, 4]);
        assert_eq!(assembly.stats().received_packets, 2);
    }

    #[test]
    fn parses_and_assembles_packet_bytes_in_one_call() {
        let packets = vec![
            vec![2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 3, 0, 4, 0],
            vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0],
        ];

        let assembly = assemble_dca1000_frame_bytes(&packets, 4, 2, 0).unwrap();

        assert_eq!(assembly.samples(), [1, 2, 3, 4]);
    }
}
