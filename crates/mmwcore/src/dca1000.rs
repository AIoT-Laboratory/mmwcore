//! Deterministic DCA1000 packet parsing and frame assembly.

use std::collections::BTreeSet;
use std::fmt;

pub const DCA1000_PACKET_HEADER_BYTES: usize = 10;
pub const DCA1000_BYTE_COUNT_MODULUS: u64 = 1_u64 << 48;
pub const DCA1000_BYTE_COUNT_MASK: u64 = DCA1000_BYTE_COUNT_MODULUS - 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Dca1000Packet {
    packet_number: u32,
    byte_count: u64,
    payload: Vec<i16>,
}

impl Dca1000Packet {
    pub fn new(
        packet_number: u32,
        byte_count: u64,
        payload: Vec<i16>,
    ) -> Result<Self, Dca1000Error> {
        if byte_count > DCA1000_BYTE_COUNT_MASK {
            return Err(Dca1000Error::ByteCountOutOfRange { byte_count });
        }
        Ok(Self {
            packet_number,
            byte_count,
            payload,
        })
    }

    pub const fn packet_number(&self) -> u32 {
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
    PacketTooShort {
        bytes: usize,
    },
    OddPayloadByteCount {
        bytes: usize,
    },
    ByteCountOutOfRange {
        byte_count: u64,
    },
    EmptyPackets,
    InvalidPacketsPerFrame {
        packets_per_frame: usize,
    },
    InvalidPayloadValuesPerPacket {
        payload_values_per_packet: usize,
    },
    InvalidRawValuesPerFrame {
        raw_values_per_frame: usize,
    },
    NonIntegralFramePacketCount {
        raw_values_per_frame: usize,
        payload_values_per_packet: usize,
    },
    UnexpectedFramePacketCount {
        expected: usize,
        actual: usize,
    },
    UnexpectedPacketPayloadLength {
        packet_number: u32,
        expected: usize,
        actual: usize,
    },
    UnexpectedPacketByteCount {
        packet_number: u32,
        expected: u64,
        actual: u64,
    },
    UnexpectedFramePacketNumber {
        expected: u32,
        actual: u32,
    },
    FrameBufferOverflow,
    ByteCountRangeOverflow,
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
            Self::ByteCountOutOfRange { byte_count } => write!(
                formatter,
                "DCA1000 byte_count must fit the unsigned 48-bit wire counter; got {byte_count}."
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
            Self::NonIntegralFramePacketCount {
                raw_values_per_frame,
                payload_values_per_packet,
            } => write!(
                formatter,
                "raw_values_per_frame must be divisible by payload_values_per_packet for stateless DCA1000 frame assembly; got {raw_values_per_frame} and {payload_values_per_packet}."
            ),
            Self::UnexpectedFramePacketCount { expected, actual } => write!(
                formatter,
                "Stateless DCA1000 frame assembly requires exactly {expected} packet(s); got {actual}."
            ),
            Self::UnexpectedPacketPayloadLength {
                packet_number,
                expected,
                actual,
            } => write!(
                formatter,
                "DCA1000 packet {packet_number} payload contains {actual} int16 value(s); expected exactly {expected}."
            ),
            Self::UnexpectedPacketByteCount {
                packet_number,
                expected,
                actual,
            } => write!(
                formatter,
                "DCA1000 packet {packet_number} has byte_count {actual}; expected {expected}."
            ),
            Self::UnexpectedFramePacketNumber { expected, actual } => write!(
                formatter,
                "DCA1000 packet sequence has packet_number {actual}; expected {expected} in byte_count order."
            ),
            Self::FrameBufferOverflow => {
                write!(formatter, "DCA1000 frame buffer size overflows usize.")
            }
            Self::ByteCountRangeOverflow => {
                write!(
                    formatter,
                    "DCA1000 frame byte span exceeds the unsigned 48-bit wire counter."
                )
            }
            Self::PacketNumberRangeOverflow => {
                write!(
                    formatter,
                    "DCA1000 packets_per_frame exceeds the unsigned 32-bit sequence space."
                )
            }
        }
    }
}

impl std::error::Error for Dca1000Error {}

pub fn parse_dca1000_packet(data: &[u8]) -> Result<Dca1000Packet, Dca1000Error> {
    if data.len() < DCA1000_PACKET_HEADER_BYTES {
        return Err(Dca1000Error::PacketTooShort { bytes: data.len() });
    }

    let packet_number = u32::from_le_bytes(
        data[..4]
            .try_into()
            .expect("DCA1000 header length was validated"),
    );
    let byte_count =
        u64::from_le_bytes([data[4], data[5], data[6], data[7], data[8], data[9], 0, 0]);
    let payload_bytes = &data[DCA1000_PACKET_HEADER_BYTES..];
    if payload_bytes.len() % size_of::<i16>() != 0 {
        return Err(Dca1000Error::OddPayloadByteCount {
            bytes: payload_bytes.len(),
        });
    }
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(payload_bytes.len() / size_of::<i16>())
        .map_err(|_| Dca1000Error::FrameBufferOverflow)?;
    payload.extend(
        payload_bytes
            .chunks_exact(size_of::<i16>())
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]])),
    );

    Dca1000Packet::new(packet_number, byte_count, payload)
}

pub fn reorder_dca1000_packets(
    packets: &[Dca1000Packet],
    frame_start_packet_number: u32,
    packets_per_frame: usize,
    payload_values_per_packet: Option<usize>,
    fill_value: i16,
) -> Result<Dca1000FrameAssembly, Dca1000Error> {
    if packets_per_frame == 0 {
        return Err(Dca1000Error::InvalidPacketsPerFrame { packets_per_frame });
    }
    validate_packet_count(packets_per_frame)?;
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
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(frame_len)
        .map_err(|_| Dca1000Error::FrameBufferOverflow)?;
    samples.resize(frame_len, fill_value);
    let mut seen = BTreeSet::new();
    let mut duplicates = Vec::new();
    let mut out_of_frame = Vec::new();

    for packet in packets {
        let packet_number = packet.packet_number();
        let relative_packet_number = packet_number.wrapping_sub(frame_start_packet_number);
        let Ok(relative_index) = usize::try_from(relative_packet_number) else {
            out_of_frame.push(i64::from(packet_number));
            continue;
        };
        if relative_index >= packets_per_frame {
            out_of_frame.push(i64::from(packet_number));
            continue;
        }
        if !seen.insert(packet_number) {
            duplicates.push(i64::from(packet_number));
            continue;
        }

        let start = relative_index
            .checked_mul(values_per_packet)
            .ok_or(Dca1000Error::FrameBufferOverflow)?;
        let payload = packet.payload();
        let copy_len = payload.len().min(values_per_packet);
        samples[start..start + copy_len].copy_from_slice(&payload[..copy_len]);
    }

    let mut missing = Vec::new();
    for offset in 0..packets_per_frame {
        let offset = u32::try_from(offset).map_err(|_| Dca1000Error::PacketNumberRangeOverflow)?;
        let expected_packet = frame_start_packet_number.wrapping_add(offset);
        if !seen.contains(&expected_packet) {
            missing.push(i64::from(expected_packet));
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
    frame_start_byte_count: u64,
) -> Result<Dca1000FrameAssembly, Dca1000Error> {
    let ordered_packets = validate_exact_frame_packets(
        packets,
        raw_values_per_frame,
        payload_values_per_packet,
        frame_start_byte_count,
    )?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(raw_values_per_frame)
        .map_err(|_| Dca1000Error::FrameBufferOverflow)?;
    for (_, packet) in ordered_packets {
        samples.extend_from_slice(packet.payload());
    }

    let packets_per_frame = packets.len();
    Ok(Dca1000FrameAssembly {
        samples,
        stats: PacketLossStats {
            expected_packets: packets_per_frame,
            received_packets: packets_per_frame,
            missing_packet_numbers: Vec::new(),
            duplicate_packet_numbers: Vec::new(),
            out_of_frame_packet_numbers: Vec::new(),
        },
    })
}

fn validate_exact_frame_packets(
    packets: &[Dca1000Packet],
    raw_values_per_frame: usize,
    payload_values_per_packet: usize,
    frame_start_byte_count: u64,
) -> Result<Vec<(u64, &Dca1000Packet)>, Dca1000Error> {
    let packets_per_frame = validate_exact_frame_config(
        raw_values_per_frame,
        payload_values_per_packet,
        frame_start_byte_count,
    )?;
    if packets.len() != packets_per_frame {
        return Err(Dca1000Error::UnexpectedFramePacketCount {
            expected: packets_per_frame,
            actual: packets.len(),
        });
    }
    for packet in packets {
        if packet.payload().len() != payload_values_per_packet {
            return Err(Dca1000Error::UnexpectedPacketPayloadLength {
                packet_number: packet.packet_number(),
                expected: payload_values_per_packet,
                actual: packet.payload().len(),
            });
        }
    }

    let payload_bytes = payload_values_per_packet
        .checked_mul(size_of::<i16>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(Dca1000Error::FrameBufferOverflow)?;
    let mut ordered_packets = Vec::new();
    ordered_packets
        .try_reserve_exact(packets.len())
        .map_err(|_| Dca1000Error::FrameBufferOverflow)?;
    ordered_packets.extend(packets.iter().map(|packet| {
        let relative_byte_count =
            packet.byte_count().wrapping_sub(frame_start_byte_count) & DCA1000_BYTE_COUNT_MASK;
        (relative_byte_count, packet)
    }));
    ordered_packets.sort_unstable_by_key(|(relative_byte_count, _)| *relative_byte_count);
    for (offset, (relative_byte_count, packet)) in ordered_packets.iter().copied().enumerate() {
        let offset_u64 = u64::try_from(offset).map_err(|_| Dca1000Error::ByteCountRangeOverflow)?;
        let expected_relative_byte_count = payload_bytes
            .checked_mul(offset_u64)
            .ok_or(Dca1000Error::ByteCountRangeOverflow)?;
        let expected_byte_count = frame_start_byte_count.wrapping_add(expected_relative_byte_count)
            & DCA1000_BYTE_COUNT_MASK;
        if relative_byte_count != expected_relative_byte_count {
            return Err(Dca1000Error::UnexpectedPacketByteCount {
                packet_number: packet.packet_number(),
                expected: expected_byte_count,
                actual: packet.byte_count(),
            });
        }
    }
    let first_packet_number = ordered_packets[0].1.packet_number();
    for (offset, (_, packet)) in ordered_packets.iter().copied().enumerate() {
        let offset = u32::try_from(offset).map_err(|_| Dca1000Error::PacketNumberRangeOverflow)?;
        let expected_packet_number = first_packet_number.wrapping_add(offset);
        if packet.packet_number() != expected_packet_number {
            return Err(Dca1000Error::UnexpectedFramePacketNumber {
                expected: expected_packet_number,
                actual: packet.packet_number(),
            });
        }
    }
    Ok(ordered_packets)
}

fn validate_exact_frame_config(
    raw_values_per_frame: usize,
    payload_values_per_packet: usize,
    frame_start_byte_count: u64,
) -> Result<usize, Dca1000Error> {
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
    if raw_values_per_frame % payload_values_per_packet != 0 {
        return Err(Dca1000Error::NonIntegralFramePacketCount {
            raw_values_per_frame,
            payload_values_per_packet,
        });
    }
    if frame_start_byte_count > DCA1000_BYTE_COUNT_MASK {
        return Err(Dca1000Error::ByteCountOutOfRange {
            byte_count: frame_start_byte_count,
        });
    }

    let packets_per_frame = raw_values_per_frame / payload_values_per_packet;
    validate_packet_count(packets_per_frame)?;
    let frame_bytes = raw_values_per_frame
        .checked_mul(size_of::<i16>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(Dca1000Error::FrameBufferOverflow)?;
    if frame_bytes > DCA1000_BYTE_COUNT_MODULUS {
        return Err(Dca1000Error::ByteCountRangeOverflow);
    }
    Ok(packets_per_frame)
}

fn validate_packet_count(packets_per_frame: usize) -> Result<(), Dca1000Error> {
    let packet_count =
        u64::try_from(packets_per_frame).map_err(|_| Dca1000Error::PacketNumberRangeOverflow)?;
    if packet_count > u64::from(u32::MAX) + 1 {
        return Err(Dca1000Error::PacketNumberRangeOverflow);
    }
    Ok(())
}

pub fn assemble_dca1000_frame_bytes(
    packets: &[Vec<u8>],
    raw_values_per_frame: usize,
    payload_values_per_packet: usize,
    frame_start_byte_count: u64,
) -> Result<Dca1000FrameAssembly, Dca1000Error> {
    let packets_per_frame = validate_exact_frame_config(
        raw_values_per_frame,
        payload_values_per_packet,
        frame_start_byte_count,
    )?;
    if packets.len() != packets_per_frame {
        return Err(Dca1000Error::UnexpectedFramePacketCount {
            expected: packets_per_frame,
            actual: packets.len(),
        });
    }
    let expected_packet_bytes = payload_values_per_packet
        .checked_mul(size_of::<i16>())
        .and_then(|bytes| DCA1000_PACKET_HEADER_BYTES.checked_add(bytes))
        .ok_or(Dca1000Error::FrameBufferOverflow)?;
    for packet in packets {
        if packet.len() != expected_packet_bytes {
            if packet.len() < DCA1000_PACKET_HEADER_BYTES {
                return Err(Dca1000Error::PacketTooShort {
                    bytes: packet.len(),
                });
            }
            let payload_bytes = packet.len() - DCA1000_PACKET_HEADER_BYTES;
            if payload_bytes % size_of::<i16>() != 0 {
                return Err(Dca1000Error::OddPayloadByteCount {
                    bytes: payload_bytes,
                });
            }
            let packet_number = u32::from_le_bytes(
                packet[..4]
                    .try_into()
                    .expect("DCA1000 header length was validated"),
            );
            return Err(Dca1000Error::UnexpectedPacketPayloadLength {
                packet_number,
                expected: payload_values_per_packet,
                actual: payload_bytes / size_of::<i16>(),
            });
        }
    }

    let mut parsed_packets = Vec::new();
    parsed_packets
        .try_reserve_exact(packets.len())
        .map_err(|_| Dca1000Error::FrameBufferOverflow)?;
    for packet in packets {
        parsed_packets.push(parse_dca1000_packet(packet)?);
    }
    assemble_dca1000_frame(
        &parsed_packets,
        raw_values_per_frame,
        payload_values_per_packet,
        frame_start_byte_count,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        DCA1000_BYTE_COUNT_MASK, DCA1000_BYTE_COUNT_MODULUS, Dca1000Error, Dca1000Packet,
        assemble_dca1000_frame, assemble_dca1000_frame_bytes, parse_dca1000_packet,
        reorder_dca1000_packets,
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

        let assembly = reorder_dca1000_packets(&packets, 5, 3, Some(1), -1).unwrap();

        assert_eq!(assembly.samples(), [50, -1, 70]);
        assert_eq!(assembly.stats().missing_packet_numbers, [6]);
        assert_eq!(assembly.stats().duplicate_packet_numbers, [7]);
        assert_eq!(assembly.stats().out_of_frame_packet_numbers, [9]);
    }

    #[test]
    fn accepts_wire_counter_zero_and_wraps_u32_reordering() {
        let packets = [
            Dca1000Packet::new(0, 0, vec![2]).unwrap(),
            Dca1000Packet::new(u32::MAX, 0, vec![1]).unwrap(),
            Dca1000Packet::new(2, 0, vec![9]).unwrap(),
        ];

        let assembly = reorder_dca1000_packets(&packets, u32::MAX, 2, Some(1), -1).unwrap();

        assert_eq!(assembly.samples(), [1, 2]);
        assert!(assembly.stats().missing_packet_numbers.is_empty());
        assert_eq!(assembly.stats().out_of_frame_packet_numbers, [2]);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rejects_packet_counts_larger_than_u32_space_before_allocation() {
        let packets = [Dca1000Packet::new(0, 0, vec![1]).unwrap()];
        let packets_per_frame = usize::try_from(u64::from(u32::MAX) + 2).unwrap();

        assert_eq!(
            reorder_dca1000_packets(&packets, 0, packets_per_frame, Some(1), 0),
            Err(Dca1000Error::PacketNumberRangeOverflow)
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn exact_assembly_rejects_packet_counts_larger_than_u32_space_before_allocation() {
        let packets = [Dca1000Packet::new(0, 0, vec![1]).unwrap()];
        let raw_values_per_frame = usize::try_from(u64::from(u32::MAX) + 2).unwrap();

        assert_eq!(
            assemble_dca1000_frame(&packets, raw_values_per_frame, 1, 0),
            Err(Dca1000Error::PacketNumberRangeOverflow)
        );
    }

    #[test]
    fn parses_high_u32_and_rejects_out_of_range_u48_byte_counts() {
        let packet = parse_dca1000_packet(&[255, 255, 255, 255, 0, 0, 0, 0, 0, 0]).unwrap();

        assert_eq!(packet.packet_number(), u32::MAX);
        assert_eq!(
            Dca1000Packet::new(0, DCA1000_BYTE_COUNT_MODULUS, vec![1]),
            Err(Dca1000Error::ByteCountOutOfRange {
                byte_count: DCA1000_BYTE_COUNT_MODULUS,
            })
        );
    }

    #[test]
    fn assembles_one_exact_out_of_order_frame() {
        let packets = [
            Dca1000Packet::new(2, 4, vec![3, 4]).unwrap(),
            Dca1000Packet::new(1, 0, vec![1, 2]).unwrap(),
        ];

        let assembly = assemble_dca1000_frame(&packets, 4, 2, 0).unwrap();

        assert_eq!(assembly.samples(), [1, 2, 3, 4]);
        assert_eq!(assembly.stats().expected_packets, 2);
        assert_eq!(assembly.stats().received_packets, 2);
        assert!(assembly.stats().missing_packet_numbers.is_empty());
        assert!(assembly.stats().duplicate_packet_numbers.is_empty());
        assert!(assembly.stats().out_of_frame_packet_numbers.is_empty());
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

    #[test]
    fn rejects_non_integral_or_wrong_frame_packet_counts() {
        let one_packet = [Dca1000Packet::new(1, 0, vec![1, 2]).unwrap()];
        assert_eq!(
            assemble_dca1000_frame(&one_packet, 3, 2, 0),
            Err(Dca1000Error::NonIntegralFramePacketCount {
                raw_values_per_frame: 3,
                payload_values_per_packet: 2,
            })
        );
        assert_eq!(
            assemble_dca1000_frame(&one_packet, 4, 2, 0),
            Err(Dca1000Error::UnexpectedFramePacketCount {
                expected: 2,
                actual: 1,
            })
        );

        let three_packets = [
            Dca1000Packet::new(1, 0, vec![1, 2]).unwrap(),
            Dca1000Packet::new(2, 4, vec![3, 4]).unwrap(),
            Dca1000Packet::new(3, 8, vec![5, 6]).unwrap(),
        ];
        assert_eq!(
            assemble_dca1000_frame(&three_packets, 4, 2, 0),
            Err(Dca1000Error::UnexpectedFramePacketCount {
                expected: 2,
                actual: 3,
            })
        );
    }

    #[test]
    fn rejects_inexact_payloads_and_unproven_frame_boundaries() {
        let short_payload = [
            Dca1000Packet::new(1, 0, vec![1]).unwrap(),
            Dca1000Packet::new(2, 4, vec![3, 4]).unwrap(),
        ];
        assert_eq!(
            assemble_dca1000_frame(&short_payload, 4, 2, 0),
            Err(Dca1000Error::UnexpectedPacketPayloadLength {
                packet_number: 1,
                expected: 2,
                actual: 1,
            })
        );

        let first_packet_missing = [
            Dca1000Packet::new(2, 4, vec![3, 4]).unwrap(),
            Dca1000Packet::new(3, 8, vec![5, 6]).unwrap(),
        ];
        assert_eq!(
            assemble_dca1000_frame(&first_packet_missing, 4, 2, 0),
            Err(Dca1000Error::UnexpectedPacketByteCount {
                packet_number: 2,
                expected: 0,
                actual: 4,
            })
        );

        let crosses_frame_boundary = [
            Dca1000Packet::new(1, 0, vec![1, 2]).unwrap(),
            Dca1000Packet::new(2, 8, vec![5, 6]).unwrap(),
        ];
        assert_eq!(
            assemble_dca1000_frame(&crosses_frame_boundary, 4, 2, 0),
            Err(Dca1000Error::UnexpectedPacketByteCount {
                packet_number: 2,
                expected: 4,
                actual: 8,
            })
        );

        let duplicate_byte_slot = [
            Dca1000Packet::new(1, 0, vec![1, 2]).unwrap(),
            Dca1000Packet::new(1, 0, vec![3, 4]).unwrap(),
        ];
        assert_eq!(
            assemble_dca1000_frame(&duplicate_byte_slot, 4, 2, 0),
            Err(Dca1000Error::UnexpectedPacketByteCount {
                packet_number: 1,
                expected: 4,
                actual: 0,
            })
        );

        let duplicate_packet_number = [
            Dca1000Packet::new(1, 0, vec![1, 2]).unwrap(),
            Dca1000Packet::new(1, 4, vec![3, 4]).unwrap(),
        ];
        assert_eq!(
            assemble_dca1000_frame(&duplicate_packet_number, 4, 2, 0),
            Err(Dca1000Error::UnexpectedFramePacketNumber {
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn assembles_u32_and_u48_counter_wrap_from_trusted_origin() {
        let packets = [
            Dca1000Packet::new(0, 0, vec![3, 4]).unwrap(),
            Dca1000Packet::new(u32::MAX, DCA1000_BYTE_COUNT_MASK - 3, vec![1, 2]).unwrap(),
        ];

        let assembly = assemble_dca1000_frame(&packets, 4, 2, DCA1000_BYTE_COUNT_MASK - 3).unwrap();

        assert_eq!(assembly.samples(), [1, 2, 3, 4]);
    }

    #[test]
    fn rejects_untrusted_or_out_of_range_frame_origins() {
        let packets = [
            Dca1000Packet::new(1, 4, vec![1, 2]).unwrap(),
            Dca1000Packet::new(2, 8, vec![3, 4]).unwrap(),
        ];
        assert_eq!(
            assemble_dca1000_frame(&packets, 4, 2, 0),
            Err(Dca1000Error::UnexpectedPacketByteCount {
                packet_number: 1,
                expected: 0,
                actual: 4,
            })
        );
        assert_eq!(
            assemble_dca1000_frame(&packets, 4, 2, DCA1000_BYTE_COUNT_MODULUS,),
            Err(Dca1000Error::ByteCountOutOfRange {
                byte_count: DCA1000_BYTE_COUNT_MODULUS,
            })
        );
    }

    #[test]
    fn packet_bytes_route_rejects_inexact_payloads() {
        let packets = vec![
            vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            vec![2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 3, 0, 4, 0],
        ];

        assert_eq!(
            assemble_dca1000_frame_bytes(&packets, 4, 2, 0),
            Err(Dca1000Error::UnexpectedPacketPayloadLength {
                packet_number: 1,
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn packet_bytes_route_rejects_wrong_count_before_parsing() {
        let malformed_packet = vec![vec![0]];

        assert_eq!(
            assemble_dca1000_frame_bytes(&malformed_packet, 4, 2, 0),
            Err(Dca1000Error::UnexpectedFramePacketCount {
                expected: 2,
                actual: 1,
            })
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn packet_bytes_route_rejects_impossible_shape_before_parsing() {
        let malformed_packet = vec![vec![0]];
        let raw_values_per_frame = usize::try_from(u64::from(u32::MAX) + 2).unwrap();

        assert_eq!(
            assemble_dca1000_frame_bytes(&malformed_packet, raw_values_per_frame, 1, 0),
            Err(Dca1000Error::PacketNumberRangeOverflow)
        );
    }
}
