use super::{AdcArchiveCodecError, MAX_RICE_PARAMETER, MAX_ZIGZAG_DELTA};

pub(super) fn best_rice_parameter_and_bit_count(values: &[u32]) -> (u8, u64) {
    let mut costs = [0_u64; MAX_RICE_PARAMETER as usize + 1];
    for &value in values {
        for (parameter, cost) in costs.iter_mut().enumerate() {
            *cost += u64::from(value >> parameter) + 1 + parameter as u64;
        }
    }
    costs
        .iter()
        .enumerate()
        .min_by_key(|(_, cost)| *cost)
        .map(|(parameter, &cost)| (parameter as u8, cost))
        .expect("Rice parameter range is non-empty")
}

#[derive(Default)]
pub(super) struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    used: u8,
}

impl BitWriter {
    pub(super) fn reset(&mut self) {
        self.bytes.clear();
        self.current = 0;
        self.used = 0;
    }

    pub(super) fn try_reserve(&mut self, encoded_bytes: usize) -> Result<(), AdcArchiveCodecError> {
        self.bytes.try_reserve_exact(encoded_bytes).map_err(|_| {
            AdcArchiveCodecError::CannotAllocateOutput {
                expected_bytes: encoded_bytes,
            }
        })
    }

    pub(super) fn write_rice(&mut self, value: u32, parameter: u8) {
        let quotient = value >> parameter;
        self.write_zeroes(quotient as usize);
        self.write_bits(1, 1);
        if parameter > 0 {
            let mask = (1_u32 << parameter) - 1;
            self.write_bits(value & mask, parameter);
        }
    }

    fn write_zeroes(&mut self, mut count: usize) {
        if self.used != 0 {
            let available = usize::from(8 - self.used);
            if count < available {
                self.used += count as u8;
                return;
            }
            self.bytes.push(self.current);
            self.current = 0;
            self.used = 0;
            count -= available;
        }

        let whole_bytes = count / 8;
        if whole_bytes != 0 {
            self.bytes.resize(self.bytes.len() + whole_bytes, 0);
        }
        self.used = (count % 8) as u8;
    }

    fn write_bits(&mut self, value: u32, mut count: u8) {
        while count != 0 {
            if self.used == 0 && count >= 8 {
                let shift = count - 8;
                self.bytes.push((value >> shift) as u8);
                count -= 8;
                continue;
            }

            let available = 8 - self.used;
            let take = count.min(available);
            let shift = count - take;
            let mask = (1_u32 << take) - 1;
            let bits = ((value >> shift) & mask) as u8;
            self.current |= bits << (available - take);
            self.used += take;
            count -= take;
            if self.used == 8 {
                self.bytes.push(self.current);
                self.current = 0;
                self.used = 0;
            }
        }
    }

    pub(super) fn finish(&mut self) -> &[u8] {
        if self.used != 0 {
            self.bytes.push(self.current);
            self.current = 0;
            self.used = 0;
        }
        &self.bytes
    }
}

pub(super) struct BitReader<'a> {
    bytes: &'a [u8],
    bit_index: usize,
}

impl<'a> BitReader<'a> {
    pub(super) const fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            bit_index: 0,
        }
    }

    pub(super) fn read_rice(&mut self, parameter: u8) -> Result<u32, AdcArchiveCodecError> {
        let maximum_quotient = MAX_ZIGZAG_DELTA >> parameter;
        let quotient = self.read_unary(maximum_quotient)?;
        let remainder = self.read_bits(parameter)?;
        quotient
            .checked_shl(u32::from(parameter))
            .and_then(|value| value.checked_add(remainder))
            .ok_or(AdcArchiveCodecError::RiceQuotientOutOfRange)
    }

    fn read_bits(&mut self, count: u8) -> Result<u32, AdcArchiveCodecError> {
        let mut count = count;
        let mut value = 0_u32;
        while count != 0 {
            let byte = *self
                .bytes
                .get(self.bit_index / 8)
                .ok_or(AdcArchiveCodecError::TruncatedBlock)?;
            let offset = (self.bit_index % 8) as u8;
            let available = 8 - offset;
            let take = count.min(available);
            let mask = (1_u16 << take) - 1;
            let bits = u32::from((u16::from(byte) >> (available - take)) & mask);
            value = (value << take) | bits;
            self.bit_index += usize::from(take);
            count -= take;
        }
        Ok(value)
    }

    fn read_unary(&mut self, maximum_quotient: u32) -> Result<u32, AdcArchiveCodecError> {
        let mut quotient = 0_u32;
        loop {
            let byte = *self
                .bytes
                .get(self.bit_index / 8)
                .ok_or(AdcArchiveCodecError::TruncatedBlock)?;
            let offset = (self.bit_index % 8) as u8;
            let available = 8 - offset;
            let remaining_mask = (1_u16 << available) - 1;
            let remaining = (u16::from(byte) & remaining_mask) as u8;
            if remaining == 0 {
                quotient = quotient
                    .checked_add(u32::from(available))
                    .ok_or(AdcArchiveCodecError::RiceQuotientOutOfRange)?;
                if quotient > maximum_quotient {
                    return Err(AdcArchiveCodecError::RiceQuotientOutOfRange);
                }
                self.bit_index += usize::from(available);
                continue;
            }

            let zeroes = remaining.leading_zeros() as u8 - offset;
            quotient = quotient
                .checked_add(u32::from(zeroes))
                .ok_or(AdcArchiveCodecError::RiceQuotientOutOfRange)?;
            if quotient > maximum_quotient {
                return Err(AdcArchiveCodecError::RiceQuotientOutOfRange);
            }
            self.bit_index += usize::from(zeroes) + 1;
            return Ok(quotient);
        }
    }

    pub(super) fn finish_block(&mut self) -> Result<usize, AdcArchiveCodecError> {
        let remainder = self.bit_index % 8;
        if remainder == 0 {
            return Ok(self.bit_index / 8);
        }
        let byte = *self
            .bytes
            .get(self.bit_index / 8)
            .ok_or(AdcArchiveCodecError::TruncatedBlock)?;
        let padding_mask = (1_u16 << (8 - remainder)) - 1;
        if u16::from(byte) & padding_mask != 0 {
            return Err(AdcArchiveCodecError::NonZeroPadding);
        }
        self.bit_index += 8 - remainder;
        Ok(self.bit_index / 8)
    }
}
