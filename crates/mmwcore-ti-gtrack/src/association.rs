// Copyright (C) 2024 Texas Instruments Incorporated.
// Rust adaptation of GTRACK 3DA; see ../TI-LICENSE.txt.
use super::{Config, math::*};

/// Only the report boundary uses TI's reserved byte labels. Ghost marking is
/// disabled in the pinned 3DA source, so no unreachable ghost state is modeled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Assignment {
    #[default]
    Unassigned,
    Excluded,
    Track(usize),
}

impl Assignment {
    pub fn label(self) -> u8 {
        match self {
            Self::Unassigned => 255,
            Self::Excluded => 254,
            Self::Track(uid) => uid as u8,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Association {
    pub assignment: Assignment,
    pub score: f32,
    pub unique: bool,
    // TI's first association bookkeeping flag, not a zero-Doppler classifier.
    pub static_point: bool,
}

impl Default for Association {
    fn default() -> Self {
        Self {
            assignment: Assignment::Unassigned,
            score: f32::MAX,
            unique: true,
            static_point: false,
        }
    }
}

pub(super) struct Candidate {
    pub score: f32,
    pub doppler: f32,
}

/// A read-only association query. Static support uses the same gate without
/// temporarily changing track counters, point Doppler or the reported EC cache.
pub(super) struct Gate<'a> {
    pub predicted: &'a [f32; 4],
    pub inverse: &'a Matrix<4, 4>,
    pub bounds: [f32; 4],
    pub gain: f32,
    pub logdet: f32,
    pub slow: bool,
}

impl Gate<'_> {
    pub fn candidate(&self, point: &[f32; 5], c: &Config) -> Option<Candidate> {
        let predicted = if ceiling(c) {
            let mut projected = position(self.predicted);
            projected[1] = position(&[point[0], point[1], point[2], point[3]])[1];
            spherical(&projected)
        } else {
            *self.predicted
        };
        let mut residual = std::array::from_fn(|i| point[i] - predicted[i]);
        let doppler = unroll(c.max_velocity, self.predicted[3], point[3]);
        residual[3] = doppler - self.predicted[3];
        if residual.iter().zip(self.bounds).any(|(r, b)| r.abs() > b) {
            return None;
        }
        let threshold = if doppler.abs() <= f32::EPSILON {
            2.0
        } else if self.slow {
            1.0
        } else {
            self.gain
        };
        if mahalanobis(&residual, self.inverse, 3) >= threshold {
            return None;
        }
        residual[3] *= 3.0;
        Some(Candidate {
            score: self.logdet + mahalanobis(&residual, self.inverse, 4),
            doppler,
        })
    }

    pub fn associate(
        &self,
        uid: usize,
        points: &mut [[f32; 5]],
        associations: &mut [Association],
        c: &Config,
    ) -> usize {
        let mut dynamic = 0;
        for (point, association) in points.iter_mut().zip(associations) {
            if association.assignment == Assignment::Excluded {
                continue;
            }
            let Some(candidate) = self.candidate(point, c) else {
                continue;
            };
            dynamic += usize::from(candidate.doppler.abs() > f32::EPSILON);
            if association.assignment == Assignment::Unassigned {
                association.assignment = Assignment::Track(uid);
                association.score = candidate.score;
                association.static_point = self.slow;
                point[3] = candidate.doppler;
            } else {
                let clear_unique = if candidate.score < association.score {
                    association.assignment = Assignment::Track(uid);
                    association.score = candidate.score;
                    point[3] = candidate.doppler;
                    self.slow || !association.static_point
                } else {
                    association.static_point || !self.slow
                };
                // Preserve the first static flag even when another track wins.
                association.unique &= !clear_unique;
            }
        }
        dynamic
    }
}
