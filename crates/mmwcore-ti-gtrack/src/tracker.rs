// Copyright (C) 2024 Texas Instruments Incorporated.
// Rust adaptation of GTRACK 3DA; see ../TI-LICENSE.txt.
use super::{
    Config, Report,
    association::{Assignment, Association},
    math::*,
    unit::{TrackState, Unit},
};
use std::collections::VecDeque;

/// Frame storage is reused; only the returned report owns newly allocated output.
struct Frame {
    points: Vec<[f32; 5]>,
    associations: Vec<Association>,
    static_hits: Vec<bool>,
    rpc_hits: Vec<bool>,
    candidate: Vec<usize>,
    selected: Vec<usize>,
    good: Vec<usize>,
}

impl Frame {
    fn new(c: &Config) -> Self {
        let capacity = c.max_points as usize;
        Self {
            points: Vec::with_capacity(capacity),
            associations: Vec::with_capacity(capacity),
            static_hits: vec![false; c.max_tracks as usize],
            rpc_hits: vec![false; c.max_tracks as usize],
            candidate: Vec::with_capacity(capacity),
            selected: Vec::with_capacity(capacity),
            good: Vec::with_capacity(capacity),
        }
    }

    fn reset(&mut self, points: &[[f32; 5]]) {
        self.points.clear();
        self.points.extend_from_slice(points);
        self.associations.clear();
        self.associations
            .resize(points.len(), Association::default());
        self.static_hits.fill(false);
        self.rpc_hits.fill(false);
        self.candidate.clear();
        self.selected.clear();
    }
}

pub struct Tracker {
    tracks: Tracks,
    frame: Frame,
}

struct Tracks {
    units: Vec<Unit>,
    active: Vec<usize>,
    free: VecDeque<usize>,
    next_id: u32,
    presence: bool,
    presence_misses: u16,
    presence_initial: bool,
    f: Matrix<9, 9>,
    q: Matrix<9, 9>,
}

impl Tracker {
    pub fn new(c: &Config) -> Self {
        Self {
            tracks: Tracks::new(c),
            frame: Frame::new(c),
        }
    }

    pub fn step(
        &mut self,
        input: &[[f32; 5]],
        variance: Option<&[[f32; 4]]>,
        static_start: Option<usize>,
        c: &Config,
    ) -> Report {
        let Self { tracks, frame } = self;
        frame.reset(input);
        for (point, association) in frame.points.iter().zip(&mut frame.associations) {
            let world = world(&position(&row(point)), c);
            if boresight(point, c)
                || !inside(&world, &c.boundary_boxes, c.boundary_count)
                || (point[3].abs() <= f32::MIN_POSITIVE
                    && c.static_count != 0
                    && !inside(&world, &c.static_boxes, c.static_count))
            {
                association.assignment = Assignment::Excluded;
            }
        }
        let rpc_count = static_start.unwrap_or(frame.points.len());
        for &uid in &tracks.active {
            tracks.units[uid].predict(c, &tracks.f, &tracks.q);
        }
        for &uid in &tracks.active {
            tracks.units[uid].score(
                &mut frame.points[..rpc_count],
                &mut frame.associations[..rpc_count],
                c,
            );
        }
        if rpc_count < frame.points.len() {
            tracks.support_static(frame, rpc_count, c);
        }
        let mut present = tracks.allocate(frame, c);
        for &uid in &tracks.active {
            tracks.units[uid].update(
                &frame.points,
                variance,
                &frame.associations,
                c,
                frame.static_hits[uid],
                &mut frame.good,
            );
        }
        // FIFO free slots and allocation order are observable through UID/TID.
        tracks.active.retain(|&uid| {
            if tracks.units[uid].state == TrackState::Free {
                tracks.free.push_back(uid);
                false
            } else {
                true
            }
        });
        if c.presence_points != 0 && c.occupancy_count != 0 {
            present |= tracks.active.iter().any(|&uid| {
                inside(
                    &world(&tracks.units[uid].center_position, c),
                    &c.occupancy_boxes,
                    c.occupancy_count,
                )
            });
        }
        tracks.update_presence(present, c.presence_on_to_off);
        Report {
            targets: tracks
                .active
                .iter()
                .map(|&uid| tracks.units[uid].report(c))
                .collect(),
            sensor_targets: Vec::new(),
            point_uid: frame
                .associations
                .iter()
                .map(|a| a.assignment.label())
                .collect(),
            point_score: frame.associations.iter().map(|a| a.score).collect(),
            point_unique: frame
                .associations
                .iter()
                .map(|a| u8::from(a.unique))
                .collect(),
            point_static: frame
                .associations
                .iter()
                .map(|a| u8::from(a.static_point))
                .collect(),
            point_tid: vec![-1; frame.points.len()],
            updated_doppler: frame.points.iter().map(|p| p[3]).collect(),
            presence: u32::from(tracks.presence),
            benchmark_ticks: [0; 7],
        }
    }
}

impl Tracks {
    fn new(c: &Config) -> Self {
        let (f, q) = motion(c);
        Self {
            units: (0..c.max_tracks).map(|_| Unit::default()).collect(),
            active: Vec::with_capacity(c.max_tracks as usize),
            free: (0..c.max_tracks as usize).collect(),
            next_id: 0,
            presence: false,
            presence_misses: 0,
            presence_initial: true,
            f,
            q,
        }
    }

    fn update_presence(&mut self, present: bool, threshold: u32) {
        if present {
            self.presence = true;
            self.presence_misses = 0;
        } else if self.presence {
            self.presence_misses = self.presence_misses.wrapping_add(1);
            if u32::from(self.presence_misses) >= threshold {
                self.presence_initial = false;
                self.presence = false;
                self.presence_misses = 0;
            }
        }
    }

    fn support_static(&self, frame: &mut Frame, start: usize, c: &Config) {
        for association in &frame.associations[..start] {
            if let Assignment::Track(uid) = association.assignment {
                frame.rpc_hits[uid] = true;
            }
        }
        for (point, association) in frame.points[start..]
            .iter()
            .zip(&mut frame.associations[start..])
        {
            association.assignment = Assignment::Excluded;
            if boresight(point, c)
                || !inside(
                    &world(&position(&row(point)), c),
                    &c.boundary_boxes,
                    c.boundary_count,
                )
            {
                continue;
            }
            let mut matches = 0;
            let mut winner = None;
            for &uid in &self.active {
                let unit = &self.units[uid];
                let gate = unit.gate(c);
                let Some(candidate) = gate
                    .candidate(point, c)
                    .filter(|candidate| candidate.doppler.abs() <= f32::EPSILON)
                else {
                    continue;
                };
                matches += 1;
                if unit.state == TrackState::Active
                    && !frame.rpc_hits[uid]
                    && inside(
                        &world(&unit.predicted_state[..3], c),
                        &c.boundary_boxes,
                        c.boundary_count,
                    )
                {
                    winner = Some((uid, candidate.score, gate.slow));
                }
            }
            if matches == 1
                && let Some((uid, score, is_static)) = winner
            {
                association.assignment = Assignment::Track(uid);
                association.score = score;
                association.static_point = is_static;
                frame.static_hits[uid] = true;
            }
        }
    }

    fn allocate(&mut self, frame: &mut Frame, c: &Config) -> bool {
        let Frame {
            points,
            associations,
            candidate,
            selected,
            ..
        } = frame;
        if self.free.is_empty() {
            return false;
        }
        let mut center = [0.0; 4];
        let mut snr = 0.0;
        let available = |i: usize| {
            associations[i].assignment == Assignment::Unassigned
                && points[i][3].abs() >= f32::EPSILON
        };
        for n in 0..points.len() {
            if !available(n) {
                continue;
            }
            candidate.clear();
            candidate.push(n);
            let mut mean = row(&points[n]);
            let mut sum = mean;
            let mut total_snr = points[n][4];
            for (k, point) in points.iter().enumerate().skip(n + 1) {
                if !available(k) {
                    continue;
                }
                let mut u = row(point);
                u[3] = unroll(c.max_velocity, mean[3], u[3]);
                if (u[3] - mean[3]).abs() < c.allocation_max_velocity
                    && distance(&mean, &u).sqrt() < c.allocation_distance
                {
                    candidate.push(k);
                    total_snr += point[4];
                    for i in 0..4 {
                        sum[i] += u[i];
                        mean[i] = sum[i] * (1.0 / candidate.len() as f32);
                    }
                }
            }
            if candidate.len() > selected.len().max(1) {
                std::mem::swap(selected, candidate);
                center = mean;
                snr = total_snr;
            }
        }
        if selected.is_empty() {
            return false;
        }
        let pos = world(&position(&center), c);
        let present = c.presence_points != 0
            && c.occupancy_count != 0
            && selected.len() >= c.presence_points as usize
            && (self.presence_initial || center[3] <= -c.presence_velocity)
            && inside(&pos, &c.occupancy_boxes, c.occupancy_count);
        if selected.len() < c.allocation_points as usize || center[3].abs() < c.allocation_velocity
        {
            return present;
        }
        let behind = self.active.iter().any(|&uid| {
            let unit = &self.units[uid];
            let u = unit.measurement_center;
            (center[1] - u[1]).abs() < unit.spread[1] / 2.0
                && (center[2] - u[2]).abs() < unit.spread[2] / 2.0
                && (center[3] - u[3]).abs() < 2.0 * unit.spread[3]
                && center[0] > u[0]
        });
        let snr_max = (6.0_f32 / 2.5) * (6.0 / 2.5) * (6.0 / 2.5) * (6.0 / 2.5) * c.allocation_snr;
        let snr_fixed = snr_max / 3.0;
        let ratio = 6.0 / center[0];
        let ratio4 = ratio * ratio * ratio * ratio;
        let threshold = if center[0] < 1.0 {
            snr_fixed
        } else if behind || !inside(&pos, &c.static_boxes, c.static_count) {
            ratio4 * c.allocation_obscured_snr
        } else if center[0] < 2.5 {
            center[0] * (snr_max - snr_fixed) / (2.5 - 1.0)
        } else {
            ratio4 * c.allocation_snr
        };
        if snr > threshold {
            let uid = self.free.pop_front().unwrap();
            for n in selected.iter().copied() {
                associations[n].assignment = Assignment::Track(uid);
            }
            self.next_id = self.next_id.wrapping_add(1);
            self.units[uid].start(uid, self.next_id, center, selected.len(), behind, c);
            self.active.push(uid);
        }
        present
    }
}

fn row(point: &[f32; 5]) -> [f32; 4] {
    [point[0], point[1], point[2], point[3]]
}
