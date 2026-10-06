//! Default raCAAll (CFAR_LOW_BW undefined), including source edge recurrences.
use super::{AZIMUTH_BINS, CaponDiagnostics};

pub(super) struct Candidate {
    pub range: usize,
    pub angle: usize,
    pub noise: f32,
}

pub(super) fn detect(
    power: &[f32],
    ranges: usize,
    diagnostics: &mut CaponDiagnostics,
) -> Vec<Candidate> {
    let mut output = Vec::new();
    let maxima: Vec<f32> = (0..ranges)
        .map(|r| {
            (0..AZIMUTH_BINS)
                .map(|a| power[a * ranges + r])
                .fold(0., f32::max)
        })
        .collect();
    for a in 2..AZIMUTH_BINS - 2 {
        let p = &power[a * ranges..(a + 1) * ranges];
        let mut evaluate = |r: usize, left: f64, right: f64| {
            let noise = (left as f32).min(right as f32) * 0.125;
            if p[r] <= noise * 5. {
                return;
            }
            diagnostics.range_candidates += 1;
            let threshold = angle_noise(power, ranges, a, r) * (8_f32 / 12.);
            let fallback = p[r] > power[(a - 1) * ranges + r]
                && p[r] > power[(a + 1) * ranges + r]
                && p[r] > 0.4 * maxima[r];
            if p[r] > threshold || fallback {
                diagnostics.accepted_before_capacity += 1;
                if output.len() < 150 {
                    output.push(Candidate {
                        range: r,
                        angle: a,
                        noise,
                    });
                } else {
                    diagnostics.capacity_dropped += 1;
                }
            }
        };
        let mut left = 0_f64;
        let mut right = 0_f64;
        // The source tests range 4 INSIDE its partial-window initialization loop.
        for k in (8..16).step_by(2) {
            left += f64::from(p[k]) + f64::from(p[k + 1]);
            right = left - f64::from(p[8]) + f64::from(p[16]);
            evaluate(4, left, right);
        }
        for r in 5..ranges - 4 {
            if r < 9 {
                left -= f64::from(p[r + 3]);
                left += f64::from(p[r + 11]);
            } else if r < 17 {
                left -= f64::from(p[r + 3]);
                left += f64::from(p[r - 5]);
            } else {
                left -= f64::from(p[r - 13]);
                left += f64::from(p[r - 5]);
            }
            if r < ranges - 16 {
                right += f64::from(p[r + 12]);
                right -= f64::from(p[r + 4]);
            } else if r < ranges - 8 {
                right += f64::from(p[r - 4]);
                right -= f64::from(p[r + 4]);
            } else {
                right += f64::from(p[r - 4]);
                right -= f64::from(p[r - 12]);
            }
            evaluate(r, left, right);
        }
    }
    output
}

fn angle_noise(p: &[f32], ranges: usize, a: usize, r: usize) -> f32 {
    let sum = |first: std::ops::Range<usize>, second: std::ops::Range<usize>| -> f32 {
        first.chain(second).map(|i| p[i * ranges + r]).sum()
    };
    let (left, right) = if a < 20 {
        let q = 20 - a;
        (
            sum(
                AZIMUTH_BINS - q..AZIMUTH_BINS - 6,
                0..12_usize.saturating_sub(q),
            ),
            sum(a + 9..a + 21, 0..0),
        )
    } else if a >= AZIMUTH_BINS - 20 {
        let q = (28 + a + 1).saturating_sub(AZIMUTH_BINS);
        (
            sum(a - 12..a, 0..0),
            sum(0..q, (a + 9).min(AZIMUTH_BINS)..AZIMUTH_BINS),
        )
    } else {
        (sum(a - 20..a - 8, 0..0), sum(a + 9..a + 21, 0..0))
    };
    left.min(right)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_partial_window_duplicates_and_edge_noise_are_visible() {
        let ranges = 64;
        let mut p = vec![1_f32; AZIMUTH_BINS * ranges];
        p[50 * ranges + 4] = 100.;
        let mut d = CaponDiagnostics::default();
        let found = detect(&p, ranges, &mut d);
        let edge: Vec<_> = found
            .iter()
            .filter(|c| c.angle == 50 && c.range == 4)
            .collect();
        assert_eq!(edge.len(), 4);
        assert_eq!(
            edge.iter().map(|c| c.noise).collect::<Vec<_>>(),
            [0.25, 0.5, 0.75, 1.]
        );
        for a in 2..AZIMUTH_BINS - 2 {
            p[a * ranges + 30] = 100.;
        }
        let mut d = CaponDiagnostics::default();
        let found = detect(&p, ranges, &mut d);
        // A flat angular ridge fails angle CFAR and strict neighbor fallback.
        assert!(!found.iter().any(|c| c.range == 30));
        assert_eq!(angle_noise(&vec![1.; p.len()], ranges, 8, 30), 6.);
        assert_eq!(
            angle_noise(&vec![1.; p.len()], ranges, AZIMUTH_BINS - 20, 30),
            12.
        );
    }

    #[test]
    fn capacity_preserves_angle_major_order_instead_of_power_sort() {
        let ranges = 128;
        let mut p = vec![1_f32; AZIMUTH_BINS * ranges];
        for a in (2..AZIMUTH_BINS - 2).step_by(3) {
            for r in [30, 60, 90] {
                p[a * ranges + r] = 100.;
            }
        }
        let mut d = CaponDiagnostics::default();
        let found = detect(&p, ranges, &mut d);
        assert_eq!(found.len(), 150);
        assert!(d.capacity_dropped > 0);
        assert_eq!(d.accepted_before_capacity, 150 + d.capacity_dropped);
        assert!(
            found
                .windows(2)
                .all(|pair| (pair[0].angle, pair[0].range) <= (pair[1].angle, pair[1].range))
        );
        assert_eq!((found[0].angle, found[0].range), (2, 30));
    }
}
