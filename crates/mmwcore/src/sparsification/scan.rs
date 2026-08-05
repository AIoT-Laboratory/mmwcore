//! Parallel Doppler-slab analysis for Cartesian sparsification.

use super::{
    Candidate, CandidateScan, CartesianSparsificationConfig, CartesianSparsificationError,
    CartesianSparsificationInput, checked_product, snr_db,
};

pub(super) fn analyze_doppler_slabs(
    input: CartesianSparsificationInput<'_>,
    config: CartesianSparsificationConfig,
    spatial_shape: [usize; 3],
    valid_spatial: &[bool],
    worker_count: usize,
) -> Result<(Vec<f32>, CandidateScan), CartesianSparsificationError> {
    if worker_count == 1 {
        return Ok(analyze_doppler_range(
            input,
            config,
            spatial_shape,
            valid_spatial,
            0,
            input.shape_dzyx[0],
        ));
    }
    let chunk_size = input.shape_dzyx[0].div_ceil(worker_count);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for start in (0..input.shape_dzyx[0]).step_by(chunk_size) {
            let stop = (start + chunk_size).min(input.shape_dzyx[0]);
            handles.push(scope.spawn(move || {
                analyze_doppler_range(input, config, spatial_shape, valid_spatial, start, stop)
            }));
        }
        let mut floors = Vec::with_capacity(input.shape_dzyx[0]);
        let mut scan = CandidateScan::default();
        for handle in handles {
            let (chunk_floors, chunk_scan) = handle
                .join()
                .map_err(|_| CartesianSparsificationError::WorkerPanicked)?;
            floors.extend(chunk_floors);
            scan.extend(chunk_scan);
        }
        Ok((floors, scan))
    })
}

fn analyze_doppler_range(
    input: CartesianSparsificationInput<'_>,
    config: CartesianSparsificationConfig,
    spatial_shape: [usize; 3],
    valid_spatial: &[bool],
    doppler_start: usize,
    doppler_stop: usize,
) -> (Vec<f32>, CandidateScan) {
    let floors = noise_floor_range(
        input,
        valid_spatial,
        config.noise_floor_scale,
        doppler_start,
        doppler_stop,
    );
    let scan = scan_candidate_range(
        input,
        config,
        spatial_shape,
        valid_spatial,
        &floors,
        doppler_start,
    );
    (floors, scan)
}

fn noise_floor_range(
    input: CartesianSparsificationInput<'_>,
    valid_spatial: &[bool],
    scale: f32,
    doppler_start: usize,
    doppler_stop: usize,
) -> Vec<f32> {
    let spatial_size = valid_spatial.len();
    (doppler_start..doppler_stop)
        .map(|doppler| {
            let mut log_sum = 0.0_f64;
            let mut positive_count = 0_usize;
            for (spatial, &spatial_valid) in valid_spatial.iter().enumerate() {
                let magnitude = magnitude_at(input, doppler, spatial, spatial_size);
                if spatial_valid && magnitude > 0.0 {
                    log_sum += f64::from(magnitude).ln();
                    positive_count += 1;
                }
            }
            if positive_count == 0 {
                0.0
            } else {
                ((log_sum / positive_count as f64).exp() * f64::from(scale)) as f32
            }
        })
        .collect()
}

fn scan_candidate_range(
    input: CartesianSparsificationInput<'_>,
    config: CartesianSparsificationConfig,
    spatial_shape: [usize; 3],
    valid_spatial: &[bool],
    noise_floors: &[f32],
    doppler_start: usize,
) -> CandidateScan {
    let spatial_size = valid_spatial.len();
    let mut scan = CandidateScan::default();
    for (doppler_offset, &noise_floor) in noise_floors.iter().enumerate() {
        let doppler = doppler_start + doppler_offset;
        for (spatial, &spatial_valid) in valid_spatial.iter().enumerate() {
            let index = doppler * spatial_size + spatial;
            let magnitude = magnitude_at(input, doppler, spatial, spatial_size);
            if magnitude <= 0.0 {
                continue;
            }
            scan.positive_volume_voxels += 1;
            let doppler_peak = doppler_local_maximum(
                input,
                input.shape_dzyx[0],
                spatial_size,
                doppler,
                spatial,
                config.doppler_peak_radius,
            );
            if doppler_peak {
                scan.doppler_peak_voxels += 1;
            }
            if !spatial_valid {
                continue;
            }
            scan.valid_positive_volume_voxels += 1;
            if !doppler_peak
                || !spatial_local_maximum(
                    input.magnitude_dzyx,
                    spatial_shape,
                    doppler,
                    spatial,
                    config.spatial_peak_radius,
                )
            {
                continue;
            }
            scan.local_peak_voxels += 1;
            let snr_db = snr_db(magnitude, noise_floor);
            if snr_db < config.min_snr_db {
                continue;
            }
            scan.threshold_peak_voxels += 1;
            scan.candidates.push(Candidate { index, snr_db });
        }
    }
    scan
}

fn doppler_local_maximum(
    input: CartesianSparsificationInput<'_>,
    doppler_bins: usize,
    spatial_size: usize,
    doppler: usize,
    spatial: usize,
    radius: usize,
) -> bool {
    if radius == 0 {
        return true;
    }
    let lower = doppler.saturating_sub(radius);
    let upper = (doppler.saturating_add(radius)).min(doppler_bins - 1);
    let value = magnitude_at(input, doppler, spatial, spatial_size);
    for neighbor_doppler in lower..=upper {
        if neighbor_doppler == doppler {
            continue;
        }
        let neighbor = magnitude_at(input, neighbor_doppler, spatial, spatial_size);
        if value < neighbor || (value == neighbor && doppler > neighbor_doppler) {
            return false;
        }
    }
    true
}

fn magnitude_at(
    input: CartesianSparsificationInput<'_>,
    doppler: usize,
    spatial: usize,
    spatial_size: usize,
) -> f32 {
    if input.suppressed_doppler_index == Some(doppler) {
        0.0
    } else {
        input.magnitude_dzyx[doppler * spatial_size + spatial]
    }
}

fn spatial_local_maximum(
    values: &[f32],
    shape_zyx: [usize; 3],
    doppler: usize,
    spatial: usize,
    radius: usize,
) -> bool {
    if radius == 0 {
        return true;
    }
    let [z_size, y_size, x_size] = shape_zyx;
    let plane_size = checked_product(&shape_zyx).expect("validated Cartesian shape");
    let z = spatial / (y_size * x_size);
    let remaining = spatial % (y_size * x_size);
    let y = remaining / x_size;
    let x = remaining % x_size;
    let lower_z = z.saturating_sub(radius);
    let upper_z = (z.saturating_add(radius)).min(z_size - 1);
    let lower_y = y.saturating_sub(radius);
    let upper_y = (y.saturating_add(radius)).min(y_size - 1);
    let lower_x = x.saturating_sub(radius);
    let upper_x = (x.saturating_add(radius)).min(x_size - 1);
    let value = values[doppler * plane_size + spatial];
    for neighbor_z in lower_z..=upper_z {
        for neighbor_y in lower_y..=upper_y {
            for neighbor_x in lower_x..=upper_x {
                let neighbor_spatial = (neighbor_z * y_size + neighbor_y) * x_size + neighbor_x;
                if neighbor_spatial == spatial {
                    continue;
                }
                let neighbor = values[doppler * plane_size + neighbor_spatial];
                if value < neighbor || (value == neighbor && spatial > neighbor_spatial) {
                    return false;
                }
            }
        }
    }
    true
}
