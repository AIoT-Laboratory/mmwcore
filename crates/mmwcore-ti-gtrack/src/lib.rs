//! Safe host API around the versioned, locally built TI-device-only plugin.
//! Unsafe code is confined to the ABI call boundary; TI code is never bundled.

use std::ffi::c_void;
use std::path::Path;

use libloading::Library;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[repr(C)]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub max_points: u32,
    pub max_tracks: u32,
    pub delta_t: f32,
    pub initial_velocity: f32,
    pub max_velocity: f32,
    pub velocity_resolution: f32,
    pub max_acceleration: [f32; 3],
    pub boresight_filtering: u32,
    pub gating_gain: f32,
    pub gating_limits: [f32; 4],
    pub allocation_snr: f32,
    pub allocation_obscured_snr: f32,
    pub allocation_velocity: f32,
    pub allocation_points: u32,
    pub allocation_distance: f32,
    pub allocation_max_velocity: f32,
    pub state_thresholds: [u32; 6],
    pub sensor_position: [f32; 3],
    pub sensor_orientation: [f32; 2],
    pub boundary_count: u32,
    pub static_count: u32,
    pub occupancy_count: u32,
    pub boundary_boxes: [f32; 12],
    pub static_boxes: [f32; 12],
    pub occupancy_boxes: [f32; 12],
    pub presence_points: u32,
    pub presence_on_to_off: u32,
    pub presence_velocity: f32,
}

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=1000).contains(&self.max_points) || !(1..=200).contains(&self.max_tracks) {
            return Err("TI capacity requires 1..1000 points and 1..200 tracks".into());
        }
        if self.boundary_count == 0 {
            return Err("At least one world boundary box is required; zero boxes mean always outside in pinned TI".into());
        }
        for (name, value) in [
            ("delta_t", self.delta_t),
            ("max_velocity", self.max_velocity),
            ("velocity_resolution", self.velocity_resolution),
            ("gating_gain", self.gating_gain),
            ("allocation_distance", self.allocation_distance),
            ("allocation_max_velocity", self.allocation_max_velocity),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and positive"));
            }
        }
        if self
            .max_acceleration
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err("max_acceleration must be finite and positive".into());
        }
        if !self.initial_velocity.is_finite()
            || self
                .sensor_position
                .iter()
                .chain(&self.sensor_orientation)
                .any(|x| !x.is_finite())
        {
            return Err("initial velocity and installation must be finite".into());
        }
        if self.sensor_position[0] != 0.0 || self.sensor_position[1] != 0.0 {
            return Err("Pinned TI transform supports height only; scene horizontal origin must be the sensor".into());
        }
        for value in self.gating_limits.iter().chain([
            &self.allocation_snr,
            &self.allocation_obscured_snr,
            &self.allocation_velocity,
            &self.presence_velocity,
        ]) {
            if !value.is_finite() || *value < 0.0 {
                return Err("TI limits and thresholds must be finite and non-negative".into());
            }
        }
        if self.boresight_filtering > 1
            || self.allocation_points == 0
            || self.allocation_points > self.max_points
            || self.presence_points > self.max_points
            || self.presence_on_to_off > 65535
            || self.state_thresholds.iter().any(|x| *x > 65535)
        {
            return Err("Invalid TI integer threshold or flag".into());
        }
        for (count, boxes) in [
            (self.boundary_count, self.boundary_boxes),
            (self.static_count, self.static_boxes),
            (self.occupancy_count, self.occupancy_boxes),
        ] {
            if count > 2 {
                return Err("TI supports at most two boxes of each kind".into());
            }
            if boxes.iter().any(|x| !x.is_finite()) {
                return Err("Scene boxes must be finite".into());
            }
            for b in boxes[..count as usize * 6].chunks_exact(6) {
                if b[0] >= b[1] || b[2] >= b[3] || b[4] >= b[5] {
                    return Err("Scene box bounds must increase".into());
                }
            }
        }
        Ok(())
    }
}

/// Original TI axes: right, forward, up; EC is inverse group covariance.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Target {
    pub uid: u32,
    pub tid: u32,
    pub state: u32,
    pub velocity_state: u32,
    pub is_static: u32,
    pub snr_weighting: u32,
    pub height_ignore: u32,
    pub point_number_estimation: u32,
    pub counters: [u32; 6],
    pub age: u64,
    pub state_vector: [f32; 9],
    pub state_covariance: [[f32; 9]; 9],
    #[serde(rename = "apriori_state_after_step")]
    pub predicted_state: [f32; 9],
    #[serde(rename = "apriori_covariance_after_step")]
    pub predicted_covariance: [[f32; 9]; 9],
    pub predicted_measurement: [f32; 4],
    pub ec: [[f32; 4]; 4],
    pub group_covariance: [[f32; 4]; 4],
    pub group_dispersion: [[f32; 4]; 4],
    pub gain: f32,
    pub dimensions: [f32; 4],
    pub measurement_center: [f32; 4],
    pub confidence: f32,
    pub expected_points: f32,
    pub range_rate: f32,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub targets: Vec<Target>,
    pub sensor_targets: Vec<SensorTarget>,
    pub point_uid: Vec<u8>,
    pub point_tid: Vec<i64>,
    pub point_unique: Vec<u8>,
    pub point_static: Vec<u8>,
    pub point_score: Vec<f32>,
    pub updated_doppler: Vec<f32>,
    pub presence: u32,
    pub benchmark_ticks: [u32; 7],
}

/// Workspace view in sensor forward/right/up, retaining the full TI report above.
#[derive(Debug, Serialize)]
pub struct SensorTarget {
    pub state_vector: [f32; 9],
    pub position_covariance: [[f32; 3]; 3],
    pub extent_covariance: [[f32; 3]; 3],
}

fn sensor_target(target: &Target) -> SensorTarget {
    let axes = [1, 0, 2, 4, 3, 5, 7, 6, 8];
    let state_vector = std::array::from_fn(|i| target.state_vector[axes[i]]);
    let position_covariance =
        std::array::from_fn(|i| std::array::from_fn(|j| target.state_covariance[axes[i]][axes[j]]));
    let [r, a, e, _] = target.measurement_center;
    let (sa, ca) = a.sin_cos();
    let (se, ce) = e.sin_cos();
    // Project measured spherical group dispersion, not TI EC (which is an inverse).
    let j = [
        [ce * ca, -r * ce * sa, -r * se * ca],
        [ce * sa, r * ce * ca, -r * se * sa],
        [se, 0.0, r * ce],
    ];
    let mut extent_covariance = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    extent_covariance[row][col] +=
                        j[row][k] * target.group_dispersion[k][l] * j[col][l];
                }
            }
        }
    }
    SensorTarget {
        state_vector,
        position_covariance,
        extent_covariance,
    }
}

type Abi = unsafe extern "C" fn(u32) -> u32;
type Create = unsafe extern "C" fn(*const Config, *mut i32) -> *mut c_void;
type Delete = unsafe extern "C" fn(*mut c_void);
type Step = unsafe extern "C" fn(
    *mut c_void,
    *const f32,
    *const f32,
    u32,
    *mut Target,
    *mut u32,
    *mut u8,
    *mut u8,
    *mut u8,
    *mut f32,
    *mut f32,
    *mut u32,
    *mut u32,
) -> i32;

pub struct Engine {
    handle: *mut c_void,
    step_fn: Step,
    delete_fn: Delete,
    config: Config,
    provenance: serde_json::Value,
    poisoned: bool,
    // Keep the library alive until after delete_fn has released its TI instance.
    _library: Library,
}

// Each engine owns one independent TI module. Calls require &mut self; no C mutable
// globals are used (the pinned defaults are const). Python additionally uses Mutex.
unsafe impl Send for Engine {}

impl Engine {
    pub fn load(manifest_path: &Path, config: Config) -> Result<Self, String> {
        config.validate()?;
        let manifest_path = manifest_path.canonicalize().map_err(|e| e.to_string())?;
        let bytes = std::fs::read(&manifest_path).map_err(|e| e.to_string())?;
        let manifest: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if manifest["schema"] != "mmwcore.ti-gtrack-plugin.v1" || manifest["abi"] != 1 {
            return Err("Unsupported TI plugin manifest schema/ABI".into());
        }
        let name = manifest["library"]
            .as_str()
            .ok_or("Missing plugin library")?;
        if Path::new(name).components().count() != 1 || Path::new(name).is_absolute() {
            return Err("TI library must be a filename next to its manifest".into());
        }
        let path = manifest_path
            .parent()
            .ok_or("Missing plugin directory")?
            .join(name);
        let content = std::fs::read(&path).map_err(|e| e.to_string())?;
        let digest = format!("{:x}", Sha256::digest(&content));
        if manifest["library_sha256"].as_str() != Some(digest.as_str()) {
            return Err("TI plugin binary hash does not match its build manifest".into());
        }
        // SAFETY: loading an explicitly selected locally built native plugin is the
        // trust boundary. Its hash, ABI version and both structure sizes are checked
        // before passing pointers. Symbol signatures are fixed by bridge.h.
        unsafe {
            let library = Library::new(&path).map_err(|e| e.to_string())?;
            let abi = *library
                .get::<Abi>(b"mmw_ti_abi\0")
                .map_err(|e| e.to_string())?;
            if abi(0) != 1
                || abi(1) as usize != size_of::<Config>()
                || abi(2) as usize != size_of::<Target>()
            {
                return Err("TI plugin ABI/structure layout mismatch".into());
            }
            let create = *library
                .get::<Create>(b"mmw_ti_create\0")
                .map_err(|e| e.to_string())?;
            let delete_fn = *library
                .get::<Delete>(b"mmw_ti_delete\0")
                .map_err(|e| e.to_string())?;
            let step_fn = *library
                .get::<Step>(b"mmw_ti_step\0")
                .map_err(|e| e.to_string())?;
            let mut error = 0;
            let handle = create(&config, &mut error);
            if handle.is_null() {
                return Err(format!("TI gtrack_create failed: {error}"));
            }
            Ok(Self {
                handle,
                step_fn,
                delete_fn,
                config,
                provenance: manifest,
                poisoned: false,
                _library: library,
            })
        }
    }

    pub fn provenance(&self) -> &serde_json::Value {
        &self.provenance
    }

    /// Cartesian contract is sensor forward/right/up, with velocity and linear SNR.
    pub fn step_cartesian(
        &mut self,
        points: &[[f32; 5]],
        variances: Option<&[[f32; 4]]>,
    ) -> Result<Report, String> {
        let spherical: Vec<_> = points
            .iter()
            .map(|p| {
                let range = p[0].hypot(p[1]).hypot(p[2]);
                [
                    range,
                    p[1].atan2(p[0]),
                    p[2].atan2(p[0].hypot(p[1])),
                    p[3],
                    p[4],
                ]
            })
            .collect();
        self.step(&spherical, variances)
    }

    /// Spherical input rows: range, azimuth right, elevation up, radial velocity,
    /// linear SNR. Variance rows have the same four measurement dimensions.
    pub fn step(
        &mut self,
        points: &[[f32; 5]],
        variances: Option<&[[f32; 4]]>,
    ) -> Result<Report, String> {
        if self.poisoned {
            return Err(
                "TI tracker encountered non-finite state; reset before another step".into(),
            );
        }
        if points.len() > self.config.max_points as usize {
            return Err("Frame exceeds configured TI max_points; no points were truncated".into());
        }
        for p in points {
            if p.iter().any(|x| !x.is_finite())
                || p[0] <= 0.0
                || p[4] <= 0.0
                || p[1].abs() >= std::f32::consts::FRAC_PI_2
                || p[2].abs() >= std::f32::consts::FRAC_PI_2
            {
                return Err("TI measurements need positive range/SNR and finite forward-hemisphere angles/velocity".into());
            }
        }
        if let Some(v) = variances
            && (v.len() != points.len() || v.iter().flatten().any(|x| !x.is_finite() || *x <= 0.0))
        {
            return Err("Explicit measurement variances must be positive finite (N,4) values; omit them if unknown".into());
        }
        let n = points.len();
        let mut result = Report {
            targets: vec![Target::default(); self.config.max_tracks as usize],
            sensor_targets: Vec::new(),
            point_uid: vec![255; n],
            point_tid: vec![-1; n],
            point_unique: vec![0; n],
            point_static: vec![0; n],
            point_score: vec![0.0; n],
            updated_doppler: vec![0.0; n],
            presence: 0,
            benchmark_ticks: [0; 7],
        };
        let mut target_count = 0;
        // SAFETY: config and row counts were checked. All buffers have the exact
        // capacities required by the pinned C ABI; the host copies const inputs.
        let code = unsafe {
            (self.step_fn)(
                self.handle,
                points.as_ptr().cast(),
                variances.map_or(std::ptr::null(), |v| v.as_ptr().cast()),
                n as u32,
                result.targets.as_mut_ptr(),
                &mut target_count,
                result.point_uid.as_mut_ptr(),
                result.point_unique.as_mut_ptr(),
                result.point_static.as_mut_ptr(),
                result.point_score.as_mut_ptr(),
                result.updated_doppler.as_mut_ptr(),
                &mut result.presence,
                result.benchmark_ticks.as_mut_ptr(),
            )
        };
        if code != 0 || target_count as usize > result.targets.len() {
            self.poisoned = true;
            return Err(format!("TI gtrack_step failed: {code}; reset required"));
        }
        result.targets.truncate(target_count as usize);
        result.sensor_targets = result.targets.iter().map(sensor_target).collect();
        if result.targets.iter().any(|t| !target_finite(t))
            || result
                .updated_doppler
                .iter()
                .chain(&result.point_score)
                .any(|x| !x.is_finite())
            || result
                .sensor_targets
                .iter()
                .any(|t| t.extent_covariance.iter().flatten().any(|x| !x.is_finite()))
        {
            self.poisoned = true;
            return Err(
                "TI gtrack_step produced non-finite state; no report emitted, reset required"
                    .into(),
            );
        }
        let mut tid_by_uid = [-1_i64; 256];
        for target in &result.targets {
            if let Some(tid) = tid_by_uid.get_mut(target.uid as usize)
                && *tid == -1
            {
                *tid = i64::from(target.tid);
            }
        }
        for (uid, tid) in result.point_uid.iter().zip(&mut result.point_tid) {
            *tid = tid_by_uid[usize::from(*uid)];
        }
        Ok(result)
    }
}

fn target_finite(t: &Target) -> bool {
    t.state_vector
        .iter()
        .chain(&t.predicted_state)
        .chain(&t.predicted_measurement)
        .chain(t.state_covariance.iter().flatten())
        .chain(t.predicted_covariance.iter().flatten())
        .chain(t.ec.iter().flatten())
        .chain(t.group_covariance.iter().flatten())
        .chain(t.group_dispersion.iter().flatten())
        .chain(&t.dimensions)
        .chain(&t.measurement_center)
        .chain([&t.gain, &t.confidence, &t.expected_points, &t.range_rate])
        .all(|x| x.is_finite())
}

impl Drop for Engine {
    fn drop(&mut self) {
        // SAFETY: the unique handle was created by this still-loaded library.
        unsafe { (self.delete_fn)(self.handle) };
    }
}
