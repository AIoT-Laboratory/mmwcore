//! Optional development oracle. Not compiled into normal builds.
use super::{Config, Report, Target};
use libloading::Library;
use sha2::{Digest, Sha256};
use std::{ffi::c_void, path::Path};
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

pub struct Reference {
    handle: *mut c_void,
    step_fn: Step,
    static_start_fn: Option<unsafe extern "C" fn(*mut c_void, u32) -> i32>,
    delete_fn: Delete,
    config: Config,
    pub provenance: serde_json::Value,
    // Keep the library alive until after delete_fn has released its TI instance.
    _library: Library,
}

// Each engine owns one independent TI module. Calls require &mut self; no C mutable
// shared mutable state is used (the optional host context is thread-local).
// Python additionally uses Mutex.
unsafe impl Send for Reference {}

impl Reference {
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
            let static_start_fn = library
                .get::<unsafe extern "C" fn(*mut c_void, u32) -> i32>(b"mmw_ti_static_start\0")
                .ok()
                .map(|symbol| *symbol);
            let mut error = 0;
            let handle = create(&config, &mut error);
            if handle.is_null() {
                return Err(format!("TI gtrack_create failed: {error}"));
            }
            Ok(Self {
                handle,
                step_fn,
                static_start_fn,
                delete_fn,
                config,
                provenance: manifest,
                _library: library,
            })
        }
    }

    pub fn step_raw(
        &mut self,
        points: &[[f32; 5]],
        variances: Option<&[[f32; 4]]>,
        static_start: Option<usize>,
    ) -> Result<Report, String> {
        let n = points.len();
        if let Some(start) = static_start {
            let set = self
                .static_start_fn
                .ok_or("Rebuild the TI plugin for static support")?;
            // SAFETY: start is checked against the validated input length/capacity.
            if unsafe { set(self.handle, start as u32) } != 0 {
                return Err("TI static-support setup failed".into());
            }
        }
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
            return Err(format!("TI gtrack_step failed: {code}; reset required"));
        }
        result.targets.truncate(target_count as usize);
        Ok(result)
    }
}
impl Drop for Reference {
    fn drop(&mut self) {
        // SAFETY: the unique handle was created by this still-loaded library.
        unsafe { (self.delete_fn)(self.handle) };
    }
}
