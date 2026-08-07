# Architecture

## Boundary

mmwcore is the reusable radar capture-decoding and signal-processing package. It must remain
independent of experiment, application, and model layers in downstream projects.
The core boundary begins at captured bytes and explicit offline contracts. The Python package
retains compatibility hardware/session I/O, but new acquisition behavior belongs in dedicated
tools.

```text
raw bytes / packets
        |
        v
Rust kernels (crates/mmwcore)
        |
        v
PyO3 boundary (crates/mmwcore-python)
        |
        v
Python contracts, I/O, composition, plotting (python/mmwcore)
```

The Rust crate is independently usable. The Python wheel embeds the extension and adds typed
composition, captured-file/session I/O, compatibility hardware adapters, and visual inspection.

## Physical data path

Every transform carries or requires the information needed to interpret its output:

1. ADC layout and frame geometry
2. range and Doppler FFT conventions
3. virtual-array mapping and TDM phase compensation
4. angle calibration
5. detection and quality channels
6. metric point projection
7. clustering and tracking

No model-specific normalization or dataset-specific pose logic belongs here.

## Versioning

Python and Rust packages share one version. Public Python contracts and Rust APIs are versioned
together. Existing schema values remain stable wire identifiers for recorded capture artifacts;
they do not create a runtime dependency on downstream systems.
