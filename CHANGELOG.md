# Changelog

All notable changes to TNIE will be documented in this file. The project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- CMake build with a reusable `tnie` library target.
- Cross-platform GitHub Actions workflow.
- Runtime, validation, and regression tests.
- Checked inference API with explicit input and output sizes.
- Public status codes, network validation, and status descriptions.

### Changed

- The XOR model now lives under `examples/xor/` instead of the public runtime.
- Documentation now distinguishes implemented features from the roadmap.
- Internal inference buffers account for both input and layer output sizes.
