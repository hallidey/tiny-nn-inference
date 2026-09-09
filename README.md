# TNIE — Tiny Neural Inference Engine

> Train anywhere. Infer on tiny devices.

TNIE is a small, auditable neural-network inference runtime written in portable
C11. It is designed for learning, prototyping, and eventually deploying small
feed-forward models in sensor and microcontroller applications.

TNIE is intentionally focused: it is not a replacement for a general-purpose
machine-learning framework. Models are trained elsewhere; TNIE runs their
forward pass with no third-party runtime dependencies.

## Current capabilities

- Dense (fully connected) layers
- Linear, ReLU, and sigmoid activations
- Multi-layer feed-forward inference using `float`
- Network and dimension validation
- Checked input and output buffer sizes
- Human-readable status codes
- CMake and Make builds
- Automated tests on Linux, macOS, and Windows
- A reproducible XOR example

Softmax, external model loading, quantization, and allocation-free inference
are not implemented yet.

## Quick start

### CMake

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure
```

Run the example from the build directory. With a single-configuration
generator:

```sh
./build/tnie_xor_demo
```

Visual Studio and other multi-configuration generators place the executable in
`build/Release/`.

### Make

```sh
make
./tnie_xor_demo
make test
```

## Minimal usage

```c
#include "tnie_nn.h"

float input[2] = {0.0f, 1.0f};
float output[1];

int status = tnie_nn_forward_checked(network, input, 2, output, 1);
if (status != TNIE_OK) {
    /* tnie_status_string(status) describes the failure. */
}
```

Weights use row-major order. For a layer with `input_size` inputs and
`output_size` outputs, the weight from input `i` to output `o` is stored at:

```c
weights[o * input_size + i]
```

The legacy convenience function `tnie_nn_forward()` remains available, but
cannot verify the size of caller-owned buffers. New code should use
`tnie_nn_forward_checked()`.

## Project structure

```text
include/        Public TNIE headers
src/            Inference runtime implementation
examples/xor/   Minimal example model and CLI
tests/          Runtime and regression tests
.github/        Continuous integration
```

## Intended use cases

- Classification from precomputed sensor features
- Vibration and anomaly detection
- Gesture and motion recognition
- Small control and automation models
- Educational embedded-ML projects

The current runtime allocates temporary buffers during each forward pass. That
makes this release suitable for desktop validation and embedded prototyping,
but not yet for memory-constrained production firmware.

## Roadmap

- **0.2 — Reliable Core:** validation, safer buffers, tests, CMake, CI
- **0.3 — Embedded First:** caller-owned workspace, zero heap during inference,
  immutable models in flash, benchmarks
- **0.4 — Model Format:** versioned `.tnie` format and Python exporter
- **0.5 — Real Sensor Demo:** reproducible vibration-classification pipeline
- **0.6 — TNIE Studio:** browser visualization powered by the C runtime through
  WebAssembly
- **1.0:** stable model format and documented ESP32/STM32 integration

See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions and
[CHANGELOG.md](CHANGELOG.md) for release notes.

## License

TNIE is available under the [MIT License](LICENSE).
