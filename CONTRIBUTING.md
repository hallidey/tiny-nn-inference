# Contributing to TNIE

Thank you for helping improve TNIE. The project values small, reviewable
changes, portable C11, predictable behavior, and tests for every bug fix.

## Development setup

Configure, build, and test with CMake:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
ctest --test-dir build --build-config Debug --output-on-failure
```

Alternatively, on a Unix-like system with Make:

```sh
make test
```

## Pull requests

- Keep the public API documented in the headers.
- Add or update tests when behavior changes.
- Do not introduce third-party runtime dependencies without prior discussion.
- Keep model-specific code under `examples/`, not in the runtime library.
- Update `README.md` and `CHANGELOG.md` when user-visible behavior changes.

Bug reports should include the compiler, operating system, reproduction steps,
and the smallest model that demonstrates the problem.
