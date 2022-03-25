# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

## [0.6.0] - 2022-03-25
### Added
- Static type analysis
  - stubs for networkx, igraph and scipy csr_matrix
  - types for Pólya-Urn method
- Automatic benchmarking with `pytest-benchmark`
- Numba support for special functions GammaLn and BetaLn
- Python's `lru_cache` for memoization inside of Pólya-Urn
- NumbaPolya cache for memoization using JIT-classes
  - This gave a significant speedup for the Pólya-Urn method, approx 80x compared to   numba python-lru version.

### Fixed
- Pólya-Urn performance issues
- Pólya-Urn docstrings

### Changed
- Args for Pólya-Urn method
- Makefile targets:
    -   make tests -> make pytests
        -   skip all the benchmarks tests
    -   make benchmark name="{BENCHMARK_NAME}"
        -   run the benchmarks
    - github actions now uses the makefile for tests
- Return types for all graph filter methods
- Pólya-Urn code organization
- Disable mypy for while until it is fixed

## [0.5.0] - 2022-03-15
### Added
- Pólya-Urn method for integer weighted graphs now uses JIT compilation to
    speed up the computation.
- Tests for integer weighted graphs.
- Makefile targets to build and install the package in a single command.
- Fast math implementation for Beta-log and Gamma-log functions.
- Tests for for fast math implementations.

### Fixed
- Disparity filter
- Pólya-Urn methdo for integer weighted graphs
- Tests

### Changed
- Same variable name for weight, weighted degree, etc.
- Folder structure.

## [0.4.1] - 2022-03-15
### Fixed
- Docstrings
- Missing args inside of `edgeseraser.polya`

## [0.4.0] - 2022-03-14
### Added
- Support for named vertex labels
- Pólya-Urn backbone filter
- Check for disconnected graphs

### Changed
- Organized the edges and weight extraction code using DRY principles

## [0.3.0] - 2022-03-13
### Added
- Python iGraph lib support

### Fixed
- Google docstring return style
- Typos in docstrings

## [0.2.1] - 2022-03-12
### Changed
- Google docstring style

## [0.2.0] - 2022-03-12
### Fixed
- Type hiting issue for python 3.7 using `Literal`

## [0.1.0] - 2022-03-12
### Added
- Noise score filter
- Disparity filter
- Networkx integration

[Unreleased]: https://github.com/devmessias/edgeseraser/compare/0.6.0...master
[0.6.0]: https://github.com/devmessias/edgeseraser/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/devmessias/edgeseraser/compare/0.4.1...0.5.0
[0.4.1]: https://github.com/devmessias/edgeseraser/compare/0.4.0...0.4.1
[0.4.0]: https://github.com/devmessias/edgeseraser/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/devmessias/edgeseraser/compare/0.2.1...0.3.0
[0.2.1]: https://github.com/devmessias/edgeseraser/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/devmessias/edgeseraser/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/devmessias/edgeserase/releases/tag/v0.1.0
