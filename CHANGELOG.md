# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.4.3] - 2026-11-18

- chore: rename parameter of damping: `alpha` -> `alpha_damping`.

## [0.4.2] - 2026-11-17

- feature: add transformation for damping oscillation.

## [0.4.1] - 2026-11-11

- lint: rename parameter `density` to `matter_density` of neutrino oscillations in the matter.
- fix: add `NUMBA_CACHE_ENABLE` parameter to `njit` function of neutrino oscillations in the matter.

## [0.4.0] - 2026-09-21

- feature: add survival probability of electron antineutrino in matter.
- chore: necessary updates for compatibility with cython-ready version of dag-modelling.

## [0.3.0] - 2026-02-24

- chore: raise minimal Python version to 3.12.
- feature: dynamic Versioning.

## [0.2.2] - 2025-10-17

- chore: disable `numba` caching as it may cause problems for parallel execution. Configurable.
- chore: step `dag-modelling` dependence version to 0.14.3.

## [0.2.1] - 2025-09-25

- [feat] Add option `leading_mass_splitting_3l_name` to class `NueSurvivalProbability` that switches between `$|\Delta m^2_{32}|$` and `$|\Delta m^2_{31}|$` inputs.

## [0.2] - 2025-07-29

- First PYPI version.

## [0.1] - 2025-07-24

The pre-release version.

### Added

### Changed

### Fixed

