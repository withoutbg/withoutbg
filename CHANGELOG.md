# Changelog

All notable changes to this project will be documented in this file. See [Conventional Commits](https://conventionalcommits.org) for commit guidelines.

## Unreleased

### Features

* Support open-weights model **v10.0.0** (unified ONNX; 448×448 canvas)

### Bug Fixes

* Default letterbox canvas to 448 when sidecar metadata is missing

## [1.0.6](https://github.com/withoutbg/withoutbg/compare/v1.0.5...v1.0.6) (2026-07-02)

### Bug Fixes

* **ci:** use Python 3.12 as mypy target for numpy 2.x stubs ([f5cfa68](https://github.com/withoutbg/withoutbg/commit/f5cfa68725fe1b9f210f006ea6bd07c0f139dfe1))

## [1.0.5](https://github.com/withoutbg/withoutbg/compare/v1.0.4...v1.0.5) (2026-07-02)

### Bug Fixes

* **ci:** skip numpy stubs in mypy to fix CI on numpy 2.x ([1b5a42d](https://github.com/withoutbg/withoutbg/commit/1b5a42db2498e0f98ac0f0fa03b50d3f39bdbfa3))

## [1.0.4](https://github.com/withoutbg/withoutbg/compare/v1.0.3...v1.0.4) (2026-07-02)

### Bug Fixes

* **ci:** update mypy target to Python 3.10 ([08941b1](https://github.com/withoutbg/withoutbg/commit/08941b1ac40b222984c3a5349671fa4b476d56a3))

## Unreleased

### Breaking Changes (with deprecation)

The two product variants now have clear canonical names: **withoutBG Open Weights Model** (local ONNX) and **withoutBG API** (cloud).

**New primary API:**
- `WithoutBG.open_weights()` replaces `WithoutBG.opensource()`
- `OpenWeightsModel` replaces `OpenSourceModel`
- `WithoutBGAPIClient` replaces `ProAPI`
- `WithoutBGOpenWeights` replaces `WithoutBGOpenSource`
- CLI `--model open-weights` replaces `--model opensource`

**Deprecated (removed in next major release):**
- `WithoutBG.opensource()` — emits `DeprecationWarning`, delegates to `open_weights()`
- `OpenSourceModel` — alias for `OpenWeightsModel`
- `ProAPI` — alias for `WithoutBGAPIClient`
- `WithoutBGOpenSource` — alias for `WithoutBGOpenWeights`
- CLI `--model opensource` — maps to `open-weights` with a deprecation notice
