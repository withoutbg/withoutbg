# Changelog

All notable changes to this project will be documented in this file. See [Conventional Commits](https://conventionalcommits.org) for commit guidelines.

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
