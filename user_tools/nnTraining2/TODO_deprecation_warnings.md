# TODO: Deprecation Warnings in Feature Extraction

Date: 2026-09-04
Owner: TBD
Status: Deferred (documented for later implementation)

## Summary

`tests/test_extract_features_integration.py` currently emits a Python deprecation warning from multiprocessing internals:

- `DeprecationWarning: This process is multi-threaded, use of fork() may lead to deadlocks in the child.`

This warning is triggered during feature extraction parallelism in `user_tools/nnTraining2/extractFeatures.py`.

## Where the Warning Comes From

Current pool creation uses default multiprocessing context (fork on Linux in many environments):

- `user_tools/nnTraining2/extractFeatures.py` line ~327
- `user_tools/nnTraining2/extractFeatures.py` line ~344
- `user_tools/nnTraining2/extractFeatures.py` line ~397

These code paths call `multiprocessing.Pool(...)` and then `imap_unordered(...)`.

## Why This Matters

- Future Python versions may become stricter about unsafe forking in multi-threaded processes.
- Even if tests pass, warning noise hides real regressions.
- The warning indicates potential deadlock risk in some host/runtime combinations.

## Constraints Agreed in Current Work

- Prefer not to change runtime behavior immediately.
- Keep current production pipeline stable.
- If possible, resolve warning at test boundary first.

## Proposed Solutions (Ranked)

### Option A (Recommended now): Test-only serial/controlled pool shim

Scope:

- Test code only (`tests/test_extract_features_integration.py` and similar tests).

Approach:

1. Monkeypatch `multiprocessing.Pool` inside test to a local fake pool.
2. Implement `imap_unordered` as synchronous iteration over inputs.
3. Preserve extractor logic path while avoiding process forking.

Pros:

- No runtime code changes.
- Removes deprecation warning from integration tests.
- Low risk to training pipeline behavior.

Cons:

- Does not address runtime warning if production execution path is exercised in a multi-threaded environment.

### Option B: Targeted warning filter in tests

Scope:

- Test config only (module-level filter or `pytest.ini`/`pyproject.toml` warnings section).

Approach:

1. Filter only this exact warning message/category from multiprocessing fork path.

Pros:

- Very small change.
- Zero behavior change.

Cons:

- Warning is hidden, not resolved.
- Could mask future related safety issues.

### Option C (Runtime change, deferred): Explicit non-fork context

Scope:

- Runtime code in `user_tools/nnTraining2/extractFeatures.py`.

Approach:

1. Replace `multiprocessing.Pool(...)` with context-backed pool creation, e.g.:
   - `ctx = multiprocessing.get_context("spawn")`
   - `with ctx.Pool(processes=worker_count) as pool: ...`
2. Apply consistently to all pool call sites.
3. Benchmark throughput/memory impact and confirm deterministic output parity.

Pros:

- Resolves root cause in production path.
- Aligns with modern multiprocessing safety guidance.

Cons:

- Runtime behavior/performance can change.
- Requires broader regression/performance validation.

## Validation Plan for Future Work

When implementing any option:

1. Run:
   - `./venv/bin/python -m pytest -q tests/test_extract_features_integration.py`
2. Confirm warning count is reduced as intended.
3. For runtime changes (Option C), also run:
   - `./venv/bin/python -m pytest -q tests user_tools/nnTraining2/tests`
4. If Option C is chosen, compare representative training/extraction runtime before vs after.

## Decision Placeholder

- Selected option: TBD
- Decision date: TBD
- Approved by: TBD
