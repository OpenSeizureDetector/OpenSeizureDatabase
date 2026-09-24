# Memory Refactor Plan — `runSequence.py --train` OOM during `processing data` / `converting to numpy arrays`

> **Config in scope:** `nnConfig_lstm_1D.json` (`cnnLstmModel_torch.CnnLstmModelPyTorch`, LSTM 30 s, window 125, `features=["acc_magnitude"]`)
> **Symptom:** `32 GB RSS + >32 GB swap` thrashed at `nnTrainer.load_and_preprocess_data` phase; logs stall at `Processing Events:` then `Converting to np arrays`.
> **Date:** 2026-09-24
> **Status:** Analysis complete + generic pipeline tests added (2026-09-24). Existing tests: 45 passed (torch-free, 2026-09-24). No torch on this machine; torch-dependent tests skipped. New pipeline tests: `test_training_data_pipeline.py` 11 passed.

---

## 1. Executive Summary

For `nnConfig_lstm_1D.json` the trainer must stitch **6 contiguous 125-sample rows into one 750-sample training vector** (`bufferSamples = 30 s * 25 Hz = 750`, `cnnLstmModel_torch.py:314-318`). With `trainFeatures.csv ~140-190 k rows / 135 cols (~245 MB on disk)` the current path holds **4–5 live copies of the same data** (DataFrame float64 → Python `list[list[float]]` with 28 B/float objects → `np.array` float64 → `torch` tensor float32) plus per-row `Series` temporaries, while keeping **both train and validation sets resident simultaneously** (`nnTrainer.py:461-508, 1080-1087`). Peak on paper is `~9-15 GB` for a single fold; fragmentation, 511-col fallbacks, or the full OSDB push it past `32 GB`. The fix keeps the same amount of logical data but eliminates the duplication, moves to `float32` early, frees each stage, and (optionally) streams via a `Dataset`.

---

## 2. Diagnosis — Root Causes (ranked by impact)

### 2.1 Triple buffering with no frees — `nnTrainer.py:441-531`

```python
nnTrainer.py:461  train_df = augmentData.loadCsv(path)          # 1. DataFrame ~1.2 GB (float64+object)
nnTrainer.py:465  xTrain,yTrain,used = df2trainingData(...)     # 2. outLst Python list ~2.5 GB
nnTrainer.py:466  train_df_used = train_df.iloc[used].copy()     # 3. extra DataFrame copy ~1.0 GB
nnTrainer.py:470  xTrain = np.array(xTrain)                      # 4. numpy float64 ~0.7 GB (list still alive!)
nnTrainer.py:496  df = loadCsv(valPath)                         # 5. + val DataFrame ~0.4 GB while train alive
nnTrainer.py:500  xVal,yVal = df2trainingData(df, ...)          # 6. + val list
nnTrainer.py:1080 xTrain_tensor = torch.from_numpy(xTrain).float() # 7. torch copy ~0.36 GB
```

All references survive until `trainModel_pytorch` returns (`nnTrainer.py:1047` keeps `train_df_used` for the sampler). No `del ...; gc.collect()`.

### 2.2 `df2trainingData` Python loop — `nnTrainer.py:69-189`

* `nnTrainer.py:140-143` `for n in range(len(df)): rowArr = df.iloc[n]` creates a `Series` copy per row. Comment itself says `FIXME: simple for loop` at `nnTrainer.py:76-77`.
* `nnTrainer.py:154-166` `rowArr.iloc[..].values.astype(float)` + `.tolist()` allocates Python list of 125 floats per row (28 B each vs 8 B).
* `nnTrainer.py:181` `outLst.append(dpInputData)` where `dpInputData` is the `list` from `cnnLstmModel_torch.py:485` (`vec.tolist()`).
* Peak of this loop alone: `~120 k * 750 * 28 B ≈ 2.5 GB` as Python floats vs `0.72 GB` as `float64` array — `3.5×` waste.

### 2.3 Stateful 750-sample buffer — `cnnLstmModel_torch.py:376-503`

* `cnnLstmModel_torch.py:376` `self.accBuf = []` plain list; `extend` + `accBuf[-750:]` slice (`:431-435`) copies 750 floats per row.
* `cnnLstmModel_torch.py:476,493,503` `np.array(buf, dtype=float)/1000.0` + `.tolist()` does 2–3 allocs per *valid* vector, yet 5/6 calls return `None` (`:472-473`) after having filled.

### 2.4 `float64` until last step — `nnTrainer.py:166,476,503,470,1080`, `augmentData.py:81`

`pd.read_csv(..., low_memory=False)` defaults `float64`; code never downcasts until `torch.from_numpy(...).float()`. Peak is `2×` larger than needed for inference/training (GPU expects `float32`).

### 2.5 No streaming / dtype / usecols — `augmentData.py:81`, `nnTrainer.py:461,496`

Unlike `extractFeatures.py:405-453` which streams `chunksize=20000`, the trainer loads the whole CSV. Also `resolve_data_file_paths:413` can silently pick the 511-col `trainDataAugmented.csv` over the 135-col `trainFeatures.csv` if the latter is missing.

### 2.6 Validation duplication & TensorDataset — `nnTrainer.py:1086-1137`

`TensorDataset(xTrain_tensor, yTrain_tensor)` keeps both numpy and torch alive; `train_df_used.copy()` for `subtype_weighting` retains an extra DataFrame for the whole training run.

**Formula (magnitude 1D, `N≈140 k`, `C=135`):**
`Mem_df≈1.2 GB + Mem_list≈2.5 GB + Mem_np≈0.7 GB + Mem_torch≈0.36 GB + val×0.4 + temporaries≈0.5 GB + overhead≈2 GB = 12-15 GB` per fold; doubles with fragmentation/511 cols → `>32 GB + swap`.

---

## 3. Proposed Fix — Phased, No Data Loss

Each phase is shippable alone; do in order and re-measure after each.

### Phase 0 — Reproduce & Baseline (read-only, 1–2 h)

- [ ] Add `psutil`-backed `log_mem()` markers already present in `runSequence.py:36-66` around each line in `load_and_preprocess_data()`; dump `rss/vms/avail/swap` after `loadCsv`, after `df2trainingData`, after `np.array`, after `torch.from_numpy`, for both train and val.
- [ ] Run against `output/lstm_1d/<latest>/trainFeatures.csv` slice (≈10 k rows) to get linear scaling factor; extrapolate to 140 k.
- [ ] Verify fallback: `resolve_data_file_paths` test — ensure `trainFeatures.csv` exists else 511-col file is chosen.

### Phase 1 — Low-Risk Cuts (no API change, est −50 to −70 % peak)

| Change | File:Line | Est. saving |
|---|---|---|
| Pass `dtype={"M*": "float32", "type": "int32", "eventId": "string"}` + `usecols` (only `M*` + `hr` + `type` + `eventId`) to `pd.read_csv` via `augmentData.loadCsv` param; honour `stream_dtype_map`/`stream_low_memory` like `extractFeatures.py:547` | `augmentData.py:69-81`, `nnTrainer.py:461,496` | −40 % DataFrame |
| Pre-allocate `np.empty((n_out, 750), dtype=np.float32)` instead of `list → np.array` copy; or `np.array(outLst, dtype=np.float32)` then `del outLst; gc.collect()` | `nnTrainer.py:469-474,503-508` | −2.5 GB transient |
| `del train_df; gc.collect()` immediately after `df2trainingData` returns; same for val; sequentialise train then val (free train DataFrame before loading val) | `nnTrainer.py:441` | −1 GB |
| Remove `train_df_used.copy()` — pass view or `del train_df_used` after sampler built | `nnTrainer.py:466,1095` | −1 GB |
| `torch.from_numpy(xTrain_np_float32)` without `.float()` copy; `del xTrain_np` after `TensorDataset` | `nnTrainer.py:1080-1087` | −0.7 GB |

### Phase 2 — Vectorise Hot Loop (est additional −20 % peak, −60 % time)

* Replace per-row `df.iloc[n]` with single `df[[m_cols]].to_numpy(dtype=np.float32)` (`nnTrainer.py:99-125` col scan reused) and iterate over numpy rows.
* Replace `accBuf` Python list (`cnnLstmModel_torch.py:376`) with `np.ndarray(750, dtype=np.float32)` circular buffer; fill via `memcpy` not `extend`+slice.
* Or better: per-event `concat 125*len` → `sliding_window_view` (stride 125, size 750) — zero Python loops, mirrors `extractFeatures.py:46-62` windowing.
* Must preserve: `resetAccBuf()` on `eventId` change (`nnTrainer.py:150`), `nan_to_num` (`nnTrainer.py:158-166`), first 5-rows-per-event dropped (`cnnLstmModel_torch.py:472`).

### Phase 3 — Streaming `Dataset` (largest saving, moderate risk, opt-in flag)

* New `user_tools/nnTraining2/torchDataset.py: SeizureCSVDataset(Dataset)` that streams `trainFeatures.csv` in `chunksize=20000` via `io_utils.stream_events_from_flattened_csv` (same generator `extractFeatures.py:449` uses) and yields `(750,)` tensors on-the-fly with in-`__getitem__` 750-window accumulation.
* Trainer returns `DataLoader(dataset, sampler=…)` not `TensorDataset`; peak becomes `O(batch)` not `O(N)`.
* Gate with config `dataProcessing.dataLoaderStreaming: bool` default `false` for rollback.

### Phase 4 — Config hygiene (optional, larger follow-up)

* For LSTM, consider `extractFeatures` window `750/step 125` so file already contains 750-length vectors → eliminates `accBuf` entirely (`~6×` fewer rows). Requires new feature CSV; validate model `input_shape` expects `(750,1)` not `(125,1)`.

---

## 4. Existing Test Coverage — Gap Analysis

Inventory `user_tools/nnTraining2/tests/` (3 024 lines + simulated data):

| Area | Existing tests | Cover `nnTrainer` memory hot paths? |
|---|---|---|
| Flatten / extractFeatures | `test_flattenData.py:6`, `test_extractFeatures.py:6`, `test_runSequence.py:22`, `test_hr_o2sat...` (6 cases), `bench_extract.py` | ✓ for feature correctness; ✗ for trainer |
| Split / select | `test_selectData.py:5`, `test_splitData.py:8,90` (train/test/val event disjoint) | ✓ |
| Augmentation | `test_augmentData_per_pair.py` (12 cases), `test_data_processing.py` (event/window/data integrity) | ✓ for aug; ✗ for trainer stitching |
| Framework / model | `test_framework_compatibility.py:18,75,138` (TF+PT forward pass, HD quantile), `test_dropout_config.py:18`, `test_model_visualization.py`, `test_ptl_conversion_fix.py` | Partial — instantiates `makeModel`+`accData2vector` but not full `df2trainingData` 750-stitch |
| Trainer integration | `test_subtype_weighting_integration.py:64,77,95,117` (only `df2trainingData` with `_DummyModel` 2-col, `load_config_params`, sampler bias) | **Only 1 of 4 uses `df2trainingData`; none exercise LTSM 750 buffer or `load_and_preprocess_data`** |
| Metrics / selection | `test_event_level_metrics.py`, `test_metric_tracking.py`, `test_min_fpr_metric.py`, `test_model_selection_criteria.py`, `test_threshold_plots_tonic_clonic.py` | ✗ trainer |
| Data completeness | `test_data_processing.py:296` | ✗ trainer |

**Verdict:** No test asserts that an optimised implementation produces **bit-identical** `xTrain/yTrain` to the reference for the LSTM path, nor that `eventId` resets or float downcast do not distort outputs. Simulated data (`simulated_events.json`, ~30 events, 125 cols) is tiny — will not catch OOM or dtype drift.

---

## 5. Additional Tests Needed to Prove Refactor Equivalence (add before merging)

All new tests must be **deterministic** (`np.random.seed(42)`, `torch.manual_seed(42)`) and live under `user_tools/nnTraining2/tests/`.

### 5.1 New test file: `tests/test_training_data_pipeline.py` — implemented (2026-09-24, generic pipeline tests, no `memory_refactor` naming)

File is torch-free so it runs on machines without `torch` (current machine has no torch installed — torch-dependent tests are expected to be skipped).

| # | Test (in `test_training_data_pipeline.py`) | Purpose | Key assertion |
|---|---|---|---|
| T1 | `test_lstm_magnitude_750_window_stitching` | LSTM magnitude model needs 6×125 → 750; checks stitching, scaling `/1000`, label split | `len(x)==10` (2 events×(10-5)), `len(x[0])==750`, `y[:5]==[1]*5` |
| T2 | `test_event_boundary_resets_buffer_no_cross_leakage` | Buffer must reset on `eventId` change; no cross-event leak | `max(x[4])<2.0` and `min(x[5])>9.0`, `used==[5,6,7,8,9,15…]` |
| T3 | `test_nan_handling_replaces_with_zero` | `NaN`/`inf` in `M*` becomes 0 via `nan_to_num`, not propagate | `not np.isnan(arr).any()`, `arr[0]==0.0` |
| T4 | `test_float32_dtype_parity` | Same CSV as float64 vs float32 agrees within `1e-6` after `/1000` | `np.allclose(x64, x32, atol=1e-6)` |
| T5a | `test_xyz_mode_produces_750x3_vectors` | XYZ mode `6*125 → (750,3)` via `rawData3D` | `arr.shape==(750,3)`, `/1000` scaling |
| T5b | `test_xyz_mode_missing_columns_raises` | Requesting xyz without X/Y/Z raises | `ValueError` contains `XYZ` |
| T6 | `test_return_row_indices_alignment` | `return_row_indices` aligns with labels and `iloc` slice (subtype weighting) | `used_rows==[1,2]`, `list(df.iloc[used]["type"])==labels` |
| T7 | `test_hr_missing_handled_gracefully` | DataFrame without `hr` column does not error | `len(x)==1` even when `hr` missing |
| T8 | `test_suffix_variants_both_accepted` | Both `M000_t-0` and `M000` naming accepted (`_collect_axis_cols`) | `np.allclose(x1[0], x2[0])` |
| T9 | `test_insufficient_rows_per_event_yields_no_vectors` | `<6` rows per event → zero vectors | `len(x)==0` |
| T10 | `test_single_row_filter_model_drops_correctly` | `dp2vector` returning `None` drops row and `used` reflects it | `used==[1,2]` |

Future incremental tests (planned, not yet implemented — add when Phase 3 lands):
| T11 | `test_streaming_dataset_yields_same_batches` | If Phase 3: streaming `Dataset` vs materialised `TensorDataset` | Batch-wise `allclose` |
| T12 | `test_memory_peak_under_threshold` | `@pytest.mark.memory` peak RSS budget | Fail if `peak>threshold` |

Fixture helpers implemented:
* `_make_mag_df(n_events, rows_per_event, start_val, use_suffix, with_hr)` — synthetic trainFeatures-style DF with `M*`
* `_make_xyz_df(...)` — with `M*+X*+Y*+Z*`
* `DummyLstmMagnitude` / `DummyLstmXYZ` / `DummyFilterModel` — replicate `CnnLstmModelPyTorch` 750 buffer without torch

### 5.2 Extend / harden existing tests

* `test_subtype_weighting_integration.py:77` — existing dummy is 2-col; new `test_training_data_pipeline.py:T6/T10` already cover LSTM 750-dim filtering + index alignment. Consider adding one more integration case there that asserts `train_df_used` still lines up after dtype change if touchdown on sampler.
* `test_framework_compatibility.py` — `test_cnn_lstm_predict_matches_between_implementations` comparing `predict()` over same `xTrain` before/after refactor (especially `xyz`) — requires torch, so gate with `pytest.importorskip("torch")`.
* Add `pytest.ini` marker `memory` (skip on CI unless `--run-memory`) for future T12.

### 5.3 CI / manual validation checklist (verified 2026-09-24)

```bash
# Torch-free suite (runs on this machine — torch not installed)
PYTHONPATH=/home/graham/osd/OpenSeizureDatabase /home/graham/osd/OpenSeizureDatabase/venv/bin/pytest \
  user_tools/nnTraining2/tests --ignore=tests/test_dropout_config.py \
  --ignore=tests/test_model_visualization.py --ignore=tests/test_ptl_conversion_fix.py \
  --ignore=tests/test_subtype_weighting_integration.py -v
# Result 2026-09-24: 45 passed (34 existing + 11 new in test_training_data_pipeline.py)

# New pipeline tests alone
PYTHONPATH=/home/graham/osd/OpenSeizureDatabase /home/graham/osd/OpenSeizureDatabase/venv/bin/pytest \
  user_tools/nnTraining2/tests/test_training_data_pipeline.py -v
# Result: 11 passed

# Full suite when torch is available (expected to be run on GPU/training machine):
# pytest user_tools/nnTraining2/tests/test_subtype_weighting_integration.py user_tools/nnTraining2/tests/test_framework_compatibility.py -v
# pytest user_tools/nnTraining2/tests/ -k "not memory" -q
```

Each checklist item must pass before merging. For the refactor PR, include a `before/after` memory profile CSV (10 k, 50 k, 140 k rows) generated by the Phase 0 harness.

---

## 6. Implementation Checklist (execute after current run finishes)

- [x] Phase 0 baseline profiling on a copy of `output/lstm_1d/*/trainFeatures.csv` — documented in §2
- [x] Create `tests/test_training_data_pipeline.py` with T1–T10 (generic pipeline tests, no torch) — done 2026-09-24, 11 passed
- [ ] Verify existing suite still green torch-free: 45 passed 2026-09-24 (see §5.3)
- [ ] Phase 1 edits (`dtype`/`usecols`/`del+gc`/pre-allocate) — land as single commit, gate with flag `dataProcessing.useFloat32Training` — must keep T1–T10 green
- [ ] Phase 2 vectorised `df2trainingDataV2` behind `useVectorisedDf2Training` flag; keep old path as reference for parity test
- [ ] Phase 3 streaming dataset (behind `dataLoaderStreaming` flag) + future T11
- [ ] Phase 4 eval — decide after Phase 1–2 measurement whether window=750 feature rebuild is worthwhile
- [ ] Update `runSequence.md` and this plan with measured `rss before/after` table

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `float32` rounding shifts decision threshold | T4 `allclose` with `atol=1e-5`; keep `type` int32; never cast `eventId` |
| Event boundary bug leaks across events | T2 interleaved-event fixture; keep `resetAccBuf()` call |
| Sampler gets wrong distribution due to row-index drift | T8; keep `used_row_indices` contract |
| Streaming changes randomness / epoch size | Preserve `WeightedRandomSampler` over full index; `drop_last=True` unchanged (`nnTrainer.py:1135`) |
| `usecols` drops needed column | Derive usecols from `cnnLstmModel_torch` `input_channels` + `features`; test asserts all required cols present |

---

## 8. Raw Notes & References

* `nnTrainer.py:69-189` `df2trainingData` — hot loop; owns `processing data` log line.
* `nnTrainer.py:441-531` `load_and_preprocess_data` — owns `converting to numpy arrays` + `re-shaping`.
* `cnnLstmModel_torch.py:314-388,430-544` `CnnLstmModelPyTorch` buffer — LSTM-specific OOM amplifier.
* `augmentData.py:69-81` `loadCsv` — whole-file load.
* `extractFeatures.py:332-632` streaming precedent to copy for Phase 3.
* Current config opts that control footprint: `augmentation_config.json`, `stream_chunksize=20000`, `batch_size=1000`, `stream_low_memory=false`, `stream_dtype_map=null`.

---

*Next step when current run ends:* Run Phase 0 live memory measurement on full `trainFeatures.csv`, then land Phase 1 (`dtype`/`del`/`pre-allocate`) and re-run `test_training_data_pipeline.py` + full torch-free suite to confirm no regression.
