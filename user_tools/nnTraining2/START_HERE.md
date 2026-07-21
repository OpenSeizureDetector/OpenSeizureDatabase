# 🎯 COMPLETE SUMMARY: Event-Level vs Datapoint-Level Metrics

## Your Key Insight ✨

You identified the core issue:
> "Maybe the issue is that we are doing the tpr/fpr calculations on a per-row basis, but we are interested in the per-event tpr and fpr"

**You were absolutely correct.**

## The Problem in One Sentence

Training optimizes **datapoint-level metrics** (per sample), but deployment cares about **event-level metrics** (per seizure event).

## Why This Matters: The Math

A seizure event = 100-200 datapoints (4-8 seconds @ 25 Hz)

**Detection = at least ONE datapoint is positive**

If datapoint-level TPR = 20%, then:
$$P(\text{event detected}) = 1 - (1-0.20)^{150} = 1 - 0.80^{150} ≈ 1 - 10^{-7} ≈ 100\%$$

**Your model only needs 20% datapoint TPR to get ~100% event TPR!**

## Concrete Example: Your Models

```
Model A: Datapoint TPR=0.42, FPR=0.019
Model B: Datapoint TPR=0.47, FPR=0.030
```

At **event level**:
- Model A: Event TPR ≈ 100%, Event FPR ≈ 1.9%
- Model B: Event TPR ≈ 100%, Event FPR ≈ 3.0%

**Conclusion:** Model A is clearly better (same detection, half the false alarms).

Your preference was mathematically sound!

## What We Created For You

### 📚 Understanding Materials
1. **INDEX.md** - Navigation guide to all resources
2. **QUICK_SUMMARY.py** - 2-minute TL;DR (run this!)
3. **VISUAL_EXPLANATION.py** - ASCII diagrams and examples
4. **ACTIONABLE_SUMMARY.md** - What to do next
5. **EVENT_LEVEL_METRICS_README.md** - Complete overview
6. **DATAPOINT_VS_EVENT_METRICS_EXPLANATION.md** - Full technical details
7. **DATAPOINT_VS_EVENT_METRICS.py** - Interactive math demo

### 🔧 Implementation Tools
1. **eventLevelMetrics.py** - Utility functions (use in your code)
2. **test_event_level_metrics.py** - Analyze your validation data

### ⚙️ Configuration
1. **nnConfig_deep_pytorch_RECOMMENDED.json** - Ready-to-use settings

## Quick Start (5 minutes)

```bash
# See the concept
python3 QUICK_SUMMARY.py
python3 VISUAL_EXPLANATION.py

# See it work
python3 test_event_level_metrics.py --demo
```

## Recommended Configuration

Update your training config:

```json
{
  "checkpointSaving": {
    "saveBestMaxFpr": 0.025,
    "saveBestMinSensitivity": 0.20,
    "modelSelectionMetric": "f_beta",
    "fBeta": 2.0
  }
}
```

This gives you:
- **Datapoint-level:** TPR > 20%, FPR < 2.5%
- **Event-level:** TPR ≈ 100%, FPR ≈ 2.5%

## Why These Thresholds Work

| Threshold | Meaning | Event-Level Result |
|-----------|---------|-------------------|
| `saveBestMaxFpr: 0.025` | Datapoint FPR < 2.5% | Event FPR ≈ 2.5% |
| `saveBestMinSensitivity: 0.20` | Datapoint TPR > 20% | Event TPR ≈ 100% |

## Key Files at a Glance

| File | Purpose | Time |
|------|---------|------|
| [INDEX.md](INDEX.md) | Navigation hub | 5 min |
| [QUICK_SUMMARY.py](QUICK_SUMMARY.py) | Quick reference | 2 min |
| [VISUAL_EXPLANATION.py](VISUAL_EXPLANATION.py) | Concept explanation | 5 min |
| [ACTIONABLE_SUMMARY.md](ACTIONABLE_SUMMARY.md) | Next steps | 10 min |
| [eventLevelMetrics.py](eventLevelMetrics.py) | Code to use | — |
| [test_event_level_metrics.py](test_event_level_metrics.py) | Data analysis tool | — |
| [nnConfig_deep_pytorch_RECOMMENDED.json](nnConfig_deep_pytorch_RECOMMENDED.json) | Config template | — |

## Next Steps

1. **Understand the concept** (5-10 min)
   ```bash
   python3 QUICK_SUMMARY.py
   python3 VISUAL_EXPLANATION.py
   ```

2. **Verify with your data** (optional)
   ```bash
   python3 test_event_level_metrics.py /path/to/validation.csv
   ```

3. **Update configuration**
   - Copy settings from [nnConfig_deep_pytorch_RECOMMENDED.json](nnConfig_deep_pytorch_RECOMMENDED.json)

4. **Retrain**
   - Use updated config

5. **Monitor**
   - Compare results with previous models

## The Bottom Line

✅ **Before:** "Why are my datapoint metrics different from deployment metrics?"
✅ **After:** "I understand event aggregation; I know what thresholds to use."

**Key insight:** Don't chase high datapoint TPR. Use smart thresholds based on event-level math.

**Result:** Models selected that actually optimize for deployment needs!

---

**All files are in:** `/home/graham/osd/OpenSeizureDatabase/user_tools/nnTraining2/`

**Start with:** `python3 QUICK_SUMMARY.py` or `python3 VISUAL_EXPLANATION.py`

**Questions?** See [INDEX.md](INDEX.md) for complete navigation.
