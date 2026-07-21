#!/usr/bin/env python3
"""
Simple summary: Why Event-Level TPR > Datapoint-Level TPR
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║           DATAPOINT vs EVENT-LEVEL METRICS - KEY INSIGHT                   ║
╚════════════════════════════════════════════════════════════════════════════╝

YOUR OBSERVATION:
  "Datapoint TPR of 0.3+ will give very high event-based TPR"

WHY THIS HAPPENS:
  
  A seizure event = 100-200 datapoints (4-8 seconds)
  Event detection = "At least one datapoint in event is detected"
  
  This is like asking: "In a room with 150 people, if I randomly pick 30% 
  of them and give them a buzzer, what's the chance at least one will buzz?"
  
  Answer: ≈100%

MATHEMATICAL PROOF:

  P(at least 1 detected) = 1 - P(none detected)
                         = 1 - (1 - datapoint_tpr)^event_length
                         = 1 - (0.7)^150              (if TPR=0.3)
                         ≈ 1 - 10^-14
                         ≈ 100%

TABLE: Datapoint TPR → Event-Level TPR

  Datapoint TPR    Event TPR (N=150)    Events Missed (per 1000)
  ─────────────    ────────────────    ──────────────────────────
      5%                99.99985%                0.00015
     10%                99.99999%                0.00001
     15%                ~100%                   ~0
     20%                ~100%                   ~0
     30%                ~100%                   ~0

IMPLICATIONS FOR YOUR TRAINING:

  1. You should optimize for datapoint metrics during training
  2. But understand that datapoint TPR of 20-30% = event TPR of ~100%
  3. Set thresholds accordingly:
     - saveBestMaxFpr: 0.025 (datapoint level, 2.5% false alarm rate)
     - saveBestMinSensitivity: 0.20 (datapoint level, 20% TPR minimum)
  
  4. These translate to:
     - Event TPR: ~100%
     - Event FPR: ~2.5% (similar to datapoint, since false alarms are independent)

WHAT THIS MEANS FOR YOUR MODEL SELECTION:

  Why you prefer Model A over Model B even though B has higher TPR:
  
  Model A: Datapoint TPR=0.42, FPR=0.019
  Model B: Datapoint TPR=0.47, FPR=0.03
  
  At EVENT level:
  Model A: Event TPR≈100%, Event FPR≈1.9%
  Model B: Event TPR≈100%, Event FPR≈3.0%
  
  Both have essentially perfect seizure detection (both ~100% event TPR).
  Model A is better because it has lower false alarm rate.

FILES CREATED:
  - eventLevelMetrics.py              : Utility functions
  - test_event_level_metrics.py       : Script to analyze your data
  - DATAPOINT_VS_EVENT_METRICS_EXPLANATION.md : Full explanation

NEXT STEPS:
  1. python3 test_event_level_metrics.py  (analyzes validation CSV)
  2. Update nnTrainer.py to log event-level metrics
  3. Use recommended config thresholds

╚════════════════════════════════════════════════════════════════════════════╝
""")
