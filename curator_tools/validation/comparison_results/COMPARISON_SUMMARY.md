================================================================================
makeOsdDb Comparison Report - Fixed Refactored Version
================================================================================
Generated: 2026-07-05 21:02:46


================================================================================
All Seizures: osdb_3min_allSeizures.json
================================================================================

Event Counts:
  Baseline (V1.10):        257
  Original (after update): 406
  Refactored (after update): 331

Original Version vs Baseline:
  Added: 149 events
  Removed: 0 events
  Modified: 0 events

Refactored Version vs Baseline:
  Added: 110 events
  Removed: 0 events
  Merged (not primary): 36 events
  Modified: 221 events

✓ All baseline events preserved in refactored version

ℹ️  36 baseline events merged into other events:
  IDs: [119, 5486, 6590, 6668, 6767, 6840, 7006, 7007, 7044, 7126, 7222, 8960, 9005, 12214, 21569, 26077, 26992, 31421, 34759, 36799]

  → Detailed comparison saved to: /home/graham/osd/OpenSeizureDatabase/curator_tools/makeOsdDb_refactor/comparison_results/detail_osdb_3min_allSeizures.txt

================================================================================
Tonic-Clonic Seizures: osdb_3min_tcSeizures.json
================================================================================

⚠️  Baseline file not found: /home/graham/osd/osdb/V1.10/osdb_3min_tcSeizures.json

================================================================================
Fall Events: osdb_3min_fallEvents.json
================================================================================

Event Counts:
  Baseline (V1.10):        48
  Original (after update): 36
  Refactored (after update): 36

Original Version vs Baseline:
  Added: 0 events
  Removed: 12 events
  Modified: 0 events

Refactored Version vs Baseline:
  Added: 0 events
  Removed: 12 events
  Merged (not primary): 0 events
  Modified: 0 events

⚠️  WARNING: 12 baseline events LOST in refactored version!
  Lost IDs: [14898, 46156, 46157, 48580, 73425, 73557, 73614, 73634, 73738, 74252, 149937, 712217]

  → Detailed comparison saved to: /home/graham/osd/OpenSeizureDatabase/curator_tools/makeOsdDb_refactor/comparison_results/detail_osdb_3min_fallEvents.txt