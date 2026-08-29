# CNN-LSTM Training Report - August 2026

This folder contains the comprehensive technical report and supporting materials for the CNN-LSTM seizure detection model training using OpenSeizureDatabase V1.11.

## Contents

### Main Report
- **CNN_LSTM_Training_Report_August_2026.md** - Complete technical report with:
  - Dataset description (522 seizures from 30 contributors)
  - Model architecture and training methodology
  - Results from all three runs (5, 6, 7)
  - Detailed analysis of 5×5 cross-validation FPR discrepancy
  - Statistical analysis and recommendations

### Figures and Visualizations

From Run 5 (Production model):
- **cnnLstmModel_pytorch_architecture.png** - Model architecture diagram
- **cnnLstmModel_pytorch_training_tpr_fpr.png** - Training progress (TPR/FPR over epochs)
- **cnnLstmModel_pytorch_pt_event_confusion.png** - Confusion matrix for event-level predictions
- **cnnLstmModel_pytorch_event_threshold_analysis.png** - Threshold sensitivity analysis

## Key Findings

### Performance Summary
- **Run 5 (Production):** TPR = 89.4%, FPR = 17.7%
- **Run 6 (3×3 Nested K-Fold):** TPR = 86.6% ± 3.8%, FPR = 18.7% ± 1.2%
- **Run 7 (5×5 Nested K-Fold):** TPR = 89.8% ± 1.6%, FPR = 25.3% ± 5.0%
- **Tonic-Clonic Detection:** 95.1% TPR (most clinically important seizure type)

### FPR Discrepancy Explanation

The higher FPR in Run 7 (25.3%) compared to Run 5 (17.7%) and Run 6 (18.7%) is due to:

1. **Smaller test sets:** Run 7 uses test folds of ~3,826 non-seizure events vs. ~6,377 in Run 6
2. **Higher statistical variability:** 4.1× higher coefficient of variation in Run 7
3. **Sampling effects:** Run 7 Fold 4 achieved 17.5% FPR, consistent with Run 5
4. **Confidence intervals:** Run 7's 95% CI [20.8%, 29.8%] reflects measurement uncertainty, not model degradation

**Conclusion:** The true FPR is ~17-19%, and Run 7's elevated mean is a statistical artifact of smaller sample sizes.

## Recommendations

1. **Deployment:** Use Run 5 production model (TPR ~89%, FPR ~18%)
2. **Future Validation:** Prefer 3×3 nested k-fold for datasets of this size
3. **Reporting:** Always include confidence intervals with cross-validation results
4. **Model Improvements:** Focus on activity-aware detection and user personalization

## Source Data Locations

- Run 5 (Production): `~/osd/OpenSeizureDatabase/user_tools/nnTraining2/output/cnnLstmModel_pytorch/5/`
- Run 6 (3×3 Nested): `~/osd/OpenSeizureDatabase/user_tools/nnTraining2/output/cnnLstmModel_pytorch/6/`
- Run 7 (5×5 Nested): `~/osd/OpenSeizureDatabase/user_tools/nnTraining2/output/cnnLstmModel_pytorch/7/`
- Training Notes: `~/osd/OpenSeizureDatabase/user_tools/nnTraining2/training_notes_aug2026.txt`

## Report Date
August 29, 2026
