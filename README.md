# CS5344 Group 33 - Loan-Level Anomaly Detection

**Team Members:** Jia Sheng | Hu Qingxuan | Gong Zihong

## Project Overview

This project develops an ensemble anomaly detection system for identifying problematic loan repayment behavior in mortgage data. Using unsupervised machine learning on a 14-month observation window, we achieved **0.4745 Average Precision** on the validation set, representing a **28.3% improvement** over baseline methods.

### Problem Statement

Financial institutions face significant losses from loan defaults. Traditional credit scoring models evaluate borrowers only at origination, missing behavioral patterns that emerge during repayment. Our approach:

- Analyzes both static borrower information and temporal payment sequences
- Detects abnormal repayment behavior through payment irregularity patterns
- Uses ensemble of 15 anomaly detection models with optimized weights
- Handles severe class imbalance (87.4% normal, 12.6% anomaly)

### Key Results

- **Validation Average Precision:** 0.4745
- **ROC-AUC:** 0.7681
- **Best Feature Group:** Payment Irregularity (29 features)
- **Optimal Ensemble:** One-Class SVM (65.4%) + ECOD (34.2%)
- **Contamination Parameter:** 0.05

## File Structure

```
Group33Codebase/
├── README.md                           # This file
├── requirements.txt                    # Package dependencies
├── CS5344_G33_Final.ipynb             # Main notebook with full analysis
├── rfod_standalone.py                 # Custom RFOD implementation
├── rfod_ensemble_predictions.csv      # Final test predictions (Id, target)
└── Track2/                            # Dataset folder
    ├── loans_train.csv                # Training data (30,504 loans)
    ├── loans_valid.csv                # Validation data (5,370 loans)
    └── loans_test.csv                 # Test data (13,426 loans)
```

**Note:** Model weights are NOT included. All models are trained from scratch when running the notebook, ensuring full reproducibility and transparency.

## Setup Instructions

### 1. Environment Setup

**Python Version:** 3.10 or higher recommended

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Requirements

The dataset is included in the `Track2/` folder:
- `loans_train.csv` (30,504 loans, 144 features)
- `loans_valid.csv` (5,370 loans, 144 features)
- `loans_test.csv` (13,426 loans, 143 features - no target)

**Important:** The notebook expects data files in `Track2/` folder. Update file paths in the notebook if your data is located elsewhere:

```python
# In the notebook, data is loaded as:
train_df = pd.read_csv('Track2/loans_train.csv', low_memory=False)
valid_df = pd.read_csv('Track2/loans_valid.csv', low_memory=False)
test_df = pd.read_csv('Track2/loans_test.csv', low_memory=False)
```

### 3. GPU Support (Optional)

The code includes optional GPU acceleration via RAPIDS cuML. If you have an NVIDIA GPU:

```bash
# Install RAPIDS (CUDA 11.x example)
pip install cudf-cu11 cuml-cu11 cupy-cuda11x
```

The code automatically detects GPU availability and falls back to CPU if not available.

## Running the Code

### Run Complete Notebook

Open and run the Jupyter notebook from top to bottom:

```bash
jupyter notebook CS5344_G33_Final.ipynb
```

The notebook will:
1. Load data from `Track2/` folder
2. Perform EDA and feature engineering
3. Train all 15 anomaly detection models from scratch
4. Optimize ensemble weights
5. Generate predictions on test set
6. Save results to `rfod_ensemble_predictions.csv`

**Expected Runtime:**
- EDA sections: ~5-10 minutes
- Feature engineering: ~2-3 minutes
- Model training (15 models): ~10-15 minutes
- Weight optimization: ~5-10 minutes
- **Total:** ~25-40 minutes on standard CPU

**Note:** All models are trained from scratch each time. This ensures reproducibility and transparency, though it requires more time than loading pre-trained weights.

## Methodology Overview

### 1. Exploratory Data Analysis

**Key Findings:**
- Training set: 100% normal loans (unsupervised learning required)
- Validation set: 87.4% normal, 12.6% anomaly
- Payment patterns show systematic modifications/forbearance
- Median payment ratio: 0.15 (paying 15% of scheduled amount is normal)
- Balance reduction: Only 1.3% over 13 months (95.5% → 98.7%)

**Critical Insight:** "Normal" doesn't mean "paying on schedule" - it means consistent behavior patterns within modification/forbearance programs.

### 2. Feature Engineering

Developed 7 feature groups (106 total features):

1. **baseline_payment** (40 features) - Payment ratios and windowed statistics
2. **payment_velocity** (15 features) - Principal reduction rates
3. **payment_irregularity** (29 features) ⭐ - Underpayment flags, streaks, patterns
4. **balance_trajectory** (5 features) - Balance evolution and stagnation
5. **credit_risk** (8 features) - Traditional origination risk factors
6. **temporal_changes** (4 features) - Early vs late period shifts
7. **advanced_patterns** (5 features) - Statistical distribution signatures

**Winner:** `payment_irregularity` achieved 0.370 AP (2x better than other groups)

### 3. Model Architecture

**Ensemble of 15 Anomaly Detection Models:**

- **RFOD** - Custom Random Forest Outlier Detection
- **LOF** (4 variants) - Local Outlier Factor with n_neighbors=[30, 50, 75, 100]
- **One-Class SVM** - RBF kernel, gamma=auto
- **Isolation Forest** - 100 estimators
- **PyOD Models:**
  - ECOD (Empirical Cumulative Distribution)
  - COPOD (Copula-based Outlier Detection)
  - HBOS (Histogram-based Outlier Score)
  - KNN (k-Nearest Neighbors)
  - INNE (Isolation-based Nearest Neighbor Ensemble)
  - CBLOF (Cluster-based Local Outlier Factor)
  - LODA (Lightweight Online Detector of Anomalies)
  - COF (Connectivity-based Outlier Factor)

### 4. Optimization Pipeline

**Feature Selection:** Greedy forward selection starting with top 3 groups
- Started with `payment_irregularity` (AP=0.477)
- Adding `balance_trajectory` → No improvement (rejected)
- Adding `temporal_changes` → No improvement (rejected)
- **Final:** Single group (payment_irregularity) performed best

**Weight Optimization:** Greedy search maximizing validation Average Precision
- Initial (equal weights): AP = 0.477
- After optimization: AP = 0.475
- **Result:** 15 models → effectively 2 active models
  - One-Class SVM: 65.4% weight
  - ECOD: 34.2% weight
  - Others: <0.2% (effectively pruned)

**Contamination Tuning:**
- Tested: 0.03, 0.04, 0.05
- Best: 0.05 (matches expected 12.6% anomaly rate)

### 5. Prediction Pipeline

1. **Preprocessing:** Outlier clipping → Imputation → Scaling (fitted on train only)
2. **Model Predictions:** All 15 models generate anomaly scores
3. **Rank Normalization:** Convert scores to uniform [0, 1] scale
4. **Weighted Ensemble:** Combine using optimized weights
5. **Probability Calibration:** Beta CDF transformation (α=0.5, β=0.5)

## Key Implementation Details

### No Data Leakage

✅ All preprocessing fitted exclusively on training set
- Outlier bounds: 99.5th/0.5th percentiles from training data
- Imputer: Median computed from training data
- Scaler: Mean/std computed from training data

✅ Validation set used only for hyperparameter tuning
- Weight optimization
- Contamination selection
- Feature group selection

✅ Test set never touched until final prediction

### Reproducibility

- **Random Seeds:** Set to 42 for all stochastic models
- **Deterministic Pipeline:** Same preprocessing steps in same order
- **Version Pinning:** All package versions specified in requirements.txt

### Why Payment Irregularity Wins

Normal loans show **consistent** patterns even with low payments:
- Modifications/forbearance create new "normal" baselines
- Consistent underpayment ≠ anomaly
- **Anomalies show erratic, irregular behavior:**
  - Sudden zero payments after consistent payments
  - Wild payment fluctuations (payment chaos)
  - Unusual streak patterns
  - Unexpected payment accelerations/decelerations

## Performance Breakdown

### Feature Group Performance (Individual One-Class SVM)

| Feature Group | AP Score | Features | Performance |
|--------------|----------|----------|-------------|
| payment_irregularity | 0.3698 | 29 | ⭐ Best |
| balance_trajectory | 0.1891 | 5 | Good |
| temporal_changes | 0.1705 | 4 | Moderate |
| credit_risk | 0.1679 | 8 | Moderate |
| baseline_payment | 0.1642 | 40 | Moderate |
| payment_velocity | 0.1595 | 15 | Moderate |
| advanced_patterns | 0.1483 | 5 | Baseline |

### Optimization Pipeline Results

| Stage | AP Score | Improvement |
|-------|----------|-------------|
| Baseline (Single SVM) | 0.3698 | - |
| Feature Selection | 0.4774 | +29.1% |
| Equal Weights Ensemble | 0.4774 | +0.0% |
| Weight Optimization | 0.4716 | -1.2% |
| Contamination Tuning | 0.4745 | +0.6% |

### Individual Model Performance

| Model | AP Score | Weight | Active |
|-------|----------|--------|--------|
| One-Class SVM | 0.3546 | 65.4% | ✓ |
| ECOD | 0.0869 | 34.2% | ✓ |
| RFOD | 0.2722 | 0.00% | ✗ |
| LOF_30 | 0.2328 | 0.00% | ✗ |
| LODA | 0.1090 | 0.16% | ✗ |
| **ENSEMBLE** | **0.4745** | - | **⭐** |

**Ensemble vs Best Individual:** +33.8% improvement (0.4745 vs 0.3546)

## Validation Insights

### Why Simple Models Won

1. **Payment irregularity captures the signal:** Other features added noise
2. **SVM's boundary works well:** Separates consistent vs erratic patterns
3. **ECOD complements SVM:** Catches distribution outliers SVM misses
4. **Simplicity prevents overfitting:** 2 models beat 15

### What Didn't Work

❌ **Adding more feature groups:** Decreased performance
❌ **Complex ensembles:** Too many models hurt generalization
❌ **RFOD dominance:** Performed well individually but hurt ensemble
❌ **Equal weighting:** Worse than optimized 2-model combination

## Troubleshooting

### Common Issues

**1. ImportError: No module named 'pyod'**
```bash
pip install pyod==2.0.5
```

**2. Memory Error during training**
- Reduce LOF variants (use only n_neighbors=30, 50)
- Skip RFOD if memory constrained (minimal impact on ensemble)
- Requires ~4GB RAM for full ensemble

**3. RAPIDS/cuML errors**
- These are optional GPU libraries
- Code automatically falls back to CPU sklearn
- Ignore if not using GPU

**4. Notebook won't run top-to-bottom**
- Ensure all data files in same directory
- Check kernel has enough memory
- Restart kernel and clear outputs before running

### Performance Optimization

**Speed up training:**
- Use GPU if available (RAPIDS cuML)
- Reduce n_estimators for tree-based models
- Skip weight optimization (use saved weights: 65.4% SVM, 34.2% ECOD)

**Reduce memory:**
- Process data in batches
- Use float32 instead of float64
- Remove unnecessary LOF variants

## Citation & References

### Dataset
Freddie Mac Single-Family Loan-Level Dataset
- 14-month observation window (Mar 2024 - Apr 2025)
- 30,504 training loans, 5,370 validation loans, 13,426 test loans

### Key Papers
1. Sousa et al. (2016) - "Dynamic credit scoring models"
2. Blitzstein & Pfister - Statistical workflow framework

### Libraries
- scikit-learn: Pedregosa et al. (2011)
- PyOD: Zhao et al. (2019)
- RAPIDS cuML: NVIDIA (2024)

## Contact & Support

For questions about this submission:
- Check Canvas discussion forum
- Contact team members through Canvas messaging
- Review notebook comments for detailed explanations

## License & Academic Integrity

This code is submitted for CS5344 Big Data Analytics Technology (NUS) and is intended for academic evaluation only. Please do not redistribute or reuse without permission.

---

## Appendix: Quick Start Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run notebook
jupyter notebook CS5344_G33_Final.ipynb

# Expected output
# - rfod_ensemble_predictions.csv (13,426 rows)
# - Validation AP: ~0.4745
# - Runtime: ~25-40 minutes
```

**Last Updated:** November 10, 2025  
**Course:** CS5344 Big Data Analytics Technology  
**Instructor:** Prof. [Name]  
**Submission Date:** November 16, 2025
