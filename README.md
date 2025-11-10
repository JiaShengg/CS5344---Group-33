# CS5344 Group 33 - Loan-Level Anomaly Detection

**Team Members:** Jia Sheng | Hu Qingxuan | Gong Zihong

## Project Overview

This project develops an ensemble anomaly detection system for identifying problematic loan repayment behavior in mortgage data. Using unsupervised machine learning on a 13-month observation window, we achieved **0.4745 Average Precision** on the validation set, representing a **28.3% improvement** over baseline methods.

### Problem Statement

Financial institutions face significant losses from loan defaults. Traditional credit scoring models evaluate borrowers only at origination, missing behavioral patterns that emerge during repayment. Our approach:

- Analyzes both static borrower information and temporal payment sequences
- Detects abnormal repayment behavior through payment irregularity patterns
- Uses ensemble of 15 anomaly detection models with optimized weights
- Handles severe class imbalance (87.4% normal, 12.6% anomaly)

### Key Results

- **Validation Average Precision:** 0.4745
- **Best Feature Group:** Payment Irregularity (29 features)
- **Optimal Ensemble:** One-Class SVM (65.4%) + ECOD (34.2%)
- **Contamination Parameter:** 0.05

## File Structure

```
Group33Codebase/
├── README.md                           
├── requirements.txt                   # Package dependencies
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

**Last Updated:** November 10, 2025  
**Course:** CS5344 Big Data Analytics Technology  
