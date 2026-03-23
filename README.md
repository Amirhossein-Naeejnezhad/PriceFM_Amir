# PriceFM
Foundation Model for Probabilistic (Day-Ahead) Electricity Price Forecasting

🦊 Summary page: https://runyao-yu.github.io/PriceFM/

🌋 Paper link: https://www.arxiv.org/pdf/2508.04875

![Description of Image](Figure/PriceFM_structure.PNG)

---

## 📢 Updates

### Mar 2026
- Dataset extended from **2022–2025** to **2022–2026**
- Temporal resolution increased from **hourly** to **quarter-hourly**
- Evaluation expanded from **1 test fold in 2025** to **3 test folds in 2026**
- Model architecture upgraded to a **Mixture-of-Experts (MoE)** design
- Graph mask simplified from a decay formulation to a **sparse mask**

---

## 🚀 Quick Start

We open-source all code for preprocessing, modeling, and analysis.  
The project directory is structured as follows:

    PriceFM/
    ├── Data/
    ├── Figure/
    ├── Model/
    ├── Result/
    ├── PriceFM/
        ├── data.py
        ├── model.py
        ├── evaluation.py
    ├── Tutorial.ipynb
    ├── README.md

The file `README.md` specifies the required package versions.

To facilitate reproducibility and accessibility, we have streamlined the entire pipeline into just three simple steps:

### 🌵 Step 1: Download the dataset

You can download the dataset from https://huggingface.co/datasets/RunyaoYu/PriceFM/tree/main 
Ensure that the energy dataset `FINAL.csv` is in the `Data` folder.

### 🌵 Step 2: Run the Pipeline

Run `Tutorial.ipynb` to:
- Preprocess the energy data
- Train, validate, and test the PriceFM model

Note that there are two phases, i.e. pretraining without graph topology and full-training with topology. 

### 🌵 Step 3: Check Results

After execution, check:
- `Model/` for saved model weights  
- `Result/` for evaluation metrics and outputs

---

## 📦 Environment & Dependencies

This project has been tested with the following environment:

- **Python 3.9.20**
- `numpy==1.25.2`
- `pandas==2.1.4`
- `scikit-learn==1.5.1`
- `scipy==1.13.1`
- `tensorflow==2.16.2`
- `protobuf>=3.19.0`
- `h5py>=3.1.0`
- `joblib`
- `setuptools`

Use the following comment to pip install:

```bash
pip install numpy==1.25.2 pandas==2.1.4 scikit-learn==1.5.1 scipy==1.13.1 tensorflow==2.16.2 protobuf>=3.19.0 h5py>=3.1.0 joblib setuptools
