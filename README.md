# Financial Fraud Detection with Explainable AI

This project combines fraud prediction with explainability so the model output is easier to trust, inspect, and present. It includes exploratory notebooks, model development work, and a Streamlit application for real-time scoring with XAI visuals.

## What this project covers

- Fraud detection on the IEEE-CIS + Vesta transaction dataset
- Model comparison across classical machine learning approaches
- Explainability with SHAP, LIME, and feature importance views
- A Streamlit app for interactive fraud scoring

## Dataset

- Source: [IEEE-CIS Fraud Detection on Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection)
- Main files used in the project:
  - `train_transaction.csv`
  - `train_identity.csv`

The fraud rate is highly imbalanced, so model evaluation should focus on metrics such as recall, precision, and ROC-AUC instead of accuracy alone.

## Repository structure

```text
Financial-Fraud-Detection-Project/
├── Explainable AI (XAI)/
├── Machine-Learning/
├── WebApp/
│   ├── app.py
│   ├── final_model1.sav
│   └── sample_data.example.csv
└── README.md
```

## Streamlit app

The app lives in `WebApp/app.py` and includes two areas:

- `Prediction`: score a single transaction and inspect the most influential features
- `XAI Dashboard`: explore feature importance, SHAP charts, and LIME explanations

### Local setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

The bundled `final_model1.sav` was serialized from a LightGBM-based model, so `lightgbm` is part of the required runtime dependencies.

3. Run the app:

```bash
streamlit run WebApp/app.py
```

4. Open the local URL shown by Streamlit, usually `http://localhost:8501`.

## Runtime files

- `WebApp/final_model1.sav`: trained fraud model required by the app
- `WebApp/sample_data.csv`: optional reference dataset for stronger SHAP/LIME examples
- `WebApp/sample_data.example.csv`: starter template you can duplicate into `sample_data.csv`

If `sample_data.csv` is missing, the app now generates a deterministic synthetic fallback dataset and clearly labels the dashboard as demo-only. That keeps the app runnable, but the best explainability results come from real reference rows.

## Suggested sample fields

The app currently expects these 10 features:

- `card1`
- `card2`
- `card4`
- `card6`
- `addr1`
- `addr2`
- `TransactionAmt`
- `P_emaildomain`
- `ProductCD`
- `DeviceType`

Optional label column for dashboard analysis:

- `isFraud`

## Portfolio talking points

- Practical fraud detection use case with business relevance
- Clear emphasis on model transparency, not just raw performance
- Interactive delivery through Streamlit for demos and interviews
- Good example of combining predictive ML with explainability techniques

## Current limitations

- The repository does not include the original full training dataset because of file size
- The web app depends on the serialized model already matching the 10 exposed features
- SHAP and LIME require their extra Python packages to be installed locally

## Recommended next improvements

- Export a real, anonymized `sample_data.csv` from the modeling pipeline
- Add model metadata such as training date, feature order, and threshold used in production
- Save evaluation plots and confusion matrices as static assets for the README
- Add a lightweight smoke test for the Streamlit app startup
