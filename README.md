# Telecom Customer Retention Assistant

Live app: Add your Streamlit URL here after deployment.

Predicts a telecom customer's churn probability and suggests possible retention actions.

## Model
Final model: tuned, class-weighted Logistic Regression pipeline.

Test-set metrics:
- Accuracy: 74.32%
- Precision: 50.97%
- Recall: 77.63%
- F1-score: 61.54%
- ROC AUC: 84.66%

## Key findings
- Month-to-month customers had substantially higher churn than customers on longer contracts.
- Customers who left had shorter average tenure.
- Customers who left had higher average monthly charges.
- Electronic-check users had a higher observed churn rate.

These are associations in the dataset, not proof that any factor causes churn.

## Files required
- `app.py`
- `requirements.txt`
- `telecom_churn_model.pkl`
- `model_features.pkl`

The Kaggle dataset file and project notebook can also be included in the repository if permitted by the dataset's licence/terms.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Built with
Python, pandas, scikit-learn, joblib, Streamlit.

Dataset: IBM Telco Customer Churn dataset as obtained from Kaggle. Check the dataset's licence and attribution requirements before redistribution.
