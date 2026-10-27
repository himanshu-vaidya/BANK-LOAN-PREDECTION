# Bank Loan Default Prediction — Streamlit

## Files
- `app.py`: Streamlit web app
- `requirements.txt`: Python libraries

## Dataset requirements
Upload a CSV with a target column called `loan_status`:
- `0` = non-default
- `1` = default

The app supports numeric and categorical features. It automatically handles missing values, scales numeric features, one-hot encodes categorical features, trains Logistic Regression, and reports Accuracy, Precision, Recall, and F1 score.

## Run locally
1. Extract this ZIP.
2. Open Command Prompt/Terminal in this folder.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Start the app:

   `streamlit run app.py`

5. Open the local URL shown in the terminal (usually http://localhost:8501) and upload your CSV.

## Note
This is an educational prototype, not a real lending decision system. Ensure the target column is correctly encoded and remove sensitive personal information before sharing a dataset.
