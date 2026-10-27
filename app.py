import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Bank Loan Default Prediction", page_icon="🏦", layout="wide")
st.title("🏦 Bank Loan Default Prediction")
st.write("Upload a loan dataset, train a simple machine-learning model, and estimate whether a loan applicant may default.")

st.sidebar.header("1. Upload dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

st.info("Your CSV must contain a target column named `loan_status`, where 0 = non-default and 1 = default.")

if uploaded_file is None:
    st.markdown("""
    ### How to use this app
    1. Upload your bank-loan CSV file from the sidebar.
    2. The app cleans the data and trains a Logistic Regression model.
    3. Review the evaluation scores.
    4. Enter an applicant's details to estimate default risk.

    **Expected target:** `loan_status` (0 = non-default, 1 = default).

    **Common dataset features:** `person_age`, `person_income`, `person_emp_length`,
    `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length`,
    and `person_home_ownership`, `loan_intent`, `loan_grade`, `cb_person_default_on_file`.
    """)
    st.stop()

try:
    raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Unable to read CSV: {e}")
    st.stop()

if "loan_status" not in raw.columns:
    st.error("The dataset must contain a column named `loan_status`. Rename your target column to `loan_status` and encode it as 0 or 1.")
    st.stop()

df = raw.copy()
df = df.drop_duplicates()
df["loan_status"] = pd.to_numeric(df["loan_status"], errors="coerce")
df = df[df["loan_status"].isin([0, 1])].copy()

if len(df) < 20 or df["loan_status"].nunique() != 2 or df["loan_status"].value_counts().min() < 2:
    st.error("Not enough usable data or both target classes are not sufficiently represented. Upload a larger dataset containing both 0 and 1 in `loan_status`.")
    st.stop()

# Avoid using ID-like fields and the target itself as input features.
drop_cols = [c for c in df.columns if c.lower() in ["loan_id", "id", "customer_id"]]
X = df.drop(columns=["loan_status"] + drop_cols)
y = df["loan_status"].astype(int)

# Keep columns that have at least one non-missing value.
X = X.dropna(axis=1, how="all")
if X.shape[1] == 0:
    st.error("No usable input features were found.")
    st.stop()

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []
if numeric_cols:
    transformers.append(("numeric", numeric_pipe, numeric_cols))
if categorical_cols:
    transformers.append(("categorical", categorical_pipe, categorical_cols))

preprocessor = ColumnTransformer(transformers=transformers)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
except Exception as e:
    st.error(f"Model training failed. Please check your CSV columns and values. Details: {e}")
    st.stop()

st.success(f"Loaded {len(df):,} rows and trained using {X.shape[1]} input features.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{accuracy_score(y_test, predictions):.3f}")
m2.metric("Precision", f"{precision_score(y_test, predictions, zero_division=0):.3f}")
m3.metric("Recall", f"{recall_score(y_test, predictions, zero_division=0):.3f}")
m4.metric("F1 score", f"{f1_score(y_test, predictions, zero_division=0):.3f}")

with st.expander("Preview cleaned dataset"):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("Default vs non-default distribution"):
    counts = df["loan_status"].value_counts().sort_index()
    counts.index = ["Non-default (0)" if i == 0 else "Default (1)" for i in counts.index]
    st.bar_chart(counts)

st.subheader("2. Estimate an applicant's default risk")
st.write("Enter values for the applicant below. Features not shown here use the most frequent or median value from the training data.")

input_values = {}
left, right = st.columns(2)
for i, col in enumerate(X.columns):
    target_col = left if i % 2 == 0 else right
    series = X[col]
    with target_col:
        if col in numeric_cols:
            valid = pd.to_numeric(series, errors="coerce").dropna()
            default_value = float(valid.median()) if not valid.empty else 0.0
            min_value = float(valid.min()) if not valid.empty else 0.0
            max_value = float(valid.max()) if not valid.empty else max(default_value + 1, 1.0)
            if min_value == max_value:
                max_value = min_value + 1.0
            step = 1.0 if pd.api.types.is_integer_dtype(series.dtype) else 0.1
            input_values[col] = st.number_input(
                col.replace("_", " ").title(),
                min_value=min_value,
                max_value=max_value,
                value=min(max(default_value, min_value), max_value),
                step=step,
                key=f"num_{col}"
            )
        else:
            options = series.dropna().astype(str).value_counts().index.tolist()
            if not options:
                options = ["Unknown"]
            input_values[col] = st.selectbox(
                col.replace("_", " ").title(),
                options=options,
                index=0,
                key=f"cat_{col}"
            )

if st.button("Predict default risk", type="primary"):
    applicant = pd.DataFrame([input_values], columns=X.columns)
    for col in categorical_cols:
        applicant[col] = applicant[col].astype(str)
    try:
        risk = float(model.predict_proba(applicant)[0][1])
        prediction = int(risk >= 0.5)
        st.subheader("Prediction result")
        st.metric("Estimated default probability", f"{risk * 100:.1f}%")
        if prediction == 1:
            st.error("Model prediction: Higher default risk (class 1)")
        else:
            st.success("Model prediction: Lower default risk (class 0)")
        st.caption("This is an educational prediction, not financial advice or a real lending decision.")
    except Exception as e:
        st.error(f"Could not generate prediction: {e}")

st.warning("Important: Results depend on the dataset and its quality. Accuracy alone is not enough to judge a credit-risk model; review precision, recall, and F1 as well.")
