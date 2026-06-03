import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# LOAD DATASET
# ==========================
df = pd.read_csv("credit_data.csv")

print("\nDataset Preview:\n")
print(df.head())

print("\nMissing Values:\n")
print(df.isnull().sum())

# ==========================
# HANDLE MISSING VALUES
# ==========================

# Numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns
categorical_cols = df.select_dtypes(include=['object','string']).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ==========================
# ENCODE CATEGORICAL COLUMNS
# ==========================
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ==========================
# FEATURES & TARGET
# ==========================
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# ==========================
# TRAIN TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# TRAIN MODEL
# ==========================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# MODEL EVALUATION
# ==========================
y_pred = model.predict(X_test)

print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================
# FEATURE IMPORTANCE
# ==========================
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:\n")
print(importance_df)

# ==========================
# SAVE MODEL
# ==========================
joblib.dump(model, "credit_scoring_model.pkl")

print("\nModel saved successfully!")

# ==========================
# SAMPLE PREDICTION
# ==========================

sample = pd.DataFrame([{
    "person_age": 25,
    "person_income": 50000,
    "person_home_ownership": 1,
    "person_emp_length": 5,
    "loan_intent": 2,
    "loan_grade": 1,
    "loan_amnt": 10000,
    "loan_int_rate": 12.5,
    "loan_percent_income": 0.20,
    "cb_person_default_on_file": 0,
    "cb_person_cred_hist_length": 3
}])

prediction = model.predict(sample)

print("\nSample Prediction:")

if prediction[0] == 1:
    print("Customer is NOT Creditworthy")
else:
    print("Customer is Creditworthy")