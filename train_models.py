import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ------------------ CONFIG ------------------
DATA_FILE = "dataset_full.csv"      # CSV file with features + phishing column
MODEL_FILE = "best_model.pkl"
FEATURE_FILE = "feature_columns.pkl"
# --------------------------------------------

# 1) Load dataset
print("📂 Loading dataset...")
df = pd.read_csv(DATA_FILE)

# 2) Separate features (X) and target (y)
if "phishing" not in df.columns:
    raise ValueError("❌ Target column 'phishing' not found in dataset!")

X = df.drop(columns=["phishing"])   # all features
y = df["phishing"]                  # target

# ✅ Save feature columns for later use in Flask app
FEATURE_COLUMNS = X.columns.tolist()
joblib.dump(FEATURE_COLUMNS, FEATURE_FILE)
print(f"✅ Saved feature columns to {FEATURE_FILE}")

# 3) Train/Test split (stratified to preserve phishing/safe ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Define candidate models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1),
    "Gaussian Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# 5) Train & evaluate each model
best_model = None
best_acc = 0
best_name = None
results = {}

print("\n🚀 Training models...\n")
for name, model in models.items():
    print(f"🔹 Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(f"   {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, digits=4))

    # track best
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

# 6) Save best model
joblib.dump(best_model, MODEL_FILE)

print("\n===================================")
print(f"🏆 Best Model: {best_name} (Accuracy: {best_acc:.4f})")
print(f"✅ Saved model as {MODEL_FILE}")
print(f"✅ Saved feature columns as {FEATURE_FILE}")
print("===================================")
