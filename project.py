import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv("your_dataset.csv")  # 🔴 CHANGE PATH

print(df.head())
print(df.info())

# =========================
# 2. BASIC EDA
# =========================
sns.countplot(x="Accident", data=df)
plt.title("Accident Distribution")
plt.show()

sns.boxplot(x="Accident", y="Traffic_Density", data=df)
plt.title("Traffic Density vs Accident")
plt.show()

# =========================
# 3. FEATURE SELECTION
# =========================
features = [
    "Hour",
    "Traffic_Density",
    "Weather",
    "Road_Type",
    "Visibility"
]

target = "Accident"

X = df[features]
y = df[target]

# =========================
# 4. PREPROCESSING
# =========================
numeric_features = ["Hour", "Traffic_Density", "Visibility"]
categorical_features = ["Weather", "Road_Type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# =========================
# 5. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# =========================
# 6. LOGISTIC REGRESSION
# =========================
log_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

print("\nLOGISTIC REGRESSION RESULTS")
print(classification_report(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_log))

# =========================
# 7. DECISION TREE
# =========================
tree_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=50,
        random_state=42
    ))
])

tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)
y_prob_tree = tree_model.predict_proba(X_test)[:, 1]

print("\nDECISION TREE RESULTS")
print(classification_report(y_test, y_pred_tree))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_tree))

# =========================
# 8. FEATURE IMPORTANCE (LOGISTIC REGRESSION)
# =========================
feature_names = (
    numeric_features +
    list(
        log_model.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_features)
    )
)

coefficients = log_model.named_steps["classifier"].coef_[0]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", ascending=False)

print("\nTOP RISK-INCREASING FEATURES")
print(importance_df.head(10))

print("\nTOP RISK-REDUCING FEATURES")
print(importance_df.tail(10))

# =========================
# 9. DECISION TREE VISUALIZATION
# =========================
plt.figure(figsize=(20, 10))
plot_tree(
    tree_model.named_steps["classifier"],
    feature_names=feature_names,
    class_names=["No Accident", "Accident"],
    filled=True
)
plt.title("Decision Tree – Accident Risk")
plt.show()
