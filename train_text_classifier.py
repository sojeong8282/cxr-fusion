import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

# ==========================================
# 1. Text feature 불러오기
# ==========================================

data = np.load(
    "data/biomedclip_text_features.npz",
    allow_pickle=True
)

X = data["features"]
y = data["labels"].astype(int)
split = data["split"]

print("Feature shape:", X.shape)
print("Label shape:", y.shape)


# ==========================================
# 2. Train / Val / Test
# ==========================================

train_mask = split == "train"
val_mask = split == "val"
test_mask = split == "test"

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("\n===== Dataset =====")
print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


# ==========================================
# 3. Validation에서 C 선택
# ==========================================

candidates = [0.01, 0.1, 1.0, 10.0]

best_model = None
best_c = None
best_auc = -1

print("\n===== Validation =====")

for c in candidates:

    model = LogisticRegression(
        C=c,
        max_iter=3000,
        random_state=42
    )

    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, val_prob)
    auprc = average_precision_score(y_val, val_prob)

    print(
        f"C={c:<5} "
        f"AUROC={auc:.4f} "
        f"AUPRC={auprc:.4f}"
    )

    if auc > best_auc:
        best_auc = auc
        best_c = c
        best_model = model


print("\nBest C:", best_c)
print("Best Validation AUROC:", round(best_auc, 4))


# ==========================================
# 4. Test 평가
# ==========================================

test_prob = best_model.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, test_prob)
test_auprc = average_precision_score(y_test, test_prob)
test_brier = brier_score_loss(y_test, test_prob)

print("\n===== C: Clinical Text-only TEST =====")
print(f"AUROC : {test_auc:.4f}")
print(f"AUPRC : {test_auprc:.4f}")
print(f"Brier : {test_brier:.4f}")


# ==========================================
# 5. 저장
# ==========================================

joblib.dump(
    best_model,
    "data/biomedclip_text_logistic.pkl"
)

test_results = pd.DataFrame({
    "patient_id": data["patient_id"][test_mask],
    "path_to_image": data["path_to_image"][test_mask],
    "label": y_test,
    "probability": test_prob
})

test_results.to_csv(
    "data/biomedclip_text_test_predictions.csv",
    index=False
)

print("\n모델 저장 완료")
print("Test prediction 저장 완료")