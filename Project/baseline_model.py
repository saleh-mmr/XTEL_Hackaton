import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
def mape_per_product_avg(y_true, y_pred, product_ids, eps=1e-8):
    tmp = pd.DataFrame({
        "PRODUCT_ID": product_ids,
        "y_true": y_true,
        "y_prediction": y_pred
    })
    tmp["ape"] = (tmp["y_true"] - tmp["y_prediction"]).abs() / (tmp["y_true"].abs() + eps)
    return tmp.groupby("PRODUCT_ID")["ape"].mean().mean()

train_path = "weekly-train-1.csv"
df = pd.read_csv(train_path)
df["DATE"] = pd.to_datetime(df["DATE"])
bool_cols = [c for c in df.columns if c.startswith("IS_")]
for c in bool_cols:
    df[c] = df[c].astype(int)

cutoff = df["DATE"].max() - pd.Timedelta(weeks=26)
train_df = df[df["DATE"] <= cutoff].copy()
val_df   = df[df["DATE"] > cutoff].copy()
print("Train size:", train_df.shape)
print("Val size:", val_df.shape)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "TOTAL_SALES" in numeric_cols:
    numeric_cols.remove("TOTAL_SALES")
feature_cols = numeric_cols

X_train = train_df[feature_cols].copy()
y_train = train_df["TOTAL_SALES"].copy()
X_val = val_df[feature_cols].copy()
y_val = val_df["TOTAL_SALES"].copy()

X_train = X_train.dropna()
X_val = X_val.dropna()

model = LinearRegression()
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
val_pred = np.maximum(val_pred, 0)
val_pid = val_df["PRODUCT_ID"].values
score = mape_per_product_avg(y_val.values, val_pred, val_pid)

print("Linear Regression MAPE (avg over products):", score)
