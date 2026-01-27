

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


def main():

    df = pd.read_csv("shopping_behavior_updated.csv")


    for col in ["Purchase Amount (USD)", "Age", "Previous Purchases", "Review Rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


    if "Discount Applied" in df.columns:
        df["DiscountFlag"] = df["Discount Applied"].map({"Yes": 1, "No": 0})
    else:
        df["DiscountFlag"] = np.nan

    if "Promo Code Used" in df.columns:
        df["PromoFlag"] = df["Promo Code Used"].map({"Yes": 1, "No": 0})
    else:
        df["PromoFlag"] = np.nan


    df = df.dropna(subset=["Purchase Amount (USD)"]).copy()

    




    user_numeric = ["Age"]
    user_categorical = ["Location", "Subscription Status", "Frequency of Purchases"]


    product_numeric = ["Previous Purchases", "Review Rating"]
    product_categorical = ["Category", "Item Purchased", "Season"]


    marketing_numeric = ["DiscountFlag", "PromoFlag"]
    marketing_categorical: list[str] = []


    logistics_numeric: list[str] = []
    logistics_categorical = ["Payment Method", "Shipping Type"]


    numeric_features_all = (
        user_numeric
        + product_numeric
        + marketing_numeric
        + logistics_numeric
    )
    categorical_features_all = (
        user_categorical
        + product_categorical
        + marketing_categorical
        + logistics_categorical
    )


    numeric_features = [c for c in numeric_features_all if c in df.columns]
    categorical_features = [c for c in categorical_features_all if c in df.columns]

    print("\nNumerical characteristics：", numeric_features)
    print("Category characteristics：", categorical_features)


    y = df["Purchase Amount (USD)"]
    X = df[numeric_features + categorical_features].copy()


    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n=== Model performance for predicting the payment amount per order ===")
    results = []

    for name, reg in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("reg", reg),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append((name, rmse, r2))
        print(f"{name:16s}  RMSE = {rmse:6.3f}   R^2 = {r2:6.3f}")


    best_name = max(results, key=lambda x: x[2])[0]
    print(f"\nCurrently, the model with the best R-squared performance is：{best_name}")

    best_reg = models[best_name]


    best_pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("reg", best_reg),
    ])
    best_pipe.fit(X, y)


    ohe = best_pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    num_feature_names = numeric_features
    cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = num_feature_names + cat_feature_names


    importances = best_pipe.named_steps["reg"].feature_importances_
    fi_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    fi_df.to_csv("txn_amount_feature_importance_raw.csv", index=False)
    print("\n[Saved] txn_amount_feature_importance_raw.csv ")
    print("=== Top 10 Key Features ===")
    print(fi_df.head(10))


    col_to_group = {}

    for col in user_numeric + user_categorical:
        col_to_group[col] = "user"
    for col in product_numeric + product_categorical:
        col_to_group[col] = "product"
    for col in marketing_numeric + marketing_categorical:
        col_to_group[col] = "marketing"
    for col in logistics_numeric + logistics_categorical:
        col_to_group[col] = "logistics"

    def map_feature_to_group(feat_name: str) -> str:

        for col in categorical_features:
            prefix = col + "_"
            if feat_name.startswith(prefix):
                return col_to_group.get(col, "OTHER")


        if feat_name in col_to_group:
            return col_to_group[feat_name]

        return "OTHER"

    fi_df["group"] = fi_df["feature"].apply(map_feature_to_group)
    group_imp = (
        fi_df.groupby("group", as_index=False)["importance"]
             .sum()
             .sort_values("importance", ascending=False)
    )
    total_imp = group_imp["importance"].sum()
    group_imp["importance_pct"] = group_imp["importance"] / total_imp * 100

    group_imp.to_csv("txn_amount_group_importance.csv", index=False)
    print("\n[Saved] txn_amount_group_importance.csv 已生成（按 user/product/marketing/logistics 汇总）")
    print("=== By attribute category importance (relative to each order amount) ===")
    for _, row in group_imp.iterrows():
        print(f"{row['group']:10s}  Importance = {row['importance_pct']:5.1f}%")



if __name__ == "__main__":
    main()
