
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def main():
    df = pd.read_csv("shopping_behavior_updated.csv")

    df["Review Rating"] = pd.to_numeric(df["Review Rating"], errors="coerce")
    df["Purchase Amount (USD)"] = pd.to_numeric(df["Purchase Amount (USD)"], errors="coerce")

    df = df.dropna(subset=["Review Rating"]).copy()


    numeric_features = ["Purchase Amount (USD)"]

    categorical_features = [
        "Category",
        "Season",
        "Location",
        "Shipping Type",
        "Discount Applied",
        "Promo Code Used",
        "Subscription Status",
        "Frequency of Purchases",
    ]

    feature_cols = numeric_features + categorical_features


    df_model = df[feature_cols + ["Review Rating"]].dropna(subset=categorical_features).copy()

    X = df_model[feature_cols]
    y = df_model["Review Rating"]


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


    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("rf", rf),
    ])


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("=== RandomForestRegressor ===")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.3f}")
    print(f"R^2 : {r2:.3f}")


    rf_model = model.named_steps["rf"]
    ohe = model.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]


    num_feature_names = numeric_features

    cat_feature_names = list(ohe.get_feature_names_out(categorical_features))

    all_feature_names = num_feature_names + cat_feature_names
    importances = rf_model.feature_importances_

    fi_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)


    def map_to_attr(feat_name: str) -> str:

        for col in categorical_features:
            if feat_name.startswith(col + "_"):
                return col

        if feat_name in numeric_features:
            return feat_name
        return "OTHER"

    fi_df["attr"] = fi_df["feature"].apply(map_to_attr)

    attr_importance = (
        fi_df.groupby("attr", as_index=False)["importance"]
             .sum()
             .sort_values("importance", ascending=False)
    )


    total_imp = attr_importance["importance"].sum()
    attr_importance["importance_pct"] = attr_importance["importance"] / total_imp * 100

    print("\n=== Importance of Different Attributes on Review Rating (Aggregated by Attribute) ===")
    for _, row in attr_importance.iterrows():
        attr = row["attr"]
        pct = row["importance_pct"]
        print(f"{attr:25s}  Importance = {pct:5.1f}%")


    attr_importance.to_csv("rating_attribute_importance.csv", index=False)
    print("\n[Saved] rating_attribute_importance.csv ")


if __name__ == "__main__":
    main()
