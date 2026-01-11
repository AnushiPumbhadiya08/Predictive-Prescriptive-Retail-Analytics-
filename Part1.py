

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


def main():

    df = pd.read_csv("shopping_behavior_updated.csv")
    print("=== The first five lines of the raw data ===")
    print(df.head())
    print("\n=== Listed ===")
    print(df.columns)


    df["coupon_user"] = (
        (df["Discount Applied"] == "Yes") | (df["Promo Code Used"] == "Yes")
    ).astype(int)

    print("\nNumber of individuals who have utilised the voucher：", df["coupon_user"].sum())
    print("Number of individuals who have not utilised the voucher：", (df["coupon_user"] == 0).sum())



    feature_cols_numeric = [
        "Age",
        "Purchase Amount (USD)",
        "Previous Purchases",
    ]


    feature_cols_categorical = [

        "Category",
        "Location",
        "Season",
        "Subscription Status",
        "Shipping Type",
        "Payment Method",
        "Frequency of Purchases",
    ]

    X = df[feature_cols_numeric + feature_cols_categorical]
    y = df["coupon_user"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols_numeric),
            ("cat", categorical_transformer, feature_cols_categorical),
        ]
    )

    log_reg = LogisticRegression(max_iter=1000)

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", log_reg),
    ])


    clf.fit(X_train, y_train)


    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n=== Classification Report (threshold = 0.5) ===")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC: {auc:.3f}")


    threshold = 0.7
    y_recommend = (y_prob >= threshold).astype(int)

    print(f"\nRecommend send coupon to {y_recommend.sum()} out of {len(y_recommend)} customers")

    X_test_with_result = X_test.copy()
    X_test_with_result["coupon_prob"] = y_prob
    X_test_with_result["recommend_send_coupon"] = y_recommend
    X_test_with_result["true_coupon_user"] = y_test.values

    print("\n=== Test set results example (first 5 rows) ===")
    print(X_test_with_result.head())




    coupon_users = df[df["coupon_user"] == 1].copy()
    total_coupon_users = len(coupon_users)
    print("\nNumber of users who have used the voucher：", total_coupon_users)


    coupon_users["AgeGroup"] = pd.cut(
        coupon_users["Age"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=["0-25", "26-35", "36-45", "46-55", "56-65", "66+"]
    )


    target_cols = [
        "Gender",
        "AgeGroup",
        "Location",
        "Season",
        "Category",
        "Frequency of Purchases",
        "Payment Method",
        "Shipping Type",
    ]

    summary_rows = []

    for col in target_cols:
        vc = coupon_users[col].value_counts(dropna=False)
        top_value = vc.index[0]
        top_count = vc.iloc[0]
        top_ratio = top_count / total_coupon_users

        summary_rows.append({
            "attribute": col,
            "most_common_value": str(top_value),
            "count": int(top_count),
            "ratio": round(top_ratio, 3),
        })

    summary_df = pd.DataFrame(summary_rows)

    print("\n=== Among those who have used coupons, the most common values for each attribute (user profile) ===")
    print(summary_df)




    try:
        feature_names_num = feature_cols_numeric
        feature_names_cat = clf.named_steps["preprocessor"] \
            .named_transformers_["cat"] \
            .get_feature_names_out(feature_cols_categorical)

        all_feature_names = np.concatenate([feature_names_num, feature_names_cat])
        coefs = clf.named_steps["model"].coef_[0]

        coef_df = pd.DataFrame({
            "feature": all_feature_names,
            "coef": coefs
        })

        coef_df_sorted = coef_df.sort_values("coef", ascending=False)

        print("\n=== The most influential positive characteristic affecting coupon usage===")
        print(coef_df_sorted.head(10))

        print("\n=== The most detrimental characteristic affecting coupon usage ===")
        print(coef_df_sorted.tail(10))

    except Exception as e:
        print("\nError in feature importance calculation (can be ignored):", e)

    coupon_users = df[df["coupon_user"] == 1].copy()
    coupon_users.to_csv("coupon_users_only.csv", index=False)


if __name__ == "__main__":
    main()
