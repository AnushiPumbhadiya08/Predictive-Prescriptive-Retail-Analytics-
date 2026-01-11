
import pandas as pd


def main():

    df = pd.read_csv("shopping_behavior_updated.csv")


    df["Purchase Amount (USD)"] = pd.to_numeric(
        df["Purchase Amount (USD)"], errors="coerce"
    )

    group_item = (
        df.groupby(["Season", "Location", "Item Purchased"], as_index=False)
          .agg(
              sales_count=("Item Purchased", "size"),
              total_revenue=("Purchase Amount (USD)", "sum"),
          )
    )

    group_item["rank_in_season_region"] = (
        group_item
        .groupby(["Season", "Location"])["sales_count"]
        .rank(method="dense", ascending=False)
    )

    total_sales_per_sr = group_item.groupby(
        ["Season", "Location"]
    )["sales_count"].transform("sum")
    group_item["sales_share_in_season_region"] = (
        group_item["sales_count"] / total_sales_per_sr
    )

    top5_sr = group_item[group_item["rank_in_season_region"] <= 5].copy()

    top5_sr = top5_sr.sort_values(
        ["Season", "Location", "rank_in_season_region"],
        ascending=[True, True, True],
    )

    top5_sr.to_csv("season_region_top5_items.csv", index=False)
    print("[Saved] season_region_top5_items.csv ")


    overall_item = (
        df.groupby("Item Purchased", as_index=False)
          .agg(
              sales_count=("Item Purchased", "size"),
              total_revenue=("Purchase Amount (USD)", "sum"),
          )
    )


    overall_item = overall_item.sort_values(
        "sales_count", ascending=False
    )


    overall_item["rank_overall"] = overall_item["sales_count"] \
        .rank(method="dense", ascending=False)


    top5_overall = overall_item.head(5).copy()

    top5_overall.to_csv("overall_top5_items.csv", index=False)
    print("[Saved] overall_top5_items.csv ）")


    print("\n=== Top 5 best-selling products ===")
    print(top5_overall)

    print("\n=== Top 5 Products for Each Season and Region ===")
    print(top5_sr.head(10))


if __name__ == "__main__":
    main()
