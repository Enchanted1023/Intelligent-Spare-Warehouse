# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


# for _, row in target_df.iterrows():
#     mihome = row["mihome"]
#     channel = row["channel"]
#     goods_id = row["goods_id"]
#
#     part_train_data_df = train_data[(train_data["mihome"] == mihome) &
#                                     (train_data["channel"] == channel) &
#                                     (train_data["goods_id"] == goods_id)]
#
#     goods_type_unique_list = list(set(part_train_data_df["goods_type"].to_list()))
#     if (len(goods_type_unique_list) == 1 and goods_type_unique_list[0] == "new") \
#             or part_train_data_df["qty"].sum() == 0:
#         continue
#
#     print(part_train_data_df)
#     assert len(part_train_data_df) == 30


def moving_average(train_data_df):
    train_data_df = train_data_df.sort_values(by=["mihome", "channel", "goods_id", "day"], ascending=True)
    train_data_df["predict_agg"] = train_data_df.groupby(["mihome", "channel", "goods_id"])["qty"].transform(
        lambda x: x.shift(1).rolling(7).mean() * 7)

    train_data_df = train_data_df.sort_values(by=["mihome", "channel", "goods_id", "day"], ascending=False)
    train_data_df["true_agg"] = train_data_df.groupby(["mihome", "channel", "goods_id"])["qty"].transform(
        lambda x: x.rolling(7).sum())

    train_data_df[["mihome", "channel", "goods_id", "day", "qty", "predict_agg", "true_agg"]] \
        .sort_values(by=["mihome", "channel", "goods_id", "day"], ascending=True) \
        .to_csv("result.csv")

    train_data_df = train_data_df.dropna()
    train_data_df = train_data_df[(train_data_df["day"] >= 20220801) &
                                  (train_data_df["day"] <= 20220814)]
    train_data_df = train_data_df[train_data_df["true_agg"] != 0]
    # train_data_df = train_data_df[train_data_df["goods_type"] == "regular"]
    print()
    true_sum = train_data_df["true_agg"].sum()
    predict_sum = train_data_df["predict_agg"].sum()

    print(f"测评数据条数:{len(train_data_df)}")
    print(f"true_sum: {true_sum}")
    print(f"predict_sum: {predict_sum}")

    train_data_df["abs_err"] = abs(train_data_df["predict_agg"] - round(train_data_df["true_agg"]))
    train_data_df["mape"] = train_data_df["abs_err"] / train_data_df["true_agg"]
    print("MAPE", train_data_df["mape"].mean())
    print("wMAPE", train_data_df["abs_err"].sum() / train_data_df["true_agg"].sum())
    train_data_df.to_csv("naive_predict_563.csv", index=False, header=True)


def test():
    # 目标仓库（24个）
    mihome_set = {563, 432, 14189, 420, 359, 112,
                  348, 14187, 505, 493, 27165, 38949,
                  14185, 4743, 455, 457, 14183, 454,
                  719, 463, 6883, 14180, 28105, 1870}
    mihome_set = {563}

    raw_train_data_df = pd.read_csv("../data/raw_train_data.csv")
    print(f"数据条数(整体原始数据): {len(raw_train_data_df)}")

    # 过滤仓库
    train_data_df = raw_train_data_df[raw_train_data_df["mihome"].isin(mihome_set)]
    print(f"数据条数(过滤目标仓库): {len(train_data_df)}")
    moving_average(train_data_df)

    # 过滤首次售卖日
    train_data_df = train_data_df[train_data_df["day"] >= train_data_df["first_sale_day"]]

    base_data_num = len(train_data_df)
    print(f"数据条数(过滤首次售卖): {base_data_num}")
    date_set = set(train_data_df["day"].to_list())
    print(f"最小日期:{min(date_set)}, 最大日期:{max(date_set)}\n")

    # 过滤有销量的数据
    non_zero_data_num = len(train_data_df[train_data_df["qty"] > 0])
    print(f"数据条数(过滤零销量): {non_zero_data_num}")
    print(f"非零销量数据占比:{round(non_zero_data_num * 100 / base_data_num, 2)}%\n")

    # SKU个数
    sku_num = len(train_data_df[train_data_df["qty"] > 0]["goods_id"].drop_duplicates())
    print(f"SKU个数: {sku_num}")

    # 渠道个数
    channel_num = len(train_data_df[train_data_df["qty"] > 0]["channel"].drop_duplicates())
    print(f"Channel个数: {channel_num}")

    # 筛选8月份有销量的 商品、仓库、渠道
    target_df = train_data_df[
        # (train_data_df["day"] >= 20220801) &
        (train_data_df["qty"] > 0)
    ]
    target_df = target_df[["mihome", "channel", "goods_id"]].drop_duplicates()
    count_df = target_df.groupby(["mihome", "channel"], as_index=False)["goods_id"].count()
    count_df.to_csv("count.csv", index=False, header=True)


if __name__ == "__main__":
    test()
