# -*- coding: utf-8 -*-
import warnings

import pandas as pd

from src import time_util


class Metrics(object):

    @classmethod
    def compute_mape(cls, pdf: pd.DataFrame):
        model_flag = pdf.iloc[0]["model"]
        pdf = pdf[pdf["y"] > 0]

        pdf["ape"] = abs(pdf["y"] - pdf["y_pred"]) / pdf["y"]
        mape = pdf["ape"].mean()
        print(f"### {model_flag} mape: {mape}")

    @classmethod
    def compute_wmape(cls, pdf: pd.DataFrame):
        model_flag = pdf.iloc[0]["model"]
        pdf["abs_error"] = abs(pdf["y"] - pdf["y_pred"])
        wmape = pdf["abs_error"].sum() / pdf["y"].sum()
        print(f"### {model_flag} wmape: {wmape}")

    @classmethod
    def compute_skew(cls, pdf: pd.DataFrame):
        model_flag = pdf.iloc[0]["model"]
        sum_y_pred = pdf["y_pred"].sum()
        sum_y = pdf["y"].sum()

        print(f"### {model_flag} skew: {sum_y_pred / sum_y}")

    @staticmethod
    def inner(pdf: pd.DataFrame):
        pred_date = pdf.iloc[0]["date"]
        model = pdf.iloc[0]["model"]
        print(f"### 模型: {model}, 预测日期: {pred_date}")

        pdf["y_pred"] = pdf["y_pred"].astype("float")
        interval = 7
        right_date = time_util.get_next_n_date(pred_date, interval)
        part_date_forecasting_df = pdf[pdf["date_pred"] < int(right_date)]
        del part_date_forecasting_df["date_pred"]

        temp_forecasting_df = part_date_forecasting_df.groupby(["mihome", "channel", "goods_id",
                                                                "date", "model"],
                                                               as_index=False).sum()
        return temp_forecasting_df[["mihome", "channel", "goods_id", "date", "y_pred"]]

    @classmethod
    def compute_metrics(cls, date):
        warnings.filterwarnings('ignore')

        test_df = pd.read_csv("../../data/testing.csv")
        prediction_df = pd.read_csv("../../data/prediciton.csv")

        lgb_base_df = test_df[(test_df["date"] == date) & (test_df["model"] == "lgb_single")]

        prediction_df = prediction_df.groupby(["model", "date"], as_index=False).apply(cls.inner)
        lgb_df = lgb_base_df[['mihome', 'channel', 'goods_id',
                              'y', 'date', 'model']].merge(prediction_df,
                                                           on=["mihome", "channel", "goods_id", "date"],
                                                           how="left")

        ma14_df = test_df[(test_df["date"] == date) & (test_df["model"] == "ma14")]

        print()
        # cls.compute_mape(lgb_base_df)
        cls.compute_skew(lgb_df)
        cls.compute_skew(ma14_df)

        print()
        # cls.compute_mape(lgb_base_df)
        cls.compute_mape(lgb_df)
        cls.compute_mape(ma14_df)

        print()
        # cls.compute_wmape(lgb_base_df)
        cls.compute_wmape(lgb_df)
        cls.compute_wmape(ma14_df)


def main():
    Metrics.compute_metrics(date=20221013)


if __name__ == "__main__":
    main()
