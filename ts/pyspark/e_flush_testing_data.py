# coding: utf-8
import argparse
import warnings

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as sf

from src import config, time_util
from src.enum.env_field import EnvField
from src.workflow_util import runtime

warnings.filterwarnings('ignore')


class FlushTestingData(object):
    """
        写入测试集数据
    """

    def __init__(self,
                 env: str,
                 start_date: str,
                 end_date: str,
                 interval: int
                 ):
        self.__env = env

        if self.__env != EnvField.local:
            self.__spark = SparkSession.builder.enableHiveSupport().appName("data_processing").getOrCreate()
            self.__spark.sql('''set hive.exec.dynamic.partition.mode=nonstrict''')

        self.__start_date = start_date
        self.__end_date = end_date
        self.__interval = interval

    def run(self):
        test_ma14_sdf = self.get_ma14_predict_sdf()

        # 计算14日均销结果
        print("### 1. 计算14日均销预测结果")
        test_ma14_sdf = test_ma14_sdf.withColumn("ma14", test_ma14_sdf["qty_rolling_mean_1_14"] * self.__interval)

        # 14日均销预测值
        ma_sdf = test_ma14_sdf.select(["mihome", "channel", "goods_id", "ma14", "date"])
        ma_sdf = ma_sdf.withColumn("model", sf.lit("ma14"))
        ma_sdf = ma_sdf.withColumn("y_pred", ma_sdf["ma14"])
        ma_sdf = ma_sdf.select("mihome", "channel", "goods_id", "y_pred", "date", "model")

        # 模型预测值
        print("### 2. 提取模型预测值")
        forecasting_sdf = self.get_model_forecasting_sdf()
        testing_sdf = forecasting_sdf.unionAll(ma_sdf)

        # 计算真实值
        print("### 3. 提取真实值")
        true_label_sdf = self.get_y_sdf()
        true_label_sdf = true_label_sdf.select(["mihome", "channel", "goods_id", "y", "date"])

        testing_sdf = testing_sdf.join(true_label_sdf,
                                       on=["mihome", "channel", "goods_id", "date"],
                                       how="left")\
            .select("mihome", "channel", "goods_id", "y", "y_pred", "date", "model")
        testing_sdf.repartition(1).write.insertInto(config.testing_database_dict[self.__env],
                                                    overwrite=True)

    def get_ma14_predict_sdf(self):
        """
            从特征表中，提取测试集14日均销数据
        :return:  mihome, channel, goods_id, qty_rolling_mean_1_14, date
        """
        sql = f"""select mihome, channel, goods_id, qty_rolling_mean_1_14, date 
                  from {config.feature_database_dict[self.__env]} 
                  where date >= {self.__start_date} 
                     and date <= {self.__end_date}
                     and data_type == "test"
                """
        print(sql)
        feature_sdf = self.__spark.sql(sql)

        return feature_sdf

    def get_model_forecasting_sdf(self):
        """
            得到模型预测结果
        :return:
        """
        sql = f"""
                select mihome, channel, goods_id, y_pred, date_pred, date, model 
                  from {config.prediction_result_database_dict[self.__env]} 
                  where date >= {self.__start_date} 
                     and date <= {self.__end_date}
                """
        model_forecasting_sdf = self.__spark.sql(sql)

        def construct_forecasting_data(interval):

            schema_str = "mihome string, channel string, goods_id string, y_pred float, date integer, model string"

            @sf.pandas_udf(schema_str, sf.PandasUDFType.GROUPED_MAP)
            def inner(pdf: pd.DataFrame):
                pred_date = pdf.iloc[0]["date"]
                model = pdf.iloc[0]["model"]
                print(f"### 模型: {model}, 预测日期: {pred_date}")

                pdf["y_pred"] = pdf["y_pred"].astype("float")

                right_date = time_util.get_next_n_date(pred_date, interval)
                part_date_forecasting_df = pdf[pdf["date_pred"] < int(right_date)]
                del part_date_forecasting_df["date_pred"]

                temp_forecasting_df = part_date_forecasting_df.groupby(["mihome", "channel", "goods_id",
                                                                        "date", "model"],
                                                                       as_index=False).sum()
                print(temp_forecasting_df.dtypes)
                return temp_forecasting_df[["mihome", "channel", "goods_id", "y_pred", "date", "model"]]

            return inner

        forecasting_sdf = model_forecasting_sdf.groupby(["model",
                                                         "date"]).apply(construct_forecasting_data(self.__interval))

        return forecasting_sdf

    def get_y_sdf(self):
        """
            从原始表中，提取原始销量数据，构造true label
        :return:  mihome, channel, goods_id, qty_rolling_mean_1_14, date
        """
        raw_database = config.raw_database_dict[self.__env]

        mihome_sql = ", ".join([str(mihome) for mihome in config.mihome_list])
        raw_df_sql = f"""   select string(mihome), string(channel), string(goods_id), day as date, qty 
                            from {raw_database} 
                            where dt={time_util.get_today_str()} 
                                and day <= {time_util.get_next_n_date(self.__end_date, 7)} 
                                and day >= {self.__start_date}
                                and mihome in ({mihome_sql})
                                    """
        print(raw_df_sql)
        raw_sales_sdf = self.__spark.sql(raw_df_sql)

        schema_str = "mihome string, channel string, goods_id string, y float, date integer"

        @sf.pandas_udf(schema_str, sf.PandasUDFType.GROUPED_MAP)
        def construct_y(pdf: pd.DataFrame):

            pdf = pdf.sort_values(by=["date"], ascending=False)
            # label 后7日汇总量
            pdf["y"] = pdf["qty"].transform(lambda x: x.rolling(7).sum())

            pdf = pdf.dropna(subset=["y"])
            return pdf[["mihome", "channel", "goods_id", "y", "date"]]

        y_sdf = raw_sales_sdf.groupby(["mihome", "channel", "goods_id"]).apply(construct_y)

        return y_sdf


@runtime
def main():
    # 加载参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--start_date', type=str)
    parser.add_argument('--end_date', type=str)
    parser.add_argument('--interval', type=str)

    args_dict = parser.parse_args()

    env = args_dict.env  # 运行环境
    start_date = args_dict.start_date  # 开始日期
    end_date = args_dict.end_date  # 结束日期
    interval = int(args_dict.interval)  # 评估区间

    print(f"### env: {env}")
    print(f"### start_date: {start_date}")
    print(f"### end_date: {end_date}")
    print(f"### interval: {interval}")

    FlushTestingData(env=env,
                     start_date=start_date,
                     end_date=end_date,
                     interval=interval,
                     ).run()


if __name__ == "__main__":
    main()
