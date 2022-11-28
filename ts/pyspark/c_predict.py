# coding: utf-8
import argparse
import datetime
from typing import List

import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as sf

from src import config, time_util, workflow_util
from src.enum.env_field import EnvField
from src.model.train_and_predict import TrainAndPredict
from src.workflow_util import runtime


class PredictParallel(object):
    """
        预测
    """

    def __init__(self,
                 env: str,
                 target_col: str,
                 date_col: str,
                 main_index_cols: List[str],
                 predict_date: str,
                 ):
        self.__env = env

        if self.__env != EnvField.local:
            self.__spark = SparkSession.builder.enableHiveSupport().appName("data_processing").getOrCreate()
            self.__spark.sql('''set hive.exec.dynamic.partition.mode=nonstrict''')

        self.__target_col = target_col
        self.__date_col = date_col
        self.__main_index_cols = main_index_cols
        self.__predict_date = predict_date

        self.__test_and_model_df = self.__get_test_and_model_df()

    def __get_test_and_model_df(self):
        if self.__env == EnvField.local:

            test_pdf = pd.read_csv("../../data/template_feature.csv")
            test_pdf = test_pdf[(test_pdf["date"] == int(self.__predict_date)) &
                                (test_pdf["data_type"] == "test")]
            test_pdf["model_key"] = test_pdf["mihome"]

            model_pdf = pd.read_csv("../../data/model.csv")
            model_pdf = model_pdf[model_pdf["date"] == int(self.__predict_date)]

            test_pdf = pd.concat([test_pdf, model_pdf])

            return test_pdf
        else:
            test_df_sql = f""" SELECT * FROM {config.feature_database_dict[self.__env]} 
                       WHERE date == {self.__predict_date}
                         and data_type == "test"
                    """
            print(test_df_sql)
            test_df = self.__spark.sql(test_df_sql)

            model_df_sql = f""" SELECT model_key, model_str FROM {config.model_database_dict[self.__env]}
                            WHERE date == {self.__predict_date}
                        """
            print(model_df_sql)
            model_df = self.__spark.sql(model_df_sql)

            # 全部列名
            columns = test_df.columns + model_df.columns

            # 模型数据填充
            for column in test_df.columns:
                model_df = model_df.withColumn(column, sf.lit(""))

            # 测试数据填充
            test_df = test_df.withColumn("model_key", sf.col("mihome"))
            test_df = test_df.withColumn("model_str", sf.lit(""))

            test_df = test_df.select(columns)
            model_df = model_df.select(columns)

            return test_df.union(model_df)

    def run(self):

        prediction_result_database = config.prediction_result_database_dict[self.__env]
        print(f"### prediction_result_database: {prediction_result_database}")

        predict_function = PredictParallel.predict(self.__main_index_cols, self.__date_col)

        if self.__env == EnvField.local:
            test_and_model_df: pd.DataFrame = self.__test_and_model_df
            prediction_df = test_and_model_df.groupby(["model_key"]).apply(predict_function)
            prediction_df.to_csv(prediction_result_database, index=False, header=True)
        else:
            schema_str = ""
            for col in self.__main_index_cols:
                schema_str += col + " string,"
            schema_str += "y_pred double,"
            schema_str += "date_pred integer,"
            schema_str += f"{self.__date_col} integer,"
            schema_str += "model string"

            @sf.pandas_udf(schema_str, sf.PandasUDFType.GROUPED_MAP)
            def _train_predict_function(pdf):
                return predict_function(pdf)

            train_df: pyspark.sql.DataFrame = self.__test_and_model_df
            train_df.cache()
            train_df.show()
            prediction_df = train_df.groupby(["model_key"]).apply(_train_predict_function)
            print(prediction_df.schema)
            prediction_df.repartition(1).write.insertInto(prediction_result_database, overwrite=True)

    @staticmethod
    def predict(main_index_cols, date_col):
        def train_and_predict_inner(pdf: pd.DataFrame):
            pdf[main_index_cols] = pdf[main_index_cols].astype(np.str)
            model_key = pdf.iloc[0]["model_key"]
            print(f"### model_key: {model_key}")
            print(pdf)

            model_str = pdf[(~pdf["model_str"].isna()) &
                            (pdf["model_str"] != "")].iloc[0]["model_str"]
            print("### model_str: ", model_str[:200])

            test_pdf = pdf[(pdf["model_str"].isna()) |
                           (pdf["model_str"] == "")]
            print(f"### 数据条数: {len(test_pdf)}")
            test_pdf[config.float_features] = test_pdf[config.float_features].astype("float", errors="ignore")
            test_pdf[config.int_features] = test_pdf[config.int_features].astype("int")
            feature_pdf = workflow_util.reduce_mem_usage(df=test_pdf, verbose=True)
            del test_pdf

            train_and_predict = TrainAndPredict(date_col=date_col,
                                                feature_pdf=feature_pdf,
                                                model_features=config.model_features,
                                                cat_features=config.cat_features,
                                                str_flag=model_key)

            prediction_df = train_and_predict.predict(model_str)

            if prediction_df is not None:
                print(f"预测数据条数: {len(prediction_df)}")
                print(f"预测总销量: ", prediction_df["y_pred"].sum())
                prediction_df["predict_factor"] = prediction_df["qty_rolling_sum_1_30"].apply(lambda x:
                                                                                              1 if x > 0 else 0)
                prediction_df["y_pred"] = prediction_df["predict_factor"] * prediction_df["y_pred"]
                print(f"修正之后预测总销量: ", prediction_df["y_pred"].sum())

                single_day_prediction_df = train_and_predict.divide_into_single_day(prediction_df)

                single_day_prediction_df["model"] = "lgb_single"
                single_day_prediction_df[date_col] = single_day_prediction_df[date_col].astype("int")
                single_day_prediction_df["date_pred"] = single_day_prediction_df["date_pred"].astype("int")
                return single_day_prediction_df[main_index_cols + ["y_pred", "date_pred", date_col, "model"]]
            else:
                return pd.DataFrame(columns=main_index_cols + ["y_pred", "date_pred", date_col, "model"])

        return train_and_predict_inner


@runtime
def main():
    # 加载参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--predict_date', type=str)

    args_dict = parser.parse_args()

    env = args_dict.env  # 运行环境
    predict_date = args_dict.predict_date  # 预测日期

    print(f"env: {env}    predict_date: {predict_date}")

    predict_parallel = PredictParallel(env=env,
                                       target_col=config.target_col,
                                       main_index_cols=config.main_index_cols,
                                       date_col=config.date_col,
                                       predict_date=predict_date,
                                       )
    predict_parallel.run()


if __name__ == "__main__":
    main()
