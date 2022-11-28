# coding: utf-8
import argparse
import datetime
from typing import List

import lightgbm as lgb

import pandas as pd
import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as sf

from src import config, time_util, workflow_util
from src.enum.env_field import EnvField
from src.model.train_and_predict import TrainAndPredict
from src.workflow_util import runtime


class TrainParallel(object):
    """
        训练&模型存档
    """

    def __init__(self,
                 env: str,
                 target_col: str,
                 date_col: str,
                 main_index_cols: List[str],
                 train_date: str,
                 ):
        self.__env = env

        if self.__env != EnvField.local:
            self.__spark = SparkSession.builder.enableHiveSupport().appName("data_processing").getOrCreate()
            self.__spark.sql('''set hive.exec.dynamic.partition.mode=nonstrict''')

        self.__target_col = target_col
        self.__date_col = date_col
        self.__main_index_cols = main_index_cols
        self.__train_date = train_date

        self.__train_df = self.__get_train_df()

    def __get_train_df(self):
        if self.__env == EnvField.local:
            time_delta = datetime.timedelta(int(31 * 24))
            start_datetime = datetime.datetime.strptime(self.__train_date, "%Y%m%d") - time_delta
            start_date = start_datetime.strftime("%Y%m%d")

            train_pdf = pd.read_csv("../../data/template_feature.csv")
            train_pdf = train_pdf[(train_pdf["date"] >= int(start_date)) &
                                  (train_pdf["date"] <= int(self.__train_date)) &
                                  (train_pdf["qty_rolling_sum_1_30"] > 0) &
                                  (~train_pdf["qty_rolling_sum_14_30"].isna())]
            return train_pdf
        else:
            mihome_list = ", ".join([str(mihome) for mihome in config.mihome_list])

            time_delta = datetime.timedelta(int(31 * 24))
            start_datetime = datetime.datetime.strptime(self.__train_date, "%Y%m%d") - time_delta
            start_date = start_datetime.strftime("%Y%m%d")

            sql = f""" SELECT * FROM {config.feature_database_dict[self.__env]} 
                       WHERE date >= {start_date} 
                         and date <= {self.__train_date}
                         and mihome in ({mihome_list})
                         and qty_rolling_sum_1_30 > 0
                         and data_type = "train"
                    """
            print(sql)
            return self.__spark.sql(sql)

    def run(self):

        model_database = config.model_database_dict[self.__env]
        print(f"### model_database:{model_database}")

        train_predict_function = TrainParallel.train_and_save_model(self.__train_date,
                                                                    self.__date_col)

        if self.__env == EnvField.local:
            train_df: pd.DataFrame = self.__train_df
            model_df = train_df.groupby(["mihome"]).apply(train_predict_function)
            model_df.to_csv(model_database, index=False, header=True)
        else:
            schema_str = f"model_str string, {self.__date_col} integer, model_key string"
            print(f"### model schema: {schema_str}")

            @sf.pandas_udf(schema_str, sf.PandasUDFType.GROUPED_MAP)
            def _train_predict_function(pdf):
                return train_predict_function(pdf)

            train_df: pyspark.sql.DataFrame = self.__train_df
            model_df = train_df.groupby(["mihome"]).apply(_train_predict_function)
            model_df.repartition(1).write.insertInto(model_database, overwrite=True)

    @staticmethod
    def train_and_save_model(train_date, date_col):
        def train_and_save_model_inner(pdf: pd.DataFrame):
            mihome = pdf.iloc[0]["mihome"]
            print(f"mihome: {mihome}")

            target_col = "qty_agg_7"  # 销量
            main_index_cols = ["mihome", "channel", "goods_id"]  # 主键列名

            print(f"数据条数: {len(pdf)}")
            pdf[config.float_features] = pdf[config.float_features].astype("float", errors="ignore")
            pdf[config.int_features] = pdf[config.int_features].astype("int")
            feature_pdf = workflow_util.reduce_mem_usage(df=pdf, verbose=True)
            del pdf

            feature_pdf[date_col] = feature_pdf[date_col].astype(str)

            train_and_predict = TrainAndPredict(feature_pdf=feature_pdf,
                                                target_col=target_col,
                                                main_index_cols=main_index_cols,
                                                date_col=date_col,
                                                train_date=train_date,
                                                model_features=config.model_features,
                                                cat_features=config.cat_features,
                                                str_flag=mihome)
            model_lgb: lgb.Booster = train_and_predict.train()

            if model_lgb is not None:
                model_str = model_lgb.model_to_string()
                return pd.DataFrame(data={"model_str": [model_str],
                                          date_col: [int(train_date)],
                                          "model_key": [mihome],
                                          })
            else:
                return pd.DataFrame(columns=["model_str", date_col, "model_key"])

        return train_and_save_model_inner


@runtime
def main():
    # 加载参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--train_date', type=str)

    args_dict = parser.parse_args()

    env = args_dict.env  # 运行环境
    train_date = args_dict.train_date  # 训练日期

    print(f"env: {env}    train_date: {train_date}")

    train_and_predict_parallel = TrainParallel(env=env,
                                               target_col=config.target_col,
                                               main_index_cols=config.main_index_cols,
                                               date_col=config.date_col,
                                               train_date=train_date,
                                               )
    train_and_predict_parallel.run()


if __name__ == "__main__":
    main()
