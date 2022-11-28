# -*- coding: utf-8 -*-
import argparse
import datetime
import math
import random
import warnings
from typing import List

import numpy as np
import pandas as pd
import pyspark.sql
from pyspark.sql import SparkSession, types
import pyspark.sql.functions as sf

from src import time_util, config
from src.enum.data_type_field import DataTypeField
from src.enum.env_field import EnvField
from src.workflow_util import runtime

warnings.filterwarnings('ignore')


class FeatureEngineering(object):

    def __init__(self,
                 env: str,
                 data_type: str,
                 target_col: str,
                 date_col: str,
                 main_index_cols: List[str],
                 start_date: str,
                 end_date: str
                 ):
        """
        数据预处理模块初始化

        :param env: 运行环境
        :param data_type: 数据处理类型
        :param target_col: 预测目标字段名
        :param main_index_cols: 主键列名
        :param start_date: 数据左时间
        :param end_date: 数据右时间

        """
        self.__env = env

        if self.__env != EnvField.local:
            self.__spark = SparkSession.builder.enableHiveSupport().appName("data_processing").getOrCreate()
            self.__spark.sql('''set hive.exec.dynamic.partition.mode=nonstrict''')
            self.__spark.sql('''set hive.exec.max.dynamic.partitions=10000''')

        self.__data_type = data_type
        self.__target_col = target_col
        self.__date_col = date_col
        self.__main_index_cols = main_index_cols
        self.__start_date = start_date
        self.__end_date = end_date
        self.__train_df = self.__get_train_df()

    def __get_train_df(self):
        # 左时间向前推60天以构造销量特征
        start_date = time_util.get_next_n_date(self.__start_date, -60)
        end_date = self.__end_date

        raw_database = config.raw_database_dict[self.__env]

        if self.__env == EnvField.local:
            train_df = pd.read_csv(raw_database)[["mihome", "channel", "goods_id", "day", "qty"]]
            train_df = train_df.astype(str)
            train_df["qty"] = train_df["qty"].astype(float)
            train_df = train_df[(train_df["day"] >= start_date) &
                                (train_df["day"] <= end_date) &
                                (train_df["goods_id"] == "25873")]

        else:
            mihome_sql = ", ".join([str(mihome) for mihome in config.mihome_list])
            raw_df_sql = f""" select * from {raw_database} 
                                where dt={time_util.get_today_str()} 
                                    and {self.__date_col}<={end_date} 
                                    and {self.__date_col} >= {start_date}
                                    and mihome in ({mihome_sql})
                            """
            print(raw_df_sql)
            raw_df = self.__spark.sql(raw_df_sql)

            train_df = raw_df.select("mihome", "channel", "goods_id", "day", "qty", "goods_type")

        return train_df

    def run(self):

        feature_engineer_function = FeatureEngineering.feature_engineer(
            main_index_cols=self.__main_index_cols,
            target_col=self.__target_col,
            date_col=self.__date_col,
            start_date=self.__start_date,
            end_date=self.__end_date,
            data_type=self.__data_type
        )

        feature_database = config.feature_database_dict[self.__env]

        if self.__env == EnvField.local:
            train_df: pd.DataFrame = self.__train_df
            feature_df = train_df.groupby(self.__main_index_cols).apply(feature_engineer_function)
            feature_df.to_csv(feature_database, index=False, header=True)
        else:
            schema_str = ""
            for col in config.feature_columns:
                if col in config.int_features + ["qty_original", "date"]:
                    schema_str += col + " int,"
                elif col in self.__main_index_cols:
                    schema_str += col + " string,"
                elif "qty" in col or "factor" in col:
                    schema_str += col + " double,"
            schema_str = schema_str[:-1]

            @sf.pandas_udf(schema_str, sf.PandasUDFType.GROUPED_MAP)
            def _feature_engineer(pdf):
                return feature_engineer_function(pdf)

            train_df: pyspark.sql.DataFrame = self.__train_df
            feature_df = train_df.groupby(self.__main_index_cols).apply(_feature_engineer)
            print(feature_df.schema)
            feature_df = feature_df.select(config.feature_columns)
            if self.__data_type == DataTypeField.train:
                feature_df = feature_df.withColumn("data_type", sf.lit(DataTypeField.train))
                feature_df.repartition(1).write.insertInto(feature_database, overwrite=True)
            elif self.__data_type == DataTypeField.test:
                feature_df = feature_df.withColumn("data_type", sf.lit(DataTypeField.test))
                feature_df.repartition(1).write.insertInto(feature_database, overwrite=True)

    @staticmethod
    def feature_engineer(main_index_cols: List[str], target_col: str, date_col: str, start_date: str, end_date: str,
                         data_type: str):
        def inner_feature_engineer(pdf: pd.DataFrame):

            pdf[main_index_cols] = pdf[main_index_cols].astype(str)
            pdf[target_col] = pdf[target_col].astype(float)

            today_str = time_util.get_today_str()
            pdf = pdf[pdf[date_col] < int(today_str)]  # 删除冗余数据

            if data_type == DataTypeField.test:
                # 昨天有销量的数据，作为今天的测试数据
                yesterday_str = time_util.get_next_n_date(today_str, -1)
                yesterday_pdf = pdf[pdf[date_col] == int(yesterday_str)]

                # 构造测试数据
                if len(yesterday_pdf) > 0:
                    yesterday_pdf[date_col] = int(today_str)
                    yesterday_pdf[target_col] = -1
                    pdf = pd.concat([pdf, yesterday_pdf])

            pdf = FeatureEngineering.preprocess(pdf=pdf, main_index_cols=main_index_cols, target_col=target_col)
            pdf = FeatureEngineering.feature_of_calender(pdf=pdf, main_index_cols=main_index_cols, date_col=date_col)
            pdf = FeatureEngineering.feature_of_time_series(pdf=pdf, main_index_cols=main_index_cols,
                                                            target_col=target_col, date_col=date_col)
            pdf = FeatureEngineering.filter_data(pdf=pdf, main_index_cols=main_index_cols,
                                                 target_col=target_col, date_col=date_col, data_type=data_type)

            pdf[main_index_cols] = pdf[main_index_cols].astype(str)
            pdf = pdf[(pdf[date_col] >= int(start_date)) &
                      (pdf[date_col] <= int(end_date))]

            if len(pdf) > 0:
                if date_col != "date":
                    pdf["date"] = pdf[date_col]

                return pdf[config.feature_columns]
            else:
                return pd.DataFrame(columns=config.feature_columns)

        return inner_feature_engineer

    @staticmethod
    def feature_of_time_series(pdf: pd.DataFrame, main_index_cols: List[str],
                               target_col: str, date_col: str) -> pd.DataFrame:
        """
        为单个item的时间序列构造序列特征

        :param pdf: 单个item的全部时间序列数据
        :param main_index_cols: 主键列名
        :param target_col: 目标列名
        :param date_col: 日期列名
        :return: 新增时间特征
        """
        print("序列特征")
        # 日志
        for index_col in main_index_cols:
            index_col_value = pdf.iloc[0][index_col]
            print(f"{index_col}: {index_col_value}")
        print()

        #  临时日期列名
        temp_date_time_col = f"temp_{date_col}_{random.Random().randint(0, 100)}"
        pdf[temp_date_time_col] = pd.to_datetime(pdf[date_col], format="%Y%m%d")

        date_list = pdf[temp_date_time_col].to_list()
        date_set = set(date_list)
        # 不能出现两个相同的日期
        assert len(date_set) == len(date_list)

        min_date = min(date_set)
        max_date = max(date_set)

        time_delta = max_date - min_date

        # 如果时间不连续，则缺失值填充0
        if time_delta.days + 1 != len(date_set):
            total_date_set = set([min_date + datetime.timedelta(i) for i in range(0, time_delta.days + 1)])

            miss_date_list = list(total_date_set - date_set)
            miss_len = len(miss_date_list)

            temp_sales_pdf = pd.DataFrame(data={
                date_col: [date_item.strftime("%Y%m%d") for date_item in miss_date_list],
                target_col: [0 for _ in range(miss_len)],
                temp_date_time_col: miss_date_list,
            })

            # Todo: check
            pdf = pd.concat([pdf, temp_sales_pdf], axis=0)

        del pdf[temp_date_time_col]

        pdf = pdf.sort_values(by=[date_col])

        # 特征1. 滚动窗口
        for d_shift in [1, 7, 14]:
            for d_window in [7, 14, 30]:
                pdf[f'{target_col}_rolling_sum_' + str(d_shift) + '_' + str(d_window)] = pdf[target_col].transform(
                    lambda x: x.shift(d_shift).rolling(d_window).sum())
                pdf[f'{target_col}_rolling_mean_' + str(d_shift) + '_' + str(d_window)] = pdf[target_col].transform(
                    lambda x: x.shift(d_shift).rolling(d_window).mean())
                pdf[f'{target_col}_rolling_std_' + str(d_shift) + '_' + str(d_window)] = pdf[target_col].transform(
                    lambda x: x.shift(d_shift).rolling(d_window).std())

        # 特征2. 前30天销量
        lag_days = [col for col in range(1, 31)]
        pdf = pdf.assign(**{
            '{}_lag_{}'.format(target_col, ld): pdf[target_col].transform(lambda x: x.shift(ld))
            for ld in lag_days
        })

        # 特征4. 统计前8周，星期的占比
        for col in ["factor_dayofweek_0", "factor_dayofweek_1", "factor_dayofweek_2", "factor_dayofweek_3",
                    "factor_dayofweek_4", "factor_dayofweek_5", "factor_dayofweek_6"]:

            def udf(sales_list):
                total = sum(sales_list)

                if total == 0:
                    return 0

                step_2_sum_dict = {}
                for step in range(7):
                    i = step
                    i_sum = 0
                    while i < len(sales_list):
                        i_sum += sales_list.to_list()[i]
                        i += 7

                    step_2_sum_dict[f"factor_dayofweek_{step}"] = i_sum / total

                return step_2_sum_dict[col]

            def get_distribution_dict(x):
                return x.shift(1).rolling(7 * 8).agg(udf)

            pdf[col] = pdf[target_col].transform(get_distribution_dict)

        # 特征5 整体信息
        pdf[f"{target_col}_max"] = pdf[target_col].transform(lambda x: x.shift(1).rolling(365).max())
        pdf[f"{target_col}_min"] = pdf[target_col].transform(lambda x: x.shift(1).rolling(365).min())
        pdf[f"{target_col}_mean"] = pdf[target_col].transform(lambda x: x.shift(1).rolling(365).mean())
        pdf[f"{target_col}_std"] = pdf[target_col].transform(lambda x: x.shift(1).rolling(365).std())

        pdf = pdf.sort_values(by=[date_col], ascending=False)
        # label 后7日汇总量
        pdf[f"{target_col}_agg_7"] = pdf[target_col].transform(lambda x: x.rolling(7).sum())

        for index_col in main_index_cols:
            index_col_value = pdf.iloc[0][index_col]
            pdf[index_col] = index_col_value

        pdf = pdf.sort_values(by=[date_col])

        return pdf

    @staticmethod
    def feature_of_calender(pdf: pd.DataFrame, main_index_cols: List[str], date_col: str) -> pd.DataFrame:
        """
        为单个item的时间序列构造日期特征

        :param pdf: 单个item的全部时间序列数据
        :param main_index_cols: 主键列名
        :param date_col: 日期列名
        :return: 新增时间特征
        """
        print("日期特征")
        if len(pdf) == 0:
            return pdf

        # 日志
        for index_col in main_index_cols:
            index_col_value = pdf.iloc[0][index_col]
            print(f"{index_col}: {index_col_value}")
        print()

        # 新增日期列
        temp_date_time_col = f"temp_{date_col}_{random.Random().randint(0, 100)}"
        pdf[temp_date_time_col] = pd.to_datetime(pdf[date_col], format="%Y%m%d")

        # Make some features from date
        pdf['tm_day'] = pdf[temp_date_time_col].dt.day.astype(np.int8)
        pdf['tm_week'] = pdf[temp_date_time_col].dt.week.astype(np.int8)
        pdf['tm_month'] = pdf[temp_date_time_col].dt.month.astype(np.int8)
        pdf['tm_year'] = pdf[temp_date_time_col].dt.year
        pdf['tm_year'] = (pdf['tm_year'] - pdf['tm_year'].min()).astype(np.int8)
        pdf['tm_week_of_month'] = pdf['tm_day'].apply(lambda x: math.ceil(x / 7)).astype(np.int8)
        pdf['tm_day_of_week'] = pdf[temp_date_time_col].dt.dayofweek.astype(np.int8)
        pdf['tm_weekend'] = (pdf['tm_day_of_week'] >= 5).astype(np.int8)

        del pdf[temp_date_time_col]

        return pdf

    @staticmethod
    def preprocess(pdf: pd.DataFrame, main_index_cols: List[str], target_col: str):
        """
        为单个item的时间序列做数据预处理
        1、iqr截断

        :param pdf: 单个item的全部时间序列数据
        :param main_index_cols: 主键列名
        :param target_col: 目标列名
        :return: 预处理之后的特征
        """
        pdf = FeatureEngineering.iqr(pdf, main_index_cols, target_col)
        return pdf

    @staticmethod
    def three_sigma(pdf: pd.DataFrame, main_index_cols: List[str], target_col: str) -> pd.DataFrame:
        """
        为单个item的时间序列做3sigma截断

        :param pdf: 单个item的全部时间序列数据
        :param main_index_cols: 主键列名
        :param target_col: 目标列名
        :return: 预处理之后的特征
        """
        print(f"3 sigma 截断处理")

        # 日志
        for index_col in main_index_cols:
            index_col_value = pdf.iloc[0][index_col]
            print(f"{index_col}: {index_col_value}")
        print()

        pdf[f"{target_col}_original"] = pdf[target_col]
        pdf["global_sales_std"] = pdf.groupby(main_index_cols)[target_col].transform("std")
        pdf["global_sales_mean"] = pdf.groupby(main_index_cols)[target_col].transform("mean")
        pdf["3_sigma"] = pdf["global_sales_mean"] + 3 * pdf["global_sales_std"]
        pdf[target_col] = pdf[[target_col, "3_sigma"]].apply(
            lambda x: x[target_col] if x[target_col] <= x["3_sigma"] else x["3_sigma"], axis=1)

        del pdf["global_sales_std"]
        del pdf["global_sales_mean"]
        del pdf["3_sigma"]

        return pdf

    @staticmethod
    def min_value_zero(pdf: pd.DataFrame, main_index_cols: List[str], target_col: str) -> pd.DataFrame:
        """
        为单个item的时间序列做最小0值截断

        :param pdf: 单个item的全部时间序列数据
        :param main_index_cols: 主键列名
        :param target_col: 目标列名
        :return: 预处理之后的数据
        """
        print(f"min value zero 截断处理")

        # 日志
        for index_col in main_index_cols:
            index_col_value = pdf.iloc[0][index_col]
            print(f"{index_col}: {index_col_value}")
        print()

        pdf[target_col] = pdf[target_col].apply(lambda x: max(x, 0))

        return pdf

    @staticmethod
    def iqr(pdf: pd.DataFrame, main_index_cols: List[str], target_col: str) -> pd.DataFrame:
        """
        为单个item的时间序列做irq截断

        :param pdf: 单个item的全部时间序列数据
        :param main_index_cols: 主键列名
        :param target_col: 目标列名
        :return: 预处理之后的特征
        """
        print(f"irq 截断处理")

        # 日志
        for index_col in main_index_cols:
            index_col_value = pdf.iloc[0][index_col]
            print(f"{index_col}: {index_col_value}")
        print()

        pdf[f"{target_col}_original"] = pdf[target_col]

        # 计算通过iqr过滤异常值后的均值
        def smooth_iqr(x):
            """

            :param x: [10,12,23]
            :return:
            """
            x = np.array(x)
            iqr = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) * 1.5
            upper_bound = np.quantile(x, 0.75) + iqr
            lower_bound = np.quantile(x, 0.25) - iqr
            # print("upper_bound:",upper_bound,",lower_bound:",lower_bound)
            return upper_bound

        ub = smooth_iqr(pdf[target_col].to_list())
        pdf[target_col] = pdf[target_col].apply(lambda x: min(x, ub))

        return pdf

    @staticmethod
    def filter_data(pdf: pd.DataFrame, main_index_cols: List[str],
                    target_col: str, date_col: str, data_type: str) -> pd.DataFrame:
        """
        为单个item的时间序列筛选

        :param pdf: 单个item的全部时间序列数据
        :param main_index_cols: 主键列名
        :param target_col: 目标列名
        :param date_col: 日期列名
        :param data_type: 数据处理类型
        :return: 新增时间特征
        """
        print("过滤")
        # 日志
        for index_col in main_index_cols:
            index_col_value = pdf.iloc[0][index_col]
            print(f"{index_col}: {index_col_value}")
        print()

        # 训练集，筛选前30天有销量的数据
        if data_type == DataTypeField.train:
            pdf = pdf[pdf["qty_rolling_sum_1_30"] > 0]
        # 测试集，筛选常规品
        elif data_type == DataTypeField.test:
            pdf = pdf.sort_values(by=date_col, ascending=True)
            pdf["goods_type"] = pdf["goods_type"].transform(lambda x: x.shift(1))
            pdf = pdf[pdf["goods_type"] == "regular"]

        return pdf


@runtime
def main():
    # 加载参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--data_type', type=str)
    parser.add_argument('--start_date', type=str)
    parser.add_argument('--end_date', type=str)

    args_dict = parser.parse_args()

    env = args_dict.env  # 运行环境
    data_type = args_dict.data_type  # 数据处理任务类型
    start_date = args_dict.start_date  # 数据左时间
    end_date = args_dict.end_date  # 数据右时间
    print(f"env: {env}; start_date: {start_date}; end_date: {end_date}")

    if start_date is None:
        start_date = "20200101"

    if end_date is None:
        end_date = time_util.get_today_str()

    target_col = 'qty'  # 销量
    main_index_cols = ["mihome", "channel", "goods_id"]  # 主键列名
    date_col = "day"

    feature_engineering = FeatureEngineering(env=env,
                                             data_type=data_type,
                                             target_col=target_col,
                                             main_index_cols=main_index_cols,
                                             date_col=date_col,
                                             start_date=start_date,
                                             end_date=end_date,
                                             )
    feature_engineering.run()


if __name__ == "__main__":
    main()
