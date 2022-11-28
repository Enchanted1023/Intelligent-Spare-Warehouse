# coding: utf-8
import argparse
from pyspark.sql import SparkSession

from src import config
from src.enum.env_field import EnvField
from src.workflow_util import runtime


class FlushResult(object):
    """
        写入预测结果
    """

    def __init__(self,
                 env: str,
                 predict_date: str,
                 ):
        self.__env = env

        if self.__env != EnvField.local:
            self.__spark = SparkSession.builder.enableHiveSupport().appName("data_processing").getOrCreate()
            self.__spark.sql('''set hive.exec.dynamic.partition.mode=nonstrict''')

        self.__predict_date = predict_date

    def run(self):
        prediction_result_database = config.prediction_result_database_dict[self.__env]
        print(f"### prediction_result_database: {prediction_result_database}")

        sql = f""" 
                SELECT 
                    int(goods_id),
                    int(mihome),
                    channel,
                    date AS day,
                    date_pred AS predict_day,
                    y_pred AS predict_cnt,
                    date AS dt,
                    model AS predict_method
                FROM
                    {prediction_result_database}
                where
                    date={self.__predict_date}
                """
        print(sql)
        prediction_result_sdf = self.__spark.sql(sql)
        prediction_result_sdf.repartition(1).write.insertInto("hive_zjyprc_hadoop.info_algo_app.dws_db_forecast_result",
                                                              overwrite=True)


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

    FlushResult(env=env,
                predict_date=predict_date,
                ).run()


if __name__ == "__main__":
    main()
