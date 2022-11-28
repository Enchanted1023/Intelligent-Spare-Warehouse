# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession

create_feature_table_sql = """
CREATE TABLE `info_algo_app`.`smart_transform_time_series_feature_test` (
  `mihome` STRING COMMENT '仓库',
  `channel` STRING COMMENT '渠道',
  `goods_id` STRING COMMENT '商品',
  `qty_original` STRING COMMENT '原始销量',
  `qty` STRING COMMENT '销量',
  `qty_agg_7` STRING COMMENT '未来7日真实销量之和(含当天)',
  `qty_rolling_sum_1_7` STRING COMMENT '-',
  `qty_rolling_mean_1_7` STRING COMMENT '-',
  `qty_rolling_std_1_7` STRING COMMENT '-',
  `qty_rolling_sum_1_14` STRING COMMENT '-',
  `qty_rolling_mean_1_14` STRING COMMENT '-',
  `qty_rolling_std_1_14` STRING COMMENT '-',
  `qty_rolling_sum_1_30` STRING COMMENT '-',
  `qty_rolling_mean_1_30` STRING COMMENT '-',
  `qty_rolling_std_1_30` STRING COMMENT '-',
  `qty_rolling_sum_7_7` STRING COMMENT '-',
  `qty_rolling_mean_7_7` STRING COMMENT '-',
  `qty_rolling_std_7_7` STRING COMMENT '-',
  `qty_rolling_sum_7_14` STRING COMMENT '-',
  `qty_rolling_mean_7_14` STRING COMMENT '-',
  `qty_rolling_std_7_14` STRING COMMENT '-',
  `qty_rolling_sum_7_30` STRING COMMENT '-',
  `qty_rolling_mean_7_30` STRING COMMENT '-',
  `qty_rolling_std_7_30` STRING COMMENT '-',
  `qty_rolling_sum_14_7` STRING COMMENT '-',
  `qty_rolling_mean_14_7` STRING COMMENT '-',
  `qty_rolling_std_14_7` STRING COMMENT '-',
  `qty_rolling_sum_14_14` STRING COMMENT '-',
  `qty_rolling_mean_14_14` STRING COMMENT '-',
  `qty_rolling_std_14_14` STRING COMMENT '-',
  `qty_rolling_sum_14_30` STRING COMMENT '-',
  `qty_rolling_mean_14_30` STRING COMMENT '-',
  `qty_rolling_std_14_30` STRING COMMENT '-',
  `qty_lag_1` STRING COMMENT '-',
  `qty_lag_2` STRING COMMENT '-',
  `qty_lag_3` STRING COMMENT '-',
  `qty_lag_4` STRING COMMENT '-',
  `qty_lag_5` STRING COMMENT '-',
  `qty_lag_6` STRING COMMENT '-',
  `qty_lag_7` STRING COMMENT '-',
  `qty_lag_8` STRING COMMENT '-',
  `qty_lag_9` STRING COMMENT '-',
  `qty_lag_10` STRING COMMENT '-',
  `qty_lag_11` STRING COMMENT '-',
  `qty_lag_12` STRING COMMENT '-',
  `qty_lag_13` STRING COMMENT '-',
  `qty_lag_14` STRING COMMENT '-',
  `qty_lag_15` STRING COMMENT '-',
  `qty_lag_16` STRING COMMENT '-',
  `qty_lag_17` STRING COMMENT '-',
  `qty_lag_18` STRING COMMENT '-',
  `qty_lag_19` STRING COMMENT '-',
  `qty_lag_20` STRING COMMENT '-',
  `qty_lag_21` STRING COMMENT '-',
  `qty_lag_22` STRING COMMENT '-',
  `qty_lag_23` STRING COMMENT '-',
  `qty_lag_24` STRING COMMENT '-',
  `qty_lag_25` STRING COMMENT '-',
  `qty_lag_26` STRING COMMENT '-',
  `qty_lag_27` STRING COMMENT '-',
  `qty_lag_28` STRING COMMENT '-',
  `qty_lag_29` STRING COMMENT '-',
  `qty_lag_30` STRING COMMENT '-',
  `qty_max` STRING COMMENT '-',
  `qty_min` STRING COMMENT '-',
  `qty_mean` STRING COMMENT '-',
  `qty_std` STRING COMMENT '-',
  `tm_day` STRING COMMENT '-',
  `tm_week` STRING COMMENT '-',
  `tm_month` STRING COMMENT '-',
  `tm_year` STRING COMMENT '-',
  `tm_week_of_month` STRING COMMENT '-',
  `tm_day_of_week` STRING COMMENT '-',
  `tm_weekend` STRING COMMENT '-',
  `factor_dayofweek_0` STRING COMMENT '-',
  `factor_dayofweek_1` STRING COMMENT '-',
  `factor_dayofweek_2` STRING COMMENT '-',
  `factor_dayofweek_3` STRING COMMENT '-',
  `factor_dayofweek_4` STRING COMMENT '-',
  `factor_dayofweek_5` STRING COMMENT '-',
  `factor_dayofweek_6` STRING COMMENT '-')
PARTITIONED BY (`date` STRING COMMENT '日期')
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
STORED AS
    INPUTFORMAT 'org.apache.hadoop.mapred.TextInputFormat'
    OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION 'hdfs://zjyprc-hadoop/user/s_info_algo_app/warehouse/info_algo_app/smart_transform_time_series_feature_test'
"""


create_forecasting_result_table_sql = """
CREATE TABLE `info_algo_app`.`smart_transform_forecasting_result_test` (
  `mihome` STRING COMMENT '仓库',
  `channel` STRING COMMENT '渠道',
  `goods_id` STRING COMMENT '商品',
  `y_pred` STRING COMMENT '预测值',
  `date_pred` STRING COMMENT '预测日期')
COMMENT '智能调拨销量预测结果表'
PARTITIONED BY (`date` STRING COMMENT '日期', 
                `model` STRING COMMENT '算法模型')
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
STORED AS
    INPUTFORMAT 'org.apache.hadoop.mapred.TextInputFormat'
    OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION 'hdfs://zjyprc-hadoop/user/s_info_algo_app/warehouse/info_algo_app/smart_transform_forecasting_result_test'

"""

drop_sql = "drop table info_algo_app.smart_transform_time_series_feature_test"


def main():
    # 生成spark session
    spark = SparkSession.builder.enableHiveSupport().appName("sql").getOrCreate()
    spark.sql('''set hive.exec.dynamic.partition.mode=nonstrict''')

    sql = """
     SELECT * FROM info_algo_app.smart_transform_time_series_feature_test 
                       WHERE date >= 20210618 
                         and date <= 20220926
                         and mihome = 14185
    """

    df = spark.sql(sql)
    df.show()
    df.repartition(1).write.csv("/user/s_info_algo_app/tmp_ted/test_smart_transform_df", mode="overwrite", header=True)


if __name__ == '__main__':
    main()
