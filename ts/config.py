# -*- coding: utf-8 -*-
from src.enum.env_field import EnvField

raw_database_dict = {
    EnvField.local: "../../data/template_raw_train_data.csv",
    EnvField.test: "hive_zjyprc_hadoop.info_algo_app.db_dwd_sales_data",
    EnvField.product: "hive_zjyprc_hadoop.info_algo_app.db_dwd_sales_data"
}

feature_database_dict = {
    EnvField.local: "../../data/feature.csv",
    EnvField.test: "hive_zjyprc_hadoop.info_algo_app.smart_transform_time_series_feature_test",
    EnvField.product: "hive_zjyprc_hadoop.info_algo_app.smart_transform_time_series_feature",
}

model_database_dict = {
    EnvField.local: "../../data/model.csv",
    EnvField.test: "hive_zjyprc_hadoop.info_algo_app.smart_transform_forecasting_model_test",
    EnvField.product: "hive_zjyprc_hadoop.info_algo_app.smart_transform_forecasting_model"
}

prediction_result_database_dict = {
    EnvField.local: "../../data/prediciton.csv",
    EnvField.test: "hive_zjyprc_hadoop.info_algo_app.smart_transform_forecasting_result_test",
    EnvField.product: "hive_zjyprc_hadoop.info_algo_app.smart_transform_forecasting_result"
}

testing_database_dict = {
    EnvField.test: "hive_zjyprc_hadoop.info_algo_app.smart_transform_testing_data_test",
    EnvField.product: "hive_zjyprc_hadoop.info_algo_app.smart_transform_testing_data",
}

# 训练数据所有特征
feature_columns = ['mihome', 'channel', 'goods_id', 'qty_original', 'qty', 'qty_agg_7',
                   'qty_rolling_sum_1_7', 'qty_rolling_mean_1_7', 'qty_rolling_std_1_7',
                   'qty_rolling_sum_1_14', 'qty_rolling_mean_1_14', 'qty_rolling_std_1_14',
                   'qty_rolling_sum_1_30', 'qty_rolling_mean_1_30', 'qty_rolling_std_1_30',
                   'qty_rolling_sum_7_7', 'qty_rolling_mean_7_7', 'qty_rolling_std_7_7',
                   'qty_rolling_sum_7_14', 'qty_rolling_mean_7_14', 'qty_rolling_std_7_14',
                   'qty_rolling_sum_7_30', 'qty_rolling_mean_7_30', 'qty_rolling_std_7_30',
                   'qty_rolling_sum_14_7', 'qty_rolling_mean_14_7', 'qty_rolling_std_14_7',
                   'qty_rolling_sum_14_14', 'qty_rolling_mean_14_14', 'qty_rolling_std_14_14',
                   'qty_rolling_sum_14_30', 'qty_rolling_mean_14_30', 'qty_rolling_std_14_30',
                   'qty_lag_1', 'qty_lag_2', 'qty_lag_3', 'qty_lag_4', 'qty_lag_5',
                   'qty_lag_6', 'qty_lag_7', 'qty_lag_8', 'qty_lag_9', 'qty_lag_10',
                   'qty_lag_11', 'qty_lag_12', 'qty_lag_13', 'qty_lag_14', 'qty_lag_15',
                   'qty_lag_16', 'qty_lag_17', 'qty_lag_18', 'qty_lag_19', 'qty_lag_20',
                   'qty_lag_21', 'qty_lag_22', 'qty_lag_23', 'qty_lag_24', 'qty_lag_25',
                   'qty_lag_26', 'qty_lag_27', 'qty_lag_28', 'qty_lag_29', 'qty_lag_30',
                   'tm_day', 'tm_week', 'tm_month', 'tm_year',
                   'tm_week_of_month', 'tm_day_of_week', 'tm_weekend',
                   "factor_dayofweek_0", "factor_dayofweek_1", "factor_dayofweek_2", "factor_dayofweek_3",
                   "factor_dayofweek_4", "factor_dayofweek_5", "factor_dayofweek_6",
                   "date"]

int_features = ['tm_day', 'tm_week', 'tm_month', 'tm_year', 'tm_week_of_month', 'tm_day_of_week', 'tm_weekend', ]

float_features = ['qty_lag_1', 'qty_lag_2', 'qty_lag_3', 'qty_lag_4', 'qty_lag_5',
                  'qty_lag_6', 'qty_lag_7', 'qty_lag_8', 'qty_lag_9', 'qty_lag_10',
                  'qty_lag_11', 'qty_lag_12', 'qty_lag_13', 'qty_lag_14', 'qty_lag_15',
                  'qty_lag_16', 'qty_lag_17', 'qty_lag_18', 'qty_lag_19', 'qty_lag_20',
                  'qty_lag_21', 'qty_lag_22', 'qty_lag_23', 'qty_lag_24', 'qty_lag_25',
                  'qty_lag_26', 'qty_lag_27', 'qty_lag_28', 'qty_lag_29', 'qty_lag_30',

                  'qty_rolling_sum_1_7',
                  'qty_rolling_sum_1_14',
                  'qty_rolling_sum_1_30',
                  'qty_rolling_sum_7_7',
                  'qty_rolling_sum_7_14',
                  'qty_rolling_sum_7_30',
                  'qty_rolling_sum_14_7',
                  'qty_rolling_sum_14_14',
                  'qty_rolling_sum_14_30',

                  'qty_rolling_mean_1_7', 'qty_rolling_std_1_7',
                  'qty_rolling_mean_1_14', 'qty_rolling_std_1_14',
                  'qty_rolling_mean_1_30', 'qty_rolling_std_1_30',
                  'qty_rolling_mean_7_7', 'qty_rolling_std_7_7',
                  'qty_rolling_mean_7_14', 'qty_rolling_std_7_14',
                  'qty_rolling_mean_7_30', 'qty_rolling_std_7_30',
                  'qty_rolling_mean_14_7', 'qty_rolling_std_14_7',
                  'qty_rolling_mean_14_14', 'qty_rolling_std_14_14',
                  'qty_rolling_mean_14_30', 'qty_rolling_std_14_30',
                  'factor_dayofweek_0', 'factor_dayofweek_1', 'factor_dayofweek_2', 'factor_dayofweek_3',
                  'factor_dayofweek_4', 'factor_dayofweek_5', 'factor_dayofweek_6']

cat_features = ['channel']

# 模型所用特征
model_features = ['channel',
                  'qty_rolling_sum_1_7', 'qty_rolling_mean_1_7', 'qty_rolling_std_1_7',
                  'qty_rolling_sum_1_14', 'qty_rolling_mean_1_14', 'qty_rolling_std_1_14',
                  'qty_rolling_sum_1_30', 'qty_rolling_mean_1_30', 'qty_rolling_std_1_30',
                  'qty_rolling_sum_7_7', 'qty_rolling_mean_7_7', 'qty_rolling_std_7_7',
                  'qty_rolling_sum_7_14', 'qty_rolling_mean_7_14', 'qty_rolling_std_7_14',
                  'qty_rolling_sum_7_30', 'qty_rolling_mean_7_30', 'qty_rolling_std_7_30',
                  'qty_rolling_sum_14_7', 'qty_rolling_mean_14_7', 'qty_rolling_std_14_7',
                  'qty_rolling_sum_14_14', 'qty_rolling_mean_14_14', 'qty_rolling_std_14_14',
                  'qty_rolling_sum_14_30', 'qty_rolling_mean_14_30', 'qty_rolling_std_14_30',
                  'qty_lag_1', 'qty_lag_2', 'qty_lag_3', 'qty_lag_4', 'qty_lag_5', 'qty_lag_6',
                  'qty_lag_7', 'qty_lag_8', 'qty_lag_9', 'qty_lag_10', 'qty_lag_11', 'qty_lag_12',
                  'qty_lag_13', 'qty_lag_14', 'qty_lag_15', 'qty_lag_16', 'qty_lag_17', 'qty_lag_18',
                  'qty_lag_19', 'qty_lag_20', 'qty_lag_21', 'qty_lag_22', 'qty_lag_23', 'qty_lag_24',
                  'qty_lag_25', 'qty_lag_26', 'qty_lag_27', 'qty_lag_28', 'qty_lag_29', 'qty_lag_30',
                  'tm_day', 'tm_week', 'tm_month', 'tm_year', 'tm_week_of_month', 'tm_day_of_week', 'tm_weekend',
                  'factor_dayofweek_0', 'factor_dayofweek_1', 'factor_dayofweek_2',
                  'factor_dayofweek_3', 'factor_dayofweek_4', 'factor_dayofweek_5',
                  'factor_dayofweek_6']

target_col = 'agg_qty_7'  # 销量
main_index_cols = ["mihome", "channel", "goods_id"]  # 主键列名
date_col = "date"  # 日期列名

mihome_list = [14185, 28105, 14183, 719, 4743, 27165, 6883, 457, 505, 38949,
               455, 14187, 463, 420, 14189, 348, 112, 432,
               359, 563]
