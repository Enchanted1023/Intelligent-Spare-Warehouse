# coding: utf-8
import gc
import warnings
from typing import List

import lightgbm as lgb
import pandas as pd

from src import time_util
from src.workflow_util import runtime

warnings.filterwarnings('ignore')


class TrainAndPredict(object):
    """
        non-recursive model
    """

    def __init__(self,
                 feature_pdf: pd.DataFrame = None,
                 target_col: str = None,
                 main_index_cols: List[str] = None,
                 date_col: str = None,
                 train_date: str = None,
                 predict_date: str = None,
                 model_features: List[str] = None,
                 cat_features: List[str] = None,
                 str_flag: str = None):
        self.__feature_pdf = feature_pdf
        self.__target_col = target_col
        self.__main_index_cols = main_index_cols
        self.__date_col = date_col
        self.__train_date = train_date  # 训练日期
        if train_date is not None:
            self.__final_train_date = time_util.get_next_n_date(train_date, -7)  # 全部训练集右时间 & 验证集天
            self.__valid_train_date = time_util.get_next_n_date(self.__final_train_date, -7)  # 交叉验证训练集右时间
        self.__predict_date = predict_date  # 测试集日期
        self.__str_flag = str_flag

        self.__model_features = model_features
        self.__cat_features = cat_features

        self.__lgb_params = {
            'boosting_type': 'gbdt',
            'objective': 'mae',
            'metric': 'mae',
            'subsample': 0.8,
            'subsample_freq': 1,
            'learning_rate': 0.01,
            'num_leaves': 512,
            'min_data_in_leaf': 32,
            'feature_fraction': 0.8,
            'max_bin': 128,
            'n_estimators': 5000,
            'boost_from_average': True,
            'verbose': 1,
            'seed': 0,
            "early_stopping_round": 100,
        }

    def train(self):
        """
        Train Model
        """

        print(f"### Train ### str_flag: {self.__str_flag}, 训练日期: {self.__train_date}")

        """ ############################### 1. 准备基础数据   ##################################"""
        feature_df = self.__feature_pdf
        feature_df[self.__target_col] = feature_df[self.__target_col].astype("float", errors="ignore")
        feature_df[self.__cat_features] = feature_df[self.__cat_features].astype('category')
        print(f"数据条数: {len(feature_df)}")

        print(self.__model_features)

        # 时间切割 训练集 和 验证集
        train_mask = (feature_df[self.__date_col] <= self.__valid_train_date)
        validate_mask = (feature_df[self.__date_col] == self.__final_train_date)
        final_train_mask = (feature_df[self.__date_col] <= self.__final_train_date)

        print(f"\n训练集右时间: {self.__valid_train_date}\n"
              f"验证集时间:{self.__final_train_date}\n"
              f"全部训练集右时间: {self.__final_train_date}\n"
              )

        """ ############################### 2. 交叉验证选择最优超参   ##################################"""
        train_df = feature_df[train_mask]
        train_df = train_df.dropna(subset=[self.__target_col])

        print(f"训练集条数: {len(train_df)}")
        train_data = lgb.Dataset(train_df[self.__model_features], label=train_df[self.__target_col])

        validate_df = feature_df[validate_mask]
        validate_df = validate_df.dropna(subset=[self.__target_col])
        print(f"验证集条数: {len(validate_df)}")
        valid_data = lgb.Dataset(validate_df[self.__model_features], label=validate_df[self.__target_col])

        model_lgb = lgb.train(self.__lgb_params,
                              train_data,
                              valid_sets=[train_data, valid_data],
                              verbose_eval=100
                              )

        """ ############################### 3. 根据最优超参，用全部训练集训练模型   ##################################"""
        bast_iteration = model_lgb.best_iteration
        del model_lgb, train_mask, train_df, train_data, validate_mask, validate_df, valid_data

        self.__lgb_params["n_estimators"] = bast_iteration
        self.__lgb_params.pop("early_stopping_round")

        final_train_df = feature_df[final_train_mask]
        final_train_df = final_train_df.dropna(subset=[self.__target_col])
        print(f"最终训练集条数: {len(final_train_df)}")
        final_train_data = lgb.Dataset(final_train_df[self.__model_features], label=final_train_df[self.__target_col])

        model_lgb = lgb.train(self.__lgb_params,
                              final_train_data,
                              valid_sets=[final_train_data],
                              verbose_eval=100
                              )

        del feature_df, final_train_mask, final_train_df, final_train_data
        gc.collect()
        print(f"训练结束")

        return model_lgb

    def predict(self, model_str):
        """
        Test
        """

        print(f"### Predict ### str_flag: {self.__str_flag}")

        """ ############################### 1. 准备基础数据   ##################################"""
        feature_df = self.__feature_pdf
        feature_df[self.__cat_features] = feature_df[self.__cat_features].astype('category')
        print(f"测试数据条数: {len(feature_df)}")

        print(self.__model_features)

        # 测试集
        test_df = feature_df
        print(f"测试集: {test_df.shape}")

        if len(test_df) == 0:
            print(f"{self.__str_flag} 无测试集, 不进行预测")
            return

        """ ############################### 2. 加载模型并预测   ##################################"""
        m_lgb = lgb.Booster(model_str=model_str)

        indices = test_df.index.tolist()
        prediction = pd.DataFrame({'y_pred': m_lgb.predict(test_df[self.__model_features])})
        prediction.index = indices

        prediction_df = pd.concat([test_df, prediction], axis=1)
        print(f"预测结果: {prediction_df.shape}")

        del feature_df, m_lgb, test_df
        gc.collect()

        return prediction_df

    def evaluate(self, prediction_df):
        """
        测试集测评结果，用于开发阶段

        :param prediction_df: 预测结果
        """
        prediction_df = prediction_df[(prediction_df[self.__target_col] > 0)]
        prediction_df = prediction_df.dropna(subset=["qty_rolling_mean_1_14"])

        prediction_df["y_pred"] = round(prediction_df["y_pred"])

        print()
        true_sum = prediction_df[self.__target_col].sum()
        predict_sum = prediction_df["y_pred"].sum()

        print(f"测评数据条数:{len(prediction_df)}")
        print(f"true_sum: {true_sum}")
        print(f"predict_sum: {predict_sum}")

        prediction_df["abs_err"] = abs(prediction_df[self.__target_col] - round(prediction_df["y_pred"]))
        prediction_df["mape"] = prediction_df["abs_err"] / prediction_df[self.__target_col]
        print("MAPE", prediction_df["mape"].mean())
        print("wMAPE", prediction_df["abs_err"].sum() / true_sum)

        print()
        true_sum = prediction_df[self.__target_col].sum()
        predict_sum = prediction_df["qty_rolling_mean_1_14"].sum() * 7

        print(f"测评数据条数:{len(prediction_df)}")
        print(f"true_sum: {true_sum}")
        print(f"predict_sum: {predict_sum}")

        prediction_df["naive_abs_err"] = abs(
            prediction_df[self.__target_col] - round(prediction_df["qty_rolling_mean_1_14"] * 7))
        prediction_df["naive_mape"] = prediction_df["naive_abs_err"] / prediction_df[self.__target_col]
        print("naive_MAPE", prediction_df["naive_mape"].mean())
        print("naive_wMAPE", prediction_df["naive_abs_err"].sum() / true_sum)

        gc.collect()

    def divide_into_single_day(self, prediction_df):
        single_day_predict_pdf_list = []
        for _, row in prediction_df.iterrows():
            mihome = row["mihome"]
            channel = row["channel"]
            goods_id = row["goods_id"]
            y_pred = row["y_pred"]
            date_pred = row[self.__date_col]
            factor_dayofweek_0 = row["factor_dayofweek_0"]
            factor_dayofweek_1 = row["factor_dayofweek_1"]
            factor_dayofweek_2 = row["factor_dayofweek_2"]
            factor_dayofweek_3 = row["factor_dayofweek_3"]
            factor_dayofweek_4 = row["factor_dayofweek_4"]
            factor_dayofweek_5 = row["factor_dayofweek_5"]
            factor_dayofweek_6 = row["factor_dayofweek_6"]

            single_day_factor_list = [factor_dayofweek_6, factor_dayofweek_5, factor_dayofweek_4, factor_dayofweek_3,
                                      factor_dayofweek_2, factor_dayofweek_1, factor_dayofweek_0]

            try:
                assert abs(sum(single_day_factor_list) - 1) <= 0.00001
            except Exception as e:
                single_day_factor_list = [1 / 7 for _ in range(7)]

            single_day_pred = []
            for factor in single_day_factor_list:
                single_day_pred.append(y_pred * factor)

            single_day_pred = 3 * single_day_pred

            date_list = []
            for i in range(len(single_day_pred)):
                date = time_util.get_next_n_date(str(date_pred), i)
                date_list.append(date)

            single_day_predict_pdf = pd.DataFrame(data={
                "date_pred": date_list,
                "y_pred": single_day_pred
            })
            single_day_predict_pdf[self.__date_col] = date_pred
            single_day_predict_pdf["mihome"] = mihome
            single_day_predict_pdf["channel"] = channel
            single_day_predict_pdf["goods_id"] = goods_id
            single_day_predict_pdf_list.append(single_day_predict_pdf)
        return pd.concat(single_day_predict_pdf_list)


@runtime
def test():
    target_col = "qty_agg_7"  # 销量
    main_index_cols = ["mihome", "channel", "goods_id"]  # 主键列名
    date_col = "date"
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

    cat_features = ['channel']
    feature_pdf = pd.read_csv("../../data/template_feature.csv")
    feature_pdf[date_col] = feature_pdf[date_col].astype(str)

    model = TrainAndPredict(feature_pdf=feature_pdf,
                            target_col=target_col,
                            main_index_cols=main_index_cols,
                            date_col=date_col,
                            train_date="20220921",
                            predict_date="20220921",
                            model_features=model_features,
                            str_flag="563",
                            cat_features=cat_features
                            )
    model.train()


if __name__ == "__main__":
    test()
