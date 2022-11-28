# -*- coding: utf-8 -*-
from src import config
from src.pyspark.b_train import TrainParallel
from src.pyspark.c_predict import PredictParallel
from src.test.compute_metric import Metrics

env = "local"  # 运行环境
train_date = "20221002"  # 训练日期

print(f"env: {env}    train_date: {train_date}")

# 1. 训练
TrainParallel(env=env,
              target_col=config.target_col,
              main_index_cols=config.main_index_cols,
              date_col=config.date_col,
              train_date=train_date,
              ).run()

# 2. 预测
predict_date = train_date  # 预测日期
PredictParallel(env=env,
                target_col=config.target_col,
                main_index_cols=config.main_index_cols,
                date_col=config.date_col,
                predict_date=predict_date,
                ).run()

# 3. 评估
Metrics.compute_metrics(date=int(train_date))
