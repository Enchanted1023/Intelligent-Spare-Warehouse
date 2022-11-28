# -*- coding: utf-8 -*-
import random
import numpy as np
import datetime

import pandas as pd


def runtime(func):
    """
    统计函数运行时间(s)

    :param func: 待运行函数
    :return:
    """
    def get_time(*args, **kwargs):
        start_time = datetime.datetime.now()
        rv = func(
            *args, **kwargs)
        end_time = datetime.datetime.now()
        cost_time = end_time - start_time
        print(func.__globals__["__name__"] + ": %f秒" % cost_time.total_seconds())

        return rv, cost_time.total_seconds()

    return get_time


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


# Seeder
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def reduce_mem_usage(df, verbose=False):
    """
    # Memory Reducer
    # :df pandas dataframe to reduce size             # type: pd.DataFrame()
    # :verbose                                        # type: bool

    :param df:
    :param verbose:
    :return:
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


def merge_by_concat(df1, df2, merge_on):
    """
    Merging by concat to not lose dtypes

    :param df1:
    :param df2:
    :param merge_on:
    :return:
    """
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1
