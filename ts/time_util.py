import datetime

hyphen_time_formation = "%Y-%m-%d"
time_formation = "%Y%m%d"
socket_time_formation = '%Y-%m-%dT%H:%M:%S'


def get_today_str() -> str:
    now_date = datetime.datetime.now()
    return now_date.strftime(time_formation)


def str2datetime(time_str: str) -> datetime:
    return datetime.datetime.strptime(time_str, time_formation)


def timestamp2str(timestamp: float) -> str:
    return datetime.datetime.fromtimestamp(timestamp).strftime(time_formation)


def socket_time_transform(socket_time_str: str) -> str:
    if isinstance(socket_time_str, int):
        return timestamp2str(datetime.datetime.fromtimestamp(socket_time_str / 1000).timestamp())

    # 零时区
    t = datetime.datetime.strptime(socket_time_str[:-9], socket_time_formation)
    # 东八区
    t += datetime.timedelta(hours=8)
    return timestamp2str(t.timestamp())


def get_next_n_date(date_str: str, n: int):
    """
    得到n天后的日期

    :param date_str: 当前日期
    :param n: n天后
    :return: n天后的日期字符串
    """
    date = datetime.datetime.strptime(str(date_str), time_formation)
    next_date = date + datetime.timedelta(days=n)
    next_date.strftime(time_formation)
    return next_date.strftime(time_formation)


def get_next_n_date_with_formation(date_str: str, n: int, out_formation: str):
    """
    得到n天后的日期

    :param date_str: 当前日期
    :param n: n天后
    :param out_formation: 格式
    :return: n天后的日期字符串
    """
    date = datetime.datetime.strptime(date_str, time_formation)
    next_date = date + datetime.timedelta(days=n)
    return next_date.strftime(out_formation)


def get_next_n_date_with_in_out_formation(date_str: str, n: int, in_out_formation: str):
    """
    得到n天后的日期

    :param date_str: 当前日期
    :param n: n天后
    :param in_out_formation: 格式
    :return: n天后的日期字符串
    """
    date = datetime.datetime.strptime(date_str, in_out_formation)
    next_date = date + datetime.timedelta(days=n)
    return next_date.strftime(in_out_formation)


def get_next_date(date_str: str):
    """
    得到1天后的日期

    :param date_str: 当前日期
    :return: 1天后的日期
    """
    return get_next_n_date(date_str, 1)


def get_diff_days(start_date_str: str, end_date_str: str):
    """
    得到日期之间相差的天数
    :param start_date_str: 开始日期
    :param end_date_str: 结束日期
    :return: 开始日期和结束日期之间相差的天数
    """

    # 开始日期 <= 结束日期
    # assert start_date_str <= end_date_str
    start_date = datetime.datetime.strptime(start_date_str, time_formation)
    end_date = datetime.datetime.strptime(end_date_str, time_formation)

    # time_delta
    td = end_date - start_date
    return td.days


def get_week_of_date_str(date_str):
    """
    得到 年+第几周
    :param date_str:
    :return: 年+第几周
    """
    date = datetime.datetime.strptime(date_str, time_formation)
    # 年、第几周、周几
    year, week, weekday = date.isocalendar()
    return str(year) + "_" + str(week)


def get_week_of_date_str_hyphen(date_str):
    """
    得到 年+第几周
    :param date_str:
    :return: 年+第几周
    """
    date = datetime.datetime.strptime(date_str, hyphen_time_formation)
    # 年、第几周、周几
    year, week, weekday = date.isocalendar()
    return str(year) + "_" + str(week)


def get_month_of_date_str_hyphen(date_str):
    """
    得到 年+月
    :param date_str:
    :return: 年+第几周
    """
    date = datetime.datetime.strptime(date_str, hyphen_time_formation)
    return str(date.year) + "_" + str(date.month)


def get_week_of_today():
    return get_week_of_date_str(get_today_str())


if __name__ == "__main__":
    print(get_week_of_date_str("20220101"))
    print(get_week_of_today())
