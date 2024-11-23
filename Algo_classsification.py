import pandas as pd
import numpy as np

def classify_by_regularity(df, date_column, category_column, threshold=30):
    """
    Классификация транзакций по регулярности.

    :param df: DataFrame с транзакциями.
    :param date_column: Название столбца с датами.
    :param category_column: Название столбца с категориями.
    :param threshold: Порог для классификации регулярности в днях.
    :return: DataFrame с дополнительным столбцом 'regularity'.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    df['time_diff'] = df.groupby(category_column)[date_column].diff().dt.days
    df['regularity'] = np.where(df['time_diff'].mean() < threshold, 'Постоянные', 'Нерегулярные')
    return df


def classify_by_amount(df, amount_column, bins=None, labels=None):
    """
    Классификация транзакций по сумме.

    :param df: DataFrame с транзакциями.
    :param amount_column: Название столбца с суммой.
    :param bins: Границы для классификации.
    :param labels: Названия категорий.
    :return: DataFrame с дополнительным столбцом 'amount_category'.
    """
    if bins is None:
        bins = [0, 1000, 5000, float('inf')]
    if labels is None:
        labels = ['Мелкие', 'Средние', 'Крупные']

    df['amount_category'] = pd.cut(df[amount_column], bins=bins, labels=labels)
    return df

def classify_by_payment_method(df, payment_column, online_wallets=None):
    """
    Классификация транзакций по способу оплаты.

    :param df: DataFrame с транзакциями.
    :param payment_column: Название столбца со способом оплаты.
    :param online_wallets: Список онлайн-кошельков.
    :return: DataFrame с дополнительным столбцом 'payment_type'.
    """
    if online_wallets is None:
        online_wallets = ['ЮMoney', 'PayPal']

    df['payment_type'] = df[payment_column].apply(
        lambda x: 'Онлайн-кошельки' if x in online_wallets else x
    )
    return df

def classify_by_time_patterns(df, date_column, category_column, bins=None, labels=None):
    """
    Классификация транзакций по временным паттернам.

    :param df: DataFrame с транзакциями.
    :param date_column: Название столбца с датами.
    :param category_column: Название столбца с категориями.
    :param bins: Границы для классификации времени.
    :param labels: Названия категорий.
    :return: DataFrame с дополнительным столбцом 'time_pattern'.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    df['daily_frequency'] = df.groupby(category_column)[date_column].transform(
        lambda x: x.diff().dt.days.mean()
    )
    if bins is None:
        bins = [0, 1, 30, float('inf')]
    if labels is None:
        labels = ['Ежедневные', 'Ежемесячные', 'Спонтанные']

    df['time_pattern'] = pd.cut(df['daily_frequency'], bins=bins, labels=labels)
    return df


def classify_by_vendor(df, vendor_column, vendor_mapping):
    """
    Классификация транзакций по контрагентам.

    :param df: DataFrame с транзакциями.
    :param vendor_column: Название столбца с продавцами.
    :param vendor_mapping: Словарь соответствия продавцов категориям.
    :return: DataFrame с дополнительным столбцом 'vendor_category'.
    """
    df['vendor_category'] = df[vendor_column].map(vendor_mapping)
    return df


def classify_by_relative_amount(df, amount_column, percentiles=None, labels=None):
    """
    Классификация транзакций по относительным порогам (процентилям).

    :param df: DataFrame с транзакциями.
    :param amount_column: Название столбца с суммой.
    :param percentiles: Процентильные границы (например, [0.33, 0.66]).
    :param labels: Названия категорий.
    :return: DataFrame с дополнительным столбцом 'amount_category'.
    """
    if percentiles is None:
        percentiles = [0.05, 0.33]  
    if labels is None:
        labels = ['Мелкие', 'Средние', 'Крупные']

    bounds = [df[amount_column].quantile(q) for q in percentiles]
    bins = [-float('inf')] + bounds + [float('inf')]
    df['amount_category'] = pd.cut(df[amount_column], bins=bins, labels=labels)
    return df
