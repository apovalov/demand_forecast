from typing import Tuple
from datetime import date

import numpy as np
import pandas as pd


def week_missed_profits(
    df: pd.DataFrame,
    sales_col: str,
    forecast_col: str,
    date_col: str = "day",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Calculates the missed profits every week for the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to calculate the missed profits for.
        (Must contain columns "sku_id", "date", "price", "sales" and "forecast")
    sales_col : str
        The column with the actual sales.
    forecast_col : str
        The column with the forecasted sales.
    price_col : str, optional
        The column with the price, by default "price".

    Returns
    -------
    pd.DataFrame
        The DataFrame with the missed profits.
        (Contains columns "day", "revenue", "missed_profits")
    """

    df['missed_profits'] = np.maximum(df[forecast_col] - df[sales_col], 0) * df[price_col]

    # Добавление столбца для расчета выручки
    df['total_revenue'] = df[sales_col] * df[price_col]

    # Преобразование столбца даты в формат datetime и группировка по неделям
    df[date_col] = pd.to_datetime(df[date_col])
    weekly_data = df.groupby(pd.Grouper(key=date_col, freq='W')).agg(
        revenue=pd.NamedAgg(column='total_revenue', aggfunc='sum'),
        missed_profits=pd.NamedAgg(column='missed_profits', aggfunc='sum')
    ).reset_index()

    # Приведение типа столбца 'revenue' к int64
    weekly_data['revenue'] = weekly_data['revenue'].astype('int64')

    # Возвращаем результат с необходимыми столбцами
    return weekly_data[[date_col, 'revenue', 'missed_profits']]





def missed_profits_ci(
    df: pd.DataFrame,
    missed_profits_col: str,
    confidence_level: float = 0.95,
    n_bootstraps: int = 1000,
) -> Tuple[Tuple[float, Tuple[float, float]], Tuple[float, Tuple[float, float]]]:
    """
    Estimates the missed profits for the given DataFrame.
    Calculates average missed_profits per week and estimates
    the 95% confidence interval.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to calculate the missed_profits for.

    missed_profits_col : str
        The column with the missed_profits.

    confidence_level : float, optional
        The confidence level for the confidence interval, by default 0.95.

    n_bootstraps : int, optional
        The number of bootstrap samples to use for the confidence interval,
        by default 1000.

    Returns
    -------
    Tuple[Tuple[float, Tuple[float, float]], Tuple[float, Tuple[float, float]]]
        Returns a tuple of tuples, where the first tuple is the absolute average
        missed profits with its CI, and the second is the relative average missed
        profits with its CI.

    Example:
    -------
    ((1200000, (1100000, 1300000)), (0.5, (0.4, 0.6)))
    """
    bootstrap_samples = []
    for _ in range(n_bootstraps):
        sample = df[missed_profits_col].sample(n=len(df), replace=True)
        bootstrap_samples.append(sample.mean())

    lower_bound = np.percentile(bootstrap_samples, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(bootstrap_samples, (1 + confidence_level) / 2 * 100)
    mean_missed_profits = np.mean(bootstrap_samples)

    relative_missed_profits = mean_missed_profits / df['revenue'].mean()
    relative_ci = (lower_bound / df['revenue'].mean(), upper_bound / df['revenue'].mean())

    return ((mean_missed_profits, (lower_bound, upper_bound)), (relative_missed_profits, relative_ci))

