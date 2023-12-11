import pandas as pd
from typing import Dict, Tuple, Optional
from os.path import dirname, join
current_dir = dirname(__file__)
file_path = join(current_dir, "../data/sales.csv")

print(current_dir)


def add_targets(df: pd.DataFrame, targets: Dict[str, Tuple[str, int]]) -> None:
    """
    Add targets to the DataFrame based on the specified aggregations.
    For each sku_id, the targets is computed as the aggregations of the next N-days.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to add the target to. Changes are applied inplace.
    targets : Dict[str, Tuple[str, int]]
        Dictionary with the following structure:
        {
            "target_name": ("agg_col", "days"),
            ...
        }
        where:
            - target_name: name of the target to add
            - agg_col: name of the column to aggregate
            - days: number of next days to include into rolling window
            (current date is always excluded from the rolling window)
    """
    for target_name, (agg_col, days) in targets.items():
        # Сначала сдвигаем данные на 1 вперед
        shifted_series = df.groupby("sku_id")[agg_col].shift(1)

        # Применяем скользящее окно для суммирования следующих 'days' дней
        rolled_series = shifted_series.rolling(window=days).sum()

        # Присваиваем результат в DataFrame, сбрасывая индекс
        df[target_name] = rolled_series.reset_index(level=0, drop=True)


# Загрузите данные
df = pd.read_csv(file_path)

filterred_df = df[df['sku_id'] == 0].head(10)
filterred_df.drop(columns=['sku','price'], inplace=True)
print(filterred_df)
print('------------------')


rolled = filterred_df.groupby("sku_id")['qty'].rolling(window=3).sum().shift(-3)
        # Сбрасываем индекс и присваиваем результат в DataFrame
filterred_df['3_days'] = rolled.reset_index(level=0, drop=True)

print(filterred_df)


print('------------------')


# rolled = filterred_df.iloc[::-1].shift(1).groupby("sku_id")['qty'].rolling(window=3).sum()
        # Сбрасываем индекс и присваиваем результат в DataFrame
# rolled = filterred_df['qty'].iloc[::-1].shift(1)
# filterred_df['3d_days'] = rolled.reset_index(level=0, drop=True)

# print(filterred_df)





# # Определите признаки и цели
# FEATURES = {
#     "qty_7d_avg": ("qty", 7, "avg", None),
#     "qty_7d_q10": ("qty", 7, "quantile", 10),
#     "qty_7d_q50": ("qty", 7, "quantile", 50),
#     "qty_7d_q90": ("qty", 7, "quantile", 90),
#     "qty_14d_avg": ("qty", 14, "avg", None),
#     "qty_14d_q10": ("qty", 14, "quantile", 10),
#     "qty_14d_q50": ("qty", 14, "quantile", 50),
#     "qty_14d_q90": ("qty", 14, "quantile", 90),
#     "qty_21d_avg": ("qty", 21, "avg", None),
#     "qty_21d_q10": ("qty", 21, "quantile", 10),
#     "qty_21d_q50": ("qty", 21, "quantile", 50),
#     "qty_21d_q90": ("qty", 21, "quantile", 90),
# }

# TARGETS = {
#     "next_7d": ("qty", 7),
#     "next_14d": ("qty", 14),
#     "next_21d": ("qty", 21),
# }

# # Примените функции add_features и add_targets
# add_features(df, FEATURES)
# add_targets(df, TARGETS)

# # Сохраните результаты в файл
# df.to_csv(file_path, index=False)