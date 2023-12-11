import pandas as pd
from model import MultiTargetModel

# Путь к файлу с данными
file_path = '../data/features.csv'

# Загрузка данных
df = pd.read_csv(file_path)

# Разделение данных на обучающую и тестовую выборки
cutoff_date = '2022-06-01'
df_train = df[df['day'] < cutoff_date]
df_test = df[df['day'] >= cutoff_date]

# Создание экземпляра модели
model = MultiTargetModel(
    features=[
        "price",
        "qty",
        "qty_7d_avg",
        "qty_7d_q10",
        "qty_7d_q50",
        "qty_7d_q90",
        "qty_14d_avg",
        "qty_14d_q10",
        "qty_14d_q50",
        "qty_14d_q90",
        "qty_21d_avg",
        "qty_21d_q10",
        "qty_21d_q50",
        "qty_21d_q90",
    ],
    horizons=[7, 14, 21],
    quantiles=[0.1, 0.5, 0.9],
)

# Обучение модели
model.fit(df_train)

# Предсказание на тестовых данных
predictions = model.predict(df_test)

# Вывод первых нескольких строк предсказаний
predictions.head()
