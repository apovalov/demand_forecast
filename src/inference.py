from typing import Dict
from typing import Optional
from typing import Tuple

import fire
import pandas as pd
from clearml import TaskTypes
from clearml.automation.controller import PipelineDecorator


@PipelineDecorator.component(
    return_values=["orders"],
    task_type=TaskTypes.data_processing,
)
def fetch_orders(orders_url: str) -> pd.DataFrame:
    import pandas as pd
    from clearml import StorageManager
    import requests
    from urllib.parse import urlencode

    print(f"Downloading orders data from {orders_url}...")

    # Construct the full URL to download data using the Yandex Disk API
    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
    full_url = base_url + urlencode(dict(public_key=orders_url))

    # Get the download URL from the Yandex Disk API
    response = requests.get(full_url)
    download_url = response.json()["href"]

    # Use the StorageManager to handle the download and local caching of the file
    local_path = StorageManager.get_local_copy(remote_url=download_url)

    # Read the data into a DataFrame
    df_orders = pd.read_csv(local_path, parse_dates=["timestamp"], dayfirst=True)

    # Assuming 'timestamp' is the column with dates, filter the last 21 days
    # last_21_days = df_orders['timestamp'].max() - pd.Timedelta(days=21)
    # df_orders = df_orders[df_orders['timestamp'] > last_21_days]

    print(f"Orders data downloaded. Shape: {df_orders.shape}")

    return df_orders


@PipelineDecorator.component(
    return_values=["sales"],
    task_type=TaskTypes.data_processing,
)
def extract_sales(df_orders: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    print("Extracting sales data...")
    df_orders['day'] = pd.to_datetime(df_orders['timestamp'], dayfirst=True).dt.date

    # Группировка данных с сохранением уникальных значений sku и price
    df_sales = df_orders.groupby(['day', 'sku_id']).agg(
        qty=('qty', 'sum'),
        sku=('sku', 'first'),  # 'first' или 'max' в зависимости от данных
        price=('price', 'first')  # 'first' или 'max' в зависимости от данных
    ).reset_index()

    unique_skus = df_orders['sku_id'].unique()
    date_range = pd.date_range(df_sales['day'].min(), df_sales['day'].max())

    df_grid = pd.MultiIndex.from_product([date_range.date, unique_skus], names=['day', 'sku_id']).to_frame(index=False)
    df_grid['day'] = df_grid['day'].astype('datetime64[ns]').dt.date

    df_sales = df_grid.merge(df_sales, on=['day', 'sku_id'], how='left')
    df_sales['qty'] = df_sales['qty'].fillna(0)

    print(f"Sales data extracted. Shape: {df_sales.shape}")
    return df_sales


@PipelineDecorator.component(
    return_values=["features"],
    task_type=TaskTypes.data_processing,
)
def extract_features(
    df_sales: pd.DataFrame,
    features: Dict[str, Tuple[str, int, str, Optional[int]]],
) -> pd.DataFrame:

    from features import add_features  # Importing the function from your features.py

    print("Extracting features...")

    # Assuming add_features is a function that takes a DataFrame and a dict of feature configurations
    # and returns a DataFrame with new features added
    df_features = df_sales.copy()

    # for feature_name, (agg_col, window, agg_func, shift_period) in features.items():
    #     df_features = add_features(df_features, agg_col, window, agg_func, shift_period)
    add_features(df_features, features)

    print(f"Features extracted. features.csv shape: {df_features.shape}")

    return df_features


@PipelineDecorator.component(
    return_values=["predictions"],
    task_type=TaskTypes.inference,
)
def predict(
    model_path: str,
    df_features: pd.DataFrame,
) -> pd.DataFrame:
    import pandas as pd
    import pickle

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    df_features["day"] = pd.to_datetime(df_features["day"])
    last_day_data = df_features[df_features['day'] == df_features['day'].max()]

    last_day_data.fillna(0, inplace=True)

    predictions = model.predict(last_day_data)
    # Apply additional logic for model-specific predictions if needed

    expected_columns = ['sku_id', 'day'] + [f'pred_{h}d_q{int(q*100)}' for q in [0.1, 0.5, 0.9] for h in [7, 14, 21]]
    predictions = predictions[expected_columns]

    return predictions


@PipelineDecorator.pipeline(
    name="Inference Pipeline",
    project="Stock Management System Task",
    version="1.0.0",
)
def run_pipeline(
    orders_url: str,
    model_path: str,
    features: Dict[str, Tuple[str, int, str, Optional[int]]],
) -> None:
    from clearml import Task
    # Fetch the latest orders data
    orders_df = fetch_orders(orders_url)

    # Extract sales data from the orders
    sales_df = extract_sales(orders_df)

    # Generate features from the sales data
    features_df = extract_features(sales_df, features)

    # Use the model to predict sales
    predictions_df = predict(model_path, features_df)

    # predictions_df now contains the predictions
    # You can now save the predictions, analyze them, or return them as needed
    # For example, you might upload them as an artifact of the pipeline task
    current_task = Task.current_task()
    current_task.upload_artifact('predictions', predictions_df)


def main(
    orders_url: str = "https://disk.yandex.ru/d/OK5gyMuEfhJA0g",
    model_path: str = "model.pkl",
    debug: bool = False,
) -> None:
    """Main function

    Args:
        orders_url (str): URL to the orders data on Yandex Disk
        model_path (str): Local path of production model
        debug (bool, optional): Run the pipeline in debug mode.
            In debug mode no Taska are created, so it is running faster.
            Defaults to False.
    """

    if debug:
        PipelineDecorator.debug_pipeline()
    else:
        PipelineDecorator.run_locally()

    features = {
        "qty_7d_avg": ("qty", 7, "avg", None),
        "qty_7d_q10": ("qty", 7, "quantile", 10),
        "qty_7d_q50": ("qty", 7, "quantile", 50),
        "qty_7d_q90": ("qty", 7, "quantile", 90),
        "qty_14d_avg": ("qty", 14, "avg", None),
        "qty_14d_q10": ("qty", 14, "quantile", 10),
        "qty_14d_q50": ("qty", 14, "quantile", 50),
        "qty_14d_q90": ("qty", 14, "quantile", 90),
        "qty_21d_avg": ("qty", 21, "avg", None),
        "qty_21d_q10": ("qty", 21, "quantile", 10),
        "qty_21d_q50": ("qty", 21, "quantile", 50),
        "qty_21d_q90": ("qty", 21, "quantile", 90),
    }

    run_pipeline(
        orders_url=orders_url,
        model_path=model_path,
        features=features,
    )


if __name__ == "__main__":
    fire.Fire(main)
