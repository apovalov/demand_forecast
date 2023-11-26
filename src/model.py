from datetime import timedelta
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from tqdm import tqdm


def split_train_test(
    df: pd.DataFrame,
    test_days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and test sets.

    The last `test_days` days are held out for testing.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        test_days (int): The number of days to include in the test set (default: 30).
            use ">=" sign for df_test

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
        A tuple containing the train and test DataFrames.
    """
    df['day'] = pd.to_datetime(df['day'], dayfirst=True)

    cutoff_date = df['day'].max() - timedelta(days=test_days)

    df_train = df[df['day'] < cutoff_date]
    df_test = df[df['day'] >= cutoff_date]
    return df_train, df_test


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Calculate the quantile loss between predictions and true values.

    Parameters:
    - y_true: np.ndarray - Array of true values.
    - y_pred: np.ndarray - Array of predicted values.
    - quantile: float - The quantile for which to calculate the loss.

    Returns:
    - The quantile loss as a float.
    """
    error = y_true - y_pred
    loss = np.maximum(quantile * error, (quantile - 1) * error)
    return np.mean(loss)

def evaluate_model(df_true: pd.DataFrame,
                   df_pred: pd.DataFrame,
                   quantiles: List[float] = [0.1, 0.5, 0.9],
                   horizons: List[int] = [7, 14, 21]) -> pd.DataFrame:
    """
    Evaluate the model performance using quantile loss.

    Parameters:
    - df_true: pd.DataFrame - DataFrame with true values.
    - df_pred: pd.DataFrame - DataFrame with predicted values.
    - quantiles: List[float] - List of quantiles.
    - horizons: List[int] - List of prediction horizons.

    Returns:
    - DataFrame summarizing the losses for each quantile and horizon.
    """
    losses = {}

    for quantile in quantiles:
        for horizon in horizons:
            true = df_true[f"next_{horizon}d"].values
            pred = df_pred[f"pred_{horizon}d_q{int(quantile * 100)}"].values
            loss = quantile_loss(true, pred, quantile)

            losses[(quantile, horizon)] = loss

    losses = pd.DataFrame(losses, index=["loss"]).T.reset_index()
    losses.columns = ["quantile", "horizon", "avg_quantile_loss"]

    return losses

class MultiTargetModel:
    """
    A class representing a model that predicts multiple target variables.

    Attributes:
    - quantiles: List[float] - Quantiles for the model.
    - horizons: List[int] - Prediction horizons.
    - features: List[str] - List of feature names.
    - targets: List[str] - List of target variable names.
    - fitted_models_: dict - A dictionary to store fitted models.
    - sku_col: str - Column name representing SKU.
    - date_col: str - Column name representing date.
    """

    def __init__(self, features: List[str], horizons: List[int] = [7, 14, 21], quantiles: List[float] = [0.1, 0.5, 0.9]):
        self.quantiles = quantiles
        self.horizons = horizons
        self.features = features
        self.targets = [f"next_{horizon}d" for horizon in horizons]
        self.fitted_models_ = {}
        self.sku_col = "sku_id"  # Убедитесь, что это поле определено
        self.date_col = "day"

    def fit(self, data: pd.DataFrame, verbose: bool = False) -> None:
        """
        Fit the model on the provided data.

        Parameters:
        - data: pd.DataFrame - The input data.
        - verbose: bool - If True, display progress.
        """
        # Очистка данных от пропусков
        data = data.copy()

        # Очистка данных от пропусков
        data = data.dropna(subset=self.features + self.targets)

        # Преобразование столбца с датами в datetime
        data[self.date_col] = pd.to_datetime(data[self.date_col])

        data.set_index([self.sku_col, self.date_col], inplace=True)
        data.sort_index(inplace=True)


        skus = tqdm(data.index.get_level_values(self.sku_col).unique()) if verbose else data.index.get_level_values(
            self.sku_col).unique()

        for sku in skus:
            sku_data = data.xs(sku, level=self.sku_col)
            self.fitted_models_[sku] = {}

            for horizon, target in zip(self.horizons, self.targets):
                for quantile in self.quantiles:
                    model = QuantileRegressor(quantile=quantile, solver="highs", alpha=0)
                    X = sku_data[self.features]
                    y = sku_data[target]

                    model.fit(X, y)
                    self.fitted_models_[sku][(quantile, horizon)] = model

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict using the fitted model.

        Parameters:
        - data: pd.DataFrame - The input data for prediction.

        Returns:
        - A DataFrame with the predictions.
        """
        data = data.copy()
        data[self.date_col] = pd.to_datetime(data[self.date_col])
        data.set_index([self.sku_col, self.date_col], inplace=True)

        predictions = pd.DataFrame()

        for sku in data.index.get_level_values(self.sku_col).unique():
            sku_data = data.xs(sku, level=self.sku_col).copy()
            sku_data[self.sku_col] = sku  # Добавляем столбец sku_id

            for horizon in self.horizons:
                for quantile in self.quantiles:
                    target_name = f"pred_{horizon}d_q{int(quantile * 100)}"
                    model = self.fitted_models_.get(sku, {}).get((quantile, horizon), None)

                    if model is not None:
                        sku_data.loc[:, target_name] = model.predict(sku_data[self.features])
                    else:
                        sku_data.loc[:, target_name] = 0

            sku_data.reset_index(inplace=True)  # Сбрасываем индекс, сохраняя sku_id как столбец

            if predictions.empty:
                predictions = sku_data
            else:
                predictions = pd.concat([predictions, sku_data])

        required_columns = ['sku_id', 'day'] + [f"pred_{horizon}d_q{int(quantile * 100)}" for quantile in self.quantiles for horizon in self.horizons]
        predictions = predictions[required_columns]

        return predictions
