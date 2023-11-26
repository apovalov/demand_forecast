import os
import sys
from typing import List

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel
from pydantic import Field

PREDICTIONS_LOCAL_PATH = os.path.join(sys.path[0], "data/predictions.csv")

app = FastAPI()
predictions = None


class SKUInfo(BaseModel):
    sku_id: int = Field(..., description="The SKU ud.")
    stock: int = Field(0, description="The current stock level.")


class SKURequest(BaseModel):
    sku: SKUInfo = Field(..., description="The sku and stock level.")
    horizon_days: int = Field(7, description="The number of days in the horizon.")
    confidence_level: float = Field(0.1, description="The confidence level.")


class LowStockSKURequest(BaseModel):
    confidence_level: float = Field(..., description="The confidence level.")
    horizon_days: int = Field(..., description="The number of days in the horizon.")
    # dict of sku and stock
    sku_stock: List[SKUInfo] = Field(..., description="The sku and stock level.")


@app.post("/api/predictions/upload")
def upload_predictions(file: UploadFile = File(...)) -> dict:
    """Upload predictions"""
    try:
        content = file.file.read()
        with open(PREDICTIONS_LOCAL_PATH, "wb") as f:
            f.write(content)

        df = pd.read_csv(PREDICTIONS_LOCAL_PATH)

        global predictions
        predictions = df

        return {"success": 1}
    except Exception as e:
        return {"success": 0, "error": str(e)}


@app.post("/api/how_much_to_order")
def how_much_to_order(request_data: SKURequest) -> dict:
    """Predict how much to order"""
    try:
        sku_id = request_data.sku.sku_id
        current_stock = request_data.sku.stock
        horizon_days = request_data.horizon_days
        confidence_level = request_data.confidence_level

        # Ensure predictions are loaded
        if predictions is None:
            raise ValueError("Predictions are not loaded")

        # Filter for the relevant prediction
        prediction_row = predictions[
            (predictions['sku_id'] == sku_id)
        ]

        # Get the prediction column name based on the horizon and confidence level
        quantile = int(confidence_level * 100)
        prediction_col = f'pred_{horizon_days}d_q{quantile}'
        predicted_sales = prediction_row[prediction_col].iloc[0]

        # Calculate the recommended order quantity by rounding up to the nearest whole number
        recommended_order_qty = np.ceil(max(0, predicted_sales - current_stock))

        return {"quantity": int(recommended_order_qty)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/stock_level_forecast")
def stock_level_forecast(request_data: SKURequest) -> dict:
    """Predict stock level"""
    try:
        sku_id = request_data.sku.sku_id
        current_stock = request_data.sku.stock
        horizon_days = request_data.horizon_days
        confidence_level = request_data.confidence_level

        # Ensure predictions are loaded
        if predictions is None:
            raise ValueError("Predictions are not loaded")

        # Filter for the relevant SKU
        sku_predictions = predictions[predictions['sku_id'] == sku_id]

        # Get the prediction column name based on the horizon and confidence level
        quantile = int(confidence_level * 100)
        prediction_col = f'pred_{horizon_days}d_q{quantile}'
        if prediction_col not in sku_predictions.columns:
            raise ValueError(f"Prediction column {prediction_col} not found in the data")

        # Get the predicted sales
        predicted_sales = sku_predictions[prediction_col].iloc[0]

        # Calculate the forecast stock level
        stock_level = max(0, current_stock - predicted_sales)

        return {"stock_forecast": stock_level}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



# The actual FastAPI endpoint
@app.post("/api/low_stock_sku_list")
def low_stock_sku_list(request_data: LowStockSKURequest) -> dict:
    """Return SKU list with low stock level"""
    try:
        horizon_days = request_data.horizon_days
        confidence_level = request_data.confidence_level
        skus_stock_list = request_data.sku_stock

        # Ensure predictions are loaded
        if predictions is None:
            raise ValueError("Predictions are not loaded")

        low_stock_list = []

        for sku_details in skus_stock_list:
            sku_id = sku_details.sku_id
            current_stock = sku_details.stock

            # Filter the predictions for the specific SKU
            prediction_row = predictions[
                (predictions['sku_id'] == sku_id)
            ]

            # Get the prediction column name based on the horizon and confidence level
            quantile = int(confidence_level * 100)
            prediction_col = f'pred_{horizon_days}d_q{quantile}'
            predicted_sales = prediction_row[prediction_col].iloc[0]

            # Calculate the forecast stock level
            forecast_stock_level = current_stock - predicted_sales

            # SKU is in the low stock list if the forecast stock level is less than or equal to zero
            if forecast_stock_level <= 0:
                low_stock_list.append(sku_id)

        return {"sku_list": low_stock_list}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost", port=5000)


if __name__ == "__main__":
    main()
