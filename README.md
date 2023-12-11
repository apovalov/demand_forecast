# Demand forecast

1. Data aggregation
2. Feature engineering
3. Quantile Regression
4. Missed Profits / Bootstrap
5. Inference Pipеline

What is demand, exactly? Demand is the amount of a product that users would buy if we had an unlimited number of units of that product in stock.

Unfortunately, demand does not always meet supply, which leads to out-of-stock (OOS) - a situation when there is no product left in stock, and you have to wait several days for the next delivery. Because of this, the user goes to a competitor for a purchase, which translates into missed profits, lost profits for us. We don't want to lose money, and we don't want to lose customers either. Therefore, we prefer to buy in advance to meet the demand of customers.

On the other hand, the more goods we buy, the more of our money and the sellers' money on our platform is frozen in unsold goods. Keeping goods in warehouses for long periods of time not only reduces the efficiency of warehouse space utilization, but can also lead to loss of goods (e.g. due to obsolescence, spoilage, etc.), which also costs money.

The difficulty lies in the fact that it is difficult to find the "golden mean" between shortage and excess of goods in stock, especially given the complex and volatile dynamics of demand. Having the right product at the right time and in the right quantity (and sometimes in the right warehouse!) is what a Stock Management System is all about.

**Turnover** is a measure of how quickly the money spent to purchase goods will be returned; the better the turnover rate, the faster the purchase-sales cycle and the faster the warehouse "metabolism". Higher turnover means more revenue.

## Demand forecasting
Unfortunately, no one knows how to look into the future, so perfect procurement in a vacuum is impossible. But there is a solution and you know it - it is machine learning, the "intuition" of corporations with a lot of experience (big data).

## Prediction error
This is how the need to develop a service for demand forecasting appears. Usually, even before complex ML-solutions they start with something simple, but already working and bringing money.

For example, a category manager takes the average sales of goods for the last 60 days and multiplies it by the number of days for which he makes a purchase. For reliability, this "forecast" is multiplied by some coefficient, adding 10-30% on top (in the future we will understand what task the category manager is trying to solve and how to do it directly). Our partners have a limited amount of money, so some goods turn out to be more "insured", some less "insured".

## ML Forecast
The current semi-manual demand forecast is not showing its best behavior:

About 15-20% of items with zero stock (out-of-stock).
About 20-30% of items are "stale" and not selling (overstock).


# 2. Future Inference Pipeline

To create an effective inference pipeline, consider the following steps:

1. **Data Collection and Preprocessing**: Use various data sources including orders, pricing history, competitor pricing, promotional calendars, stock levels, product categories, etc.

2. **Feature Generation**: Calculate aggregates and various statistical properties over different time periods in the past (e.g., 7/14/30/60 days).

3. **Sales Forecasting**: Provide pessimistic/conservative/optimistic estimates of demand for each product using a trained model. The latest model version is supported by a product registry.

4. **Order Generation Automation**: For each product, compare the sales forecast for the next three weeks against the current stock. Decisions about replenishing stock and generating orders with suppliers are based on this. For example, if the sales forecast is better than the "FuturoHana" service for the next three weeks:
    - Pessimistic — 106 units,
    - Conservative — 353 units,
    - Optimistic — 657 units.

Ensure that stock levels do not fall below the pessimistic demand estimate. For example, currently, we have 82 units in stock (out of a 24-hour permissible level). The FuturoHana service requests 575 units to match the 75-th percentile demand estimate (optimistic forecast).

We check the stock every hour against each product to ensure alignment with these forecasts.

In subsequent sections, we will discuss how we evaluate demand and how we decide on which demand estimate to use for different initiatives, without making exact predictions. We'll also cover how we proceed with the loading of the dataset.


## Sales Aggregation
Let's start with the first step of the Pipeline and collect the initial dataset for our ML system.

The atomic unit of the dataset will be a commodity-day - (sku, day). Accordingly, each prediction will be built for each moment of time (each day) and each commodity - for some period in the future (1-2-3-4 weeks).

In order to calculate both aggregates of future sales (targets: what will we predict?) and aggregates of past sales (attributes: on the basis of what will we predict?), we need aggregates of sales by day.


![Alt text](/img/image.png)

Attributes and target variables
For each row of the dataset (a row is a product-day), we will calculate as attributes:

The average number of sales for the last N days,
Quantile X of sales for the last N days.
As targets we will calculate:

Total number of sales for the next N days.
What attributes and targets should be formed is specified by dictionaries:

```
FEATURES = {
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

TARGETS = {
    "next_7d": ("qty", 7),
    "next_14d": ("qty", 14),
    "next_21d": ("qty", 21),
}
```

## 3.Quantile Regression

The dataset contains the following attributes:

day - day.
sku_id, sku - id and name of the product.
price - price of the product.
qty - quantity of the item that was sold on the current day.
qty_7d_avg, qty_14d_avg, qty_21d_avg - average number of sales for the previous 1, 2 and 3 weeks, including today's day.
qty_7d_q10, qty_7d_q50, qty_7d_q90 (similar to 14d, 21d) - quantiles of 0.1, 0.5, 0.9 sales for the previous 1, 2, 3 weeks.
next_7d, next_14d, next_21d - total number of sales for the next 1, 2, 3 weeks, not including today (targets).

As we discussed earlier, the demand forecast itself is of little concern to business. Businesses are concerned with problems like out-of-stocks (when goods run out even though there is demand for them), overstock (when goods are "stale" and not selling - we have to "liquidate" them by drastically reducing the price and selling them below cost, just so they don't take up space in the warehouse), missed profits (lost revenue due to out-of-stocks), turnover (how long it takes from delivery of a unit of goods to shipment to the customer), etc.

So how do you make these predictions?

It's very simple. Let's forecast not only average values of sales for the next N days, but quantiles. In this case, each quantile will represent a forecast with a different level of confidence.

0.1 quantile is a "pessimistic" forecast (in 90% of cases the forecast is lower than actual sales);
0.5 quantile - "conservative" forecast (in 50% of cases real sales are higher, in 50% of cases real sales are lower);
0.9 quantile - "optimistic" forecast (in 90% of cases the forecast will be higher than real sales).


![Alt text](/img/image-1.png)



Let's look at some examples
Example 1. For a laptop, the quantiles of the sales forecast for the next two weeks are:

0.1 quantile (pessimistic forecast) - 3 pcs. There is a 90% probability that sales will be 3 pc or higher.
0.5 quantile (conservative forecast) - 5 pcs.
0.9 quantile (optimistic forecast) - 8 pcs. With 90% probability sales will be 8 pcs or lower.
It is important for the company to have laptops in stock at any forecast. The delivery time takes 2-3 weeks, and it is difficult to quickly purchase a new batch if sales are higher than the conservative forecast. Therefore, the decision is made to stock 8 laptops.

Example 2. For the keyboard, the quantiles of the sales forecast for the next 2 weeks are:

0.1 quantile (pessimistic forecast) - 10 pcs. There is a 90% probability that sales will be 10 pc or more.
0.5 quantile (conservative forecast) - 25 pcs.
0.9 quantile (optimistic forecast) - 47 pcs. With a probability of 90% sales will be 47 pcs. or lower.
Keyboards can be ordered very quickly. Delivery time takes 2-3 days. Profit from the sale of keyboards is not very significant for the company. Therefore, it is decided to stock 25 keyboards to provide a conservative sales forecast. If the keyboards suddenly run out, the company will quickly reorder them or will not be too "upset" about the lost revenue.


## Quantile Loss Function

The Quantile Loss function is crucial for regression tasks where the objective is to predict a specific quantile of the target variable's distribution rather than its mean. It's especially useful when the cost of overestimation differs from the cost of underestimation.

### Why Quantile Loss?

Unlike Mean Squared Error (MSE) or Mean Absolute Error (MAE), Quantile Loss allows models to focus on the tail of the distribution which is more informative for certain types of predictions, such as sales, where understanding the variability is as important as predicting the central tendency.

### Definition

The Quantile Loss is defined as:

$$L(y, p, q) = q \times \max(y - p, 0) + (1 - q) \times \max(p - y, 0)$$

Where:
- `L(y, p, q)`: Quantile Loss function.
- `y`: Actual value.
- `p`: Predicted value.
- `q`: The quantile being estimated, a value between 0 and 1.

### Implementation Note

The `max` function is used to calculate the penalties for underestimation and overestimation, which are weighted by the quantile `q`. This asymmetric loss function penalizes errors differently, allowing the model to adjust predictions according to the desired quantile of the target distribution.



## 4. Missed Profits
Your colleague has run the payplane on history in one-week increments. Assuming that we make all purchases once a week, he calculated weekly forecasts pred_7d_q50.

Calculate how much money the company would have saved, assuming real demand was at least as high as the median demand.

You need to calculate all the "missed profits" in money in the weeks when SKU sales for the week were less than the median forecast. Group the result by week for all products. One row of the resulting dataset contains the "missed profits" for all products for the week.

Example
Suppose the product "Monitor" was sold in quantities of: [10, 12, 4, 0].

The median forecast pred_7d_q50 of sales for these days is: [9, 10, 8, 9].

The "missed sales" are: [0, 0, 4, 9].

The price of the monitor is 1,500. "Missed sales" in money (missed profit) are: [0, 0, 6000, 13500].

This is a very rough estimate, but will give us at least the magnitude of the losses.

Now we have a sample of 104 observations over 2 years (52 weeks per year x 2 years). Let's apply bootstrap to this estimate and calculate the 95% confidence interval for the estimate of week missed profits.


 ## 5. Stock Management System As a Service

 ![Alt text](/img/image-2.png)

ClearML training pipeline: the model is trained daily on the latest available data for which targets can be calculated. The trained model is stored in the ClearML Model Registry.
ClearML inference pipeline: 9 forecasts for each product (3 horizons x 3 quantiles) are generated daily for the latest available data. The generated forecasts are updated in ML Backend.
ML Backend (Fast API): based on the current forecasts, it gives recommendations on stock replenishment for each commodity.
Stock Management System: an external inventory management system that generates auto-orders. Accesses ML Backend to get recommendations on which items to replenish and in what quantities.