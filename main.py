import pandas as pd
from flask import Flask, request, jsonify
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
import warnings
import os
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# Load and preprocess data
def load_data(file_path="Product_Sales_Data_Final.csv"):
    """Load and preprocess the sales data."""
    try:
        df = pd.read_csv(file_path)
        # Convert date to datetime
        df["Date"] = pd.to_datetime(df["Date"])
        # Create year-month field
        df["YearMonth"] = df["Date"].dt.strftime("%Y-%m")
        # Create month field (for seasonality analysis)
        df["Month"] = df["Date"].dt.month
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


# Global Data
DATA = None


@app.before_request
def initialize():
    """Initialize the data when the API starts."""
    global DATA
    if DATA is None:
        DATA = load_data()
        logger.info("Data loaded successfully!")


# Helper function to get target date
def get_target_date(date_param=None):
    """Get target date from param or use current month"""
    if date_param is None:
        today = datetime.now()
        return datetime(today.year, today.month, 1)
    return pd.to_datetime(date_param)


# Helper functions for forecasting
def forecast_product_sales(product_id, target_date):
    """Forecast sales for a specific product for a target date."""
    global DATA
    # Filter data for the specific product
    product_data = DATA[DATA["Product_ID"] == product_id]
    if product_data.empty:
        return {"error": f"No data found for product ID {product_id}"}

    # Group by YearMonth and sum sales
    monthly_sales = (
        product_data.groupby("YearMonth")["Sales_Quantity"].sum().reset_index()
    )
    monthly_sales.set_index("YearMonth", inplace=True)

    # Check if we have enough data points
    if len(monthly_sales) < 3:
        return {"error": f"Insufficient data for forecasting product ID {product_id}"}

    try:
        # Try ARIMA model for forecasting
        model = ARIMA(monthly_sales, order=(1, 1, 1))
        model_fit = model.fit()

        # Determine number of steps to forecast
        last_date = pd.to_datetime(monthly_sales.index[-1])
        target_date = pd.to_datetime(target_date)
        steps = (
            (target_date.year - last_date.year) * 12
            + target_date.month
            - last_date.month
        )

        if steps <= 0:
            # We already have data for this month
            target_sales = monthly_sales.loc[target_date.strftime("%Y-%m")].iloc[0]
        else:
            # Need to forecast
            forecast = model_fit.forecast(steps=steps)
            target_sales = max(round(forecast.iloc[-1]), 0)  # Ensure non-negative

        # Get product name
        product_name = product_data["Product_Name"].iloc[-1]
        product_category = product_data["Product_Category"].iloc[-1]

        return {
            "product_id": product_id,
            "product_name": product_name,
            "category": product_category,
            "forecast_date": target_date.strftime("%Y-%m"),
            "forecasted_sales": int(target_sales),
        }
    except Exception as e:
        logger.error(f"Error forecasting for product {product_id}: {e}")
        # Fallback to exponential smoothing
        try:
            model = ExponentialSmoothing(
                monthly_sales, seasonal_periods=12, trend="add", seasonal="add"
            )
            model_fit = model.fit()
            # Forecast
            forecast = model_fit.forecast(steps)
            target_sales = max(round(forecast.iloc[-1]), 0)
            product_name = product_data["Product_Name"].iloc[-1]
            product_category = product_data["Product_Category"].iloc[-1]
            return {
                "product_id": product_id,
                "product_name": product_name,
                "category": product_category,
                "forecast_date": target_date.strftime("%Y-%m"),
                "forecasted_sales": int(target_sales),
            }
        except Exception as e2:
            logger.error(
                f"Error with fallback forecasting for product {product_id}: {e2}"
            )
            return {"error": f"Failed to forecast for product {product_id}"}


def forecast_category_sales(category, target_date):
    """Forecast sales for a specific category for a target date."""
    global DATA
    # Filter data for the specific category
    category_data = DATA[DATA["Product_Category"] == category]
    if category_data.empty:
        return {"error": f"No data found for category {category}"}

    # Group by YearMonth and sum sales
    monthly_sales = (
        category_data.groupby("YearMonth")["Sales_Quantity"].sum().reset_index()
    )
    monthly_sales.set_index("YearMonth", inplace=True)

    try:
        # Try ARIMA model for forecasting
        model = ARIMA(monthly_sales, order=(1, 1, 1))
        model_fit = model.fit()

        # Determine number of steps to forecast
        last_date = pd.to_datetime(monthly_sales.index[-1])
        target_date = pd.to_datetime(target_date)
        steps = (
            (target_date.year - last_date.year) * 12
            + target_date.month
            - last_date.month
        )

        if steps <= 0:
            # We already have data for this month
            target_sales = monthly_sales.loc[target_date.strftime("%Y-%m")].iloc[0]
        else:
            # Need to forecast
            forecast = model_fit.forecast(steps=steps)
            target_sales = max(round(forecast.iloc[-1]), 0)  # Ensure non-negative

        return {
            "category": category,
            "forecast_date": target_date.strftime("%Y-%m"),
            "forecasted_sales": int(target_sales),
        }
    except Exception as e:
        logger.error(f"Error forecasting for category {category}: {e}")
        # Fallback to exponential smoothing
        try:
            model = ExponentialSmoothing(
                monthly_sales, seasonal_periods=12, trend="add", seasonal="add"
            )
            model_fit = model.fit()
            # Forecast
            forecast = model_fit.forecast(steps)
            target_sales = max(round(forecast.iloc[-1]), 0)
            return {
                "category": category,
                "forecast_date": target_date.strftime("%Y-%m"),
                "forecasted_sales": int(target_sales),
            }
        except Exception as e2:
            logger.error(
                f"Error with fallback forecasting for category {category}: {e2}"
            )
            return {"error": f"Failed to forecast for category {category}"}


def get_historical_seasonality():
    """Get historical seasonality factors for better forecasting."""
    global DATA
    # Create monthly seasonality factors
    monthly_sales = DATA.groupby(["Month"])["Sales_Quantity"].sum().reset_index()
    total_sales = monthly_sales["Sales_Quantity"].sum()
    monthly_sales["SeasonalityFactor"] = monthly_sales["Sales_Quantity"] / (
        total_sales / 12
    )
    return (
        monthly_sales[["Month", "SeasonalityFactor"]]
        .set_index("Month")
        .to_dict()["SeasonalityFactor"]
    )


def get_product_monthly_seasonality(product_id):
    """Get monthly seasonality factors for a specific product."""
    global DATA
    product_data = DATA[DATA["Product_ID"] == product_id]
    if product_data.empty:
        return None

    # Create monthly seasonality factors
    monthly_sales = (
        product_data.groupby(["Month"])["Sales_Quantity"].sum().reset_index()
    )
    total_sales = monthly_sales["Sales_Quantity"].sum()
    if total_sales == 0:
        return None

    monthly_sales["SeasonalityFactor"] = monthly_sales["Sales_Quantity"] / (
        total_sales / 12
    )
    return (
        monthly_sales[["Month", "SeasonalityFactor"]]
        .set_index("Month")
        .to_dict()["SeasonalityFactor"]
    )


def predict_top_products(target_date, top_n=10):
    """
    Predict the top N products for a target date using time series forecasting
    and historical seasonality patterns.
    """
    global DATA
    # Get unique product IDs
    product_ids = DATA["Product_ID"].unique()

    # Forecast sales for each product
    forecasts = []
    for product_id in product_ids:
        forecast = forecast_product_sales(product_id, target_date)
        if "error" not in forecast:
            forecasts.append(forecast)

    # Sort by forecasted sales and get top N
    top_products = sorted(forecasts, key=lambda x: x["forecasted_sales"], reverse=True)[
        :top_n
    ]
    return top_products


def predict_top_categories(target_date):
    """
    Predict the top categories for a target date using time series forecasting
    and historical seasonality patterns.
    """
    global DATA
    # Get unique categories
    categories = DATA["Product_Category"].unique()

    # Forecast sales for each category
    forecasts = []
    for category in categories:
        forecast = forecast_category_sales(category, target_date)
        if "error" not in forecast:
            forecasts.append(forecast)

    # Sort by forecasted sales
    top_categories = sorted(
        forecasts, key=lambda x: x["forecasted_sales"], reverse=True
    )
    return top_categories


def predict_top_products_by_category(category, target_date, top_n=5):
    """
    Predict the top N products for a specific category for a target date.
    """
    global DATA
    # Ensure category filtering is strict
    category_products = DATA[
        DATA["Product_Category"].str.strip().str.lower() == category.lower()
    ]["Product_ID"].unique()

    forecasts = []
    for product_id in category_products:
        forecast = forecast_product_sales(product_id, target_date)
        if "error" not in forecast:
            forecasts.append(forecast)

    # Filter only the products that actually belong to the given category
    forecasts = [
        f for f in forecasts if f.get("category", "").lower() == category.lower()
    ]

    # Sort by forecasted sales and get top N
    top_products = sorted(forecasts, key=lambda x: x["forecasted_sales"], reverse=True)[
        :top_n
    ]
    return top_products


# RESTRUCTURED API ENDPOINTS
# --------------------------


# 1. Top Products Endpoint
@app.route("/api/top-products", methods=["GET"])
def get_top_products():
    """
    Endpoint to get top products forecast

    Query Parameters:
    - date: Target date in YYYY-MM format (default: current month)
    - top_n: Number of top products to return (default: 10)

    Returns:
    JSON with top products forecast
    """
    # Get query parameters
    target_date = request.args.get("date", None)
    top_n = int(request.args.get("top_n", 10))

    # Process target date
    target_date = get_target_date(target_date)

    try:
        # Predict top products
        top_products = predict_top_products(target_date, top_n)

        return jsonify(
            {
                "forecast_date": target_date.strftime("%Y-%m"),
                "top_products": top_products,
            }
        )
    except Exception as e:
        logger.error(f"Error generating top products forecast: {e}")
        return jsonify({"error": str(e)}), 500


# 2. Top Categories Endpoint
@app.route("/api/top-categories", methods=["GET"])
def get_top_categories():
    """
    Endpoint to get top categories forecast

    Query Parameters:
    - date: Target date in YYYY-MM format (default: current month)

    Returns:
    JSON with top categories forecast
    """
    # Get query parameters
    target_date = request.args.get("date", None)

    # Process target date
    target_date = get_target_date(target_date)

    try:
        # Predict top categories
        top_categories = predict_top_categories(target_date)

        return jsonify(
            {
                "forecast_date": target_date.strftime("%Y-%m"),
                "top_categories": top_categories,
            }
        )
    except Exception as e:
        logger.error(f"Error generating top categories forecast: {e}")
        return jsonify({"error": str(e)}), 500


# 3. Category-Specific Top Products Endpoint
@app.route("/api/category/<category>/top-products", methods=["GET"])
def get_category_top_products(category):
    """
    Endpoint to get top products for a specific category

    Path Parameters:
    - category: Category name

    Query Parameters:
    - date: Target date in YYYY-MM format (default: current month)
    - top_n: Number of top products to return (default: 5)

    Returns:
    JSON with top products for the specified category
    """
    # Get query parameters
    target_date = request.args.get("date", None)
    top_n = int(request.args.get("top_n", 5))

    # Process target date
    target_date = get_target_date(target_date)

    try:
        # Get top products for the category
        category_top_products = predict_top_products_by_category(
            category, target_date, top_n
        )

        # Get category forecast
        category_forecast = forecast_category_sales(category, target_date)

        return jsonify(
            {
                "forecast_date": target_date.strftime("%Y-%m"),
                "category": category,
                "category_forecast": (
                    category_forecast.get("forecasted_sales", 0)
                    if "error" not in category_forecast
                    else 0
                ),
                "top_products": category_top_products,
            }
        )
    except Exception as e:
        logger.error(f"Error generating category top products forecast: {e}")
        return jsonify({"error": str(e)}), 500


# 4. Single Product Forecast Endpoint
@app.route("/api/product/<product_id>/forecast", methods=["GET"])
def get_product_forecast(product_id):
    """
    Endpoint to get forecast for a specific product

    Path Parameters:
    - product_id: ID of the product to forecast

    Query Parameters:
    - date: Target date in YYYY-MM format (default: current month)
    - months: Number of months to forecast (default: 1)

    Returns:
    JSON with forecast results for the product
    """
    # Get query parameters
    target_date = request.args.get("date", None)
    months = int(request.args.get("months", 1))

    # Process target date
    target_date = get_target_date(target_date)

    try:
        # Get base forecast for the target date
        base_forecast = forecast_product_sales(product_id, target_date)

        if "error" in base_forecast:
            return jsonify(base_forecast), 404

        # If multiple months requested, generate forecasts for each month
        if months > 1:
            monthly_forecasts = [base_forecast]

            for i in range(1, months):
                next_month = target_date + pd.DateOffset(months=i)
                forecast = forecast_product_sales(product_id, next_month)

                if "error" not in forecast:
                    monthly_forecasts.append(forecast)

            return jsonify(
                {
                    "product_id": product_id,
                    "product_name": base_forecast["product_name"],
                    "category": base_forecast["category"],
                    "monthly_forecasts": monthly_forecasts,
                }
            )
        else:
            return jsonify(base_forecast)
    except Exception as e:
        logger.error(f"Error generating forecast for product {product_id}: {e}")
        return jsonify({"error": str(e)}), 500


# 5. Product Seasonality Endpoint
@app.route("/api/product/<product_id>/seasonality", methods=["GET"])
def get_product_seasonality(product_id):
    """
    Endpoint to get seasonality factors for a specific product

    Path Parameters:
    - product_id: ID of the product

    Returns:
    JSON with monthly seasonality factors
    """
    try:
        seasonality = get_product_monthly_seasonality(product_id)

        if seasonality is None:
            return (
                jsonify(
                    {"error": f"No seasonality data available for product {product_id}"}
                ),
                404,
            )

        # Convert to list format for easier consumption
        seasonality_list = [
            {"month": month, "factor": factor} for month, factor in seasonality.items()
        ]

        # Get product info
        product_data = DATA[DATA["Product_ID"] == product_id]

        if product_data.empty:
            return jsonify({"error": f"Product {product_id} not found"}), 404

        product_name = product_data["Product_Name"].iloc[-1]
        product_category = product_data["Product_Category"].iloc[-1]

        return jsonify(
            {
                "product_id": product_id,
                "product_name": product_name,
                "category": product_category,
                "seasonality": seasonality_list,
            }
        )
    except Exception as e:
        logger.error(f"Error getting seasonality for product {product_id}: {e}")
        return jsonify({"error": str(e)}), 500


# 6. Overall Seasonality Endpoint
@app.route("/api/seasonality", methods=["GET"])
def get_overall_seasonality():
    """
    Endpoint to get overall seasonality factors across all products

    Returns:
    JSON with monthly seasonality factors
    """
    try:
        seasonality = get_historical_seasonality()

        # Convert to list format for easier consumption
        seasonality_list = [
            {"month": month, "factor": factor} for month, factor in seasonality.items()
        ]

        return jsonify({"seasonality": seasonality_list})
    except Exception as e:
        logger.error(f"Error getting overall seasonality: {e}")
        return jsonify({"error": str(e)}), 500


# 7. Categories List Endpoint
@app.route("/api/categories", methods=["GET"])
def get_categories():
    """
    Endpoint to get list of all product categories

    Returns:
    JSON with list of all categories
    """
    try:
        categories = DATA["Product_Category"].unique().tolist()
        return jsonify({"categories": categories})
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        return jsonify({"error": str(e)}), 500


# 8. Products List Endpoint
@app.route("/api/products", methods=["GET"])
def get_products():
    """
    Endpoint to get list of all products

    Query Parameters:
    - category: Filter products by category (optional)

    Returns:
    JSON with list of products
    """
    try:
        category = request.args.get("category", None)

        if category:
            filtered_data = DATA[
                DATA["Product_Category"].str.lower() == category.lower()
            ]
        else:
            filtered_data = DATA

        products = (
            filtered_data[["Product_ID", "Product_Name", "Product_Category"]]
            .drop_duplicates()
            .to_dict("records")
        )

        return jsonify({"products": products})
    except Exception as e:
        logger.error(f"Error getting products: {e}")
        return jsonify({"error": str(e)}), 500


# Original combined endpoint (now as a separate option)
@app.route("/api/forecasts", methods=["GET"])
def get_forecasts():
    """
    Combined forecast endpoint that returns top products, top categories,
    and top products in each category.

    Query Parameters:
    - date: Target date in YYYY-MM format (default: current month)
    - top_n: Number of top products to return (default: 10)

    Returns:
    JSON with forecast results
    """
    # Get query parameters
    target_date = request.args.get("date", None)
    top_n = int(request.args.get("top_n", 10))

    # Process target date
    target_date = get_target_date(target_date)

    try:
        # Predict top products overall
        top_products = predict_top_products(target_date, top_n)

        # Predict top categories
        top_categories = predict_top_categories(target_date)

        # Predict top products for each top category
        top_category_products = {}
        for category_forecast in top_categories:
            category = category_forecast["category"]
            # Get top products specific to this category only
            category_top_products = predict_top_products_by_category(
                category, target_date, top_n=5
            )
            top_category_products[category] = category_top_products

        # Prepare response
        response = {
            "forecast_date": target_date.strftime("%Y-%m"),
            "top_products": top_products,
            "top_categories": top_categories,
            "top_products_by_category": top_category_products,
        }

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error generating forecasts: {e}")
        return jsonify({"error": str(e)}), 500


# Run the application
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
