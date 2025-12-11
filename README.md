# retail-forecast

A machine learning-based solution for retail sales forecasting using historical revenue data with Portuguese column names.

## Overview

This project implements an end-to-end retail forecasting pipeline that:
- Loads retail transaction data with Portuguese column names (data, venda, preco)
- Performs comprehensive feature engineering with temporal and lag features
- Trains multiple machine learning models (Linear Regression, Random Forest)
- Evaluates model performance using RMSE and MAE metrics
- Provides statistical analysis of price-based promotions
- Generates actionable insights through feature importance analysis

## Features

### Data Processing
- Handles CSV files with Portuguese column names
- Robust date parsing with error handling
- Revenue calculation from units sold and unit price
- Monthly aggregation for time-series analysis

### Feature Engineering
- **Temporal Features**: Year, month, sinusoidal encoding for seasonality
- **Lag Features**: 1, 2, 3, 6, and 12-month historical lags
- **Rolling Statistics**: 3, 6, and 12-month rolling mean and standard deviation
- **Automatic NaN Handling**: Proper handling of missing values post-feature creation

### Machine Learning Models
1. **Baseline Linear Regression** with StandardScaler preprocessing
2. **Random Forest Regressor** with hyperparameter optimization via RandomizedSearchCV
   - Parameter tuning for n_estimators, max_depth, and min_samples_split
   - 3-fold cross-validation for robust evaluation

### Statistical Analysis
- T-test for price-based promotion impact analysis
- Feature importance extraction from Random Forest models
- Comprehensive error metrics (RMSE, MAE)

## Project Structure

```
retail-forecast/
├── README.md                      # This file
├── retail_forecast_portuguese.py  # Main forecasting script
└── data/
    └── retail_kaggle/             # Directory for CSV data
        └── mock_kaggle.csv        # Retail transaction data (not included)
```

## Requirements

Python 3.7+

Key Dependencies:
- pandas - Data manipulation and analysis
- numpy - Numerical computations
- scikit-learn - Machine learning models and preprocessing
- matplotlib - Visualization
- seaborn - Statistical visualizations
- scipy - Statistical functions
- joblib - Model persistence

Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib
```

## Usage

### Prepare Data
1. Place your retail CSV file at: `data/retail_kaggle/mock_kaggle.csv`
2. Ensure the CSV contains columns:
   - `data` (date column)
   - `venda` (units sold)
   - `preco` (unit price)
   - Any additional columns will be ignored

### Run the Script
```bash
python retail_forecast_portuguese.py
```

### Output
The script generates:
- **Console Output**: 
  - Data summary and sample rows
  - Model performance metrics (RMSE, MAE)
  - Best Random Forest hyperparameters
  - Statistical significance tests
  - Top 10 feature importances
  - Actionable drivers for product managers

- **Visualizations**:
  - Monthly revenue trend plot
  - Actual vs predicted values for test period
  - Feature importance bar chart

- **Saved Models**:
  - `models/rf_retail_pt.joblib` - Best Random Forest model

## Key Metrics

- **RMSE** (Root Mean Squared Error): Penalizes large errors, useful for revenue forecasting
- **MAE** (Mean Absolute Error): Average absolute deviation, directly interpretable
- **Feature Importance**: Identifies key drivers of revenue variations

## Model Training

### Train/Test Split
- **Chronological Split**: Last 12 months as test set (respects time-series nature)
- **Adaptive Fallback**: If dataset is small, automatically uses 20% for testing
- **Feature Set**: All engineered features excluding date and target variable

### Hyperparameter Optimization
Randomized search with 8 iterations and 3-fold cross-validation

## Insights & Recommendations

After running the script, review:
1. **Feature Importances**: Identify which factors most influence revenue
2. **Statistical Tests**: Understand if promotional prices significantly impact sales
3. **Model Comparison**: Compare Linear Regression vs Random Forest performance
4. **Prediction Patterns**: Analyze residuals for systematic biases

## Column Name Reference

If your CSV uses different Portuguese column names:
- Modify the `date_col`, `units_col`, and `price_col` variables in the script
- Common alternatives:
  - Date: `date`, `Data`, `data_venda`
  - Units: `quantidade`, `qtd`, `units`
  - Price: `valor`, `valor_unitario`, `price`

## Future Enhancements

- [ ] ARIMA/SARIMA models for pure time-series forecasting
- [ ] Ensemble methods combining multiple models
- [ ] External variables (holidays, promotions, marketing spend)
- [ ] Forecasting with confidence intervals
- [ ] Real-time prediction API
- [ ] Dashboard for visualization and monitoring

## Author

Ritvik(@ritvikvr)

## License

MIT License - Feel free to use, modify, and distribute

## Notes

- Ensure sufficient data history (at least 25-30 months recommended for robust seasonal patterns)
- Handle outliers appropriately before running the model
- Review feature engineering assumptions based on your business domain
- Consider data quality issues and missing values in preprocessing
