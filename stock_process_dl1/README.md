# stock_process_dl1

This module contains a complete notebook workflow for stock market next-day direction prediction.

## Main file
- stock_process_dl1.ipynb

## Pipeline summary
1. Download OHLCV data with yfinance
2. Engineer indicators (20/50 MA, daily return, Bollinger Bands)
3. Build binary target for next-day movement
4. Train RandomForestClassifier
5. Evaluate with precision/recall and classification report
6. Generate next 30-day directional forecast table

## Dependencies
- pandas
- yfinance
- scikit-learn

## Notes
- The iterative 30-day forecast is directional and keeps the feature vector stable between steps.
- For production-grade forecasting, use walk-forward validation and richer exogenous features.
