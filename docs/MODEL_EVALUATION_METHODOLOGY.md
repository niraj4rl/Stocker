# Model Evaluation Methodology

## Purpose
Stocker evaluates models rigorously to ascertain their predictive performance, directional accuracy, and real-world trading viability. We distinguish clearly between statistical prediction accuracy and trading strategy returns.

## A. Regression Metrics
Regression models predict the precise future percentage return. We compute errors on the **Price Scale (Currency)** by inverse-transforming returns using the most recent close.

### 1. Mean Absolute Error (MAE)
- **Purpose:** Measures the average absolute prediction error in currency units.
- **Formula:** $\text{MAE} = \frac{1}{n} \sum |Y_i - \hat{Y}_i|$ where $Y$ is the actual target price.
- **Interpretation:** Lower is better. Tells you the average absolute deviation of the prediction from the real price.
- **Limitation:** Does not heavily penalize large outliers.

### 2. Mean Squared Error (MSE)
- **Purpose:** Heavily penalizes large prediction errors.
- **Formula:** $\text{MSE} = \frac{1}{n} \sum (Y_i - \hat{Y}_i)^2$
- **Interpretation:** Lower is better. More sensitive to outliers than MAE.
- **Limitation:** In units of price squared, which is difficult to interpret directly.

### 3. Root Mean Squared Error (RMSE)
- **Purpose:** Like MSE but in the same units as the price.
- **Formula:** $\text{RMSE} = \sqrt{\text{MSE}}$
- **Interpretation:** Lower is better. Represents the typical error magnitude while strongly penalizing large errors.

### 4. R-squared ($R^2$)
- **Purpose:** Measures performance relative to a constant-mean prediction baseline.
- **Formula:** $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ (computed on returns for meaningful scale).
- **Interpretation:** Can be negative on test data. A higher $R^2$ indicates the model captures variance better than the mean, but does not strictly imply profitability.
- **Limitation:** Extremely sensitive to the evaluation period's trend.

### 5. Mean Absolute Percentage Error (MAPE)
- **Purpose:** Measures error magnitude as a percentage of the actual price.
- **Formula:** $\text{MAPE} = \frac{1}{n} \sum \left|\frac{Y_i - \hat{Y}_i}{Y_i}\right|$
- **Interpretation:** Lower is better. Useful for comparing error sizes across stocks with vastly different prices.

### 6. Directional Accuracy
- **Purpose:** Percentage of times the predicted sign matches the actual return sign.
- **Formula:** $\frac{1}{n} \sum I(\text{sign}(y_i) = \text{sign}(\hat{y}_i))$
- **Interpretation:** Higher is better. Directly translates to "hit rate" if trading long/short on the prediction.

## B. Classification Metrics
Classification models predict market directions (`strong_up`, `neutral`, `strong_down`).

### 1. Accuracy
- **Purpose:** Overall fraction of correctly classified directions.
- **Limitation:** Misleading in highly imbalanced regimes (e.g., strong bull market where `strong_up` dominates).

### 2. Precision and Recall
- **Purpose:** Precision measures the exactness (how many `strong_up` predictions were actually `strong_up`), while Recall measures completeness.
- **Interpretation:** We use a weighted average across all classes to account for imbalances.

### 3. F1-Score
- **Purpose:** Harmonic mean of precision and recall. Best single metric for imbalanced directional classification.

### 4. ROC-AUC
- **Purpose:** Assesses the model's ability to rank probabilities across classes (computed via one-vs-rest average).
- **Interpretation:** 0.5 is random guessing, 1.0 is perfect ranking.

### 5. Confusion Matrix
- **Purpose:** Shows exact class-by-class misclassifications to identify biases (e.g., predicting `strong_up` when the actual is `strong_down`).

## C. Financial Performance Metrics (Backtesting)
Models are evaluated on their generated trading signals to simulate actual trading performance.

### 1. Sharpe Ratio
- **Purpose:** Measures excess return relative to return volatility.
- **Formula:** $\frac{\text{Mean}(R - R_f)}{\text{StdDev}(R - R_f)} \times \sqrt{252}$ (assuming 252 trading days per year).
- **Interpretation:** Higher is better. Risk-adjusted return metric. We assume a 0% baseline risk-free rate for comparative simplicity unless configured otherwise.

### 2. Maximum Drawdown
- **Purpose:** Measures the largest peak-to-trough drop in strategy equity.
- **Formula:** $\text{Min} \left( \frac{\text{Equity}_t - \text{Peak}_t}{\text{Peak}_t} \right)$
- **Interpretation:** Lower magnitude is better. Reflects downside risk.

### 3. Profit Factor
- **Purpose:** Ratio of gross gains to gross losses.
- **Formula:** $\frac{\sum \text{Gains}}{|\sum \text{Losses}|}$
- **Interpretation:** > 1 indicates profitability.

### 4. Baseline Comparisons
Models are implicitly compared against a **Buy-and-Hold** benchmark, which serves to check whether the strategy adds value beyond passive market exposure.
