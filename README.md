# NiftyVision-AI-Powered-Forecasting-for-Bank-Nifty

NiftyVision is a project focused on forecasting the closing prices of Bank Nifty using historical data from Yahoo Finance and advanced deep learning techniques. Our goal is to capture the inherent market variability and achieve high-precision predictions with low error metrics.

---

## 1. Problem Statement

In today’s volatile financial markets, accurately forecasting the price movement of key indices such as Bank Nifty is challenging but critical for investment decisions and risk management. The primary problem we address in this project is:

- **Objective:** Develop a predictive model that accurately forecasts the next day's closing price of Bank Nifty.
- **Challenge:** Capture the complex temporal dynamics and variability inherent in financial time series while keeping forecasting errors minimal.

Our target is to reduce the mean absolute error (MAE) and root mean squared error (RMSE) to a level that is competitive (ideally in the low hundreds or as a low percentage of the index value).

---

## 2. Our Approach

### Data Acquisition & Feature Engineering
- **Data Source:** We use Yahoo Finance (via the `yfinance` library) to download daily historical data for Bank Nifty.
- **Technical Indicators:**  
  - **Moving Averages (MA10, MA20, MA50):** These smooth out price fluctuations and highlight trends.  
  - **RSI (Relative Strength Index):** Measures momentum and identifies overbought/oversold conditions.  
  - **Volatility:** Calculated as the 10-day rolling standard deviation to capture market variability.  
  - **Log Returns:** Helps to capture relative changes in price.
  
*Graph Image – Technical Indicators:*  
![Data Indicators Graph](https://your-image-link.com/data_indicators.png)  
_This graph shows the Close price along with MA10, MA20, MA50, RSI, and Volatility over time._

### Model Architecture
- **Transformer-Based Model with Residual Connection:**  
  We project the input features into a higher-dimensional space and feed them into several Transformer encoder layers. A residual connection is added (by summing the encoder’s final output with the projected input’s last timestep) to help preserve the original variance of the signal. This architecture allows the model to capture long-range dependencies and complex patterns in the data.
  
*Graph Image – Model Architecture Diagram:*  
![Model Architecture Diagram](https://your-image-link.com/model_architecture.png)  
_This diagram illustrates the Transformer model workflow, including input projection, Transformer encoding, residual connection, and final prediction._

### Training Process
- **Loss Function & Optimizer:**  
  The model is trained using Mean Squared Error (MSE) loss with the Adam optimizer.
- **Training Details:**  
  The training loop runs for a fixed number of epochs (e.g., 50), and the training loss is monitored.
  
*Graph Image – Training Loss Curve:*  
![Training Loss Graph](https://your-image-link.com/training_loss.png)  
_This plot shows the reduction in training loss over epochs, indicating the model’s learning progress._

---

## 3. Results & Graphs

After training, the model is evaluated on both the training and test datasets. We plot the actual versus predicted prices on a single graph for easy comparison, and compute key error metrics:

- **Test MAE:** ~286 points  
- **Test RMSE:** ~363 points

*Graph Image – Actual vs. Predicted Prices:*  
![Actual vs. Predicted Graph](https://your-image-link.com/actual_vs_predicted.png)  
_This graph shows the actual Bank Nifty closing prices (blue) and the model’s predictions (red) over time, with a clear indication of the train/test split._

*Graph Image – Error Distribution:*  
![Error Distribution Graph](https://your-image-link.com/error_distribution.png)  
_This histogram displays the distribution of prediction errors (actual - predicted) for the test set._

---

## 4. Features Explanation

Our feature set is designed to provide the model with a rich context of the market dynamics. The features used include:

- **Close Price:** The raw closing price, representing the main target variable.
- **MA10:** A 10-day moving average that smooths short-term fluctuations.
- **RSI:** An indicator that measures momentum to signal potential reversals or continuations.
- **LogReturn:** The logarithmic return captures percentage changes, offering a more stationary view of price movements.

Normalization is applied independently to each feature (using MinMaxScaler), ensuring that each input is scaled between 0 and 1, which helps in training the neural network.

---

## 5. Conclusion

NiftyVision successfully integrates advanced feature engineering with a state-of-the-art Transformer-based model, enhanced by a residual connection. The project achieves a low mean absolute error and root mean squared error, demonstrating its ability to capture the inherent variability of Bank Nifty prices. This model provides a robust foundation for future extensions, such as incorporating additional technical indicators, hybrid model architectures, or ensemble methods to further refine predictions.

Our approach not only meets the current forecasting needs but also sets the stage for continuous improvements and real-world applications in risk management and investment strategy.

---

