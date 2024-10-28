Project Overview

This project implements a baseline Long Short-Term Memory (LSTM) model to predict stock prices of technology sector companies. The data is sourced from a CSV file containing daily stock data for various tickers. This model is designed to handle multiple stock tickers, normalize the data, and create sequences for training the LSTM. The model includes Dropout layers to prevent overfitting and uses Early Stopping to halt training once validation loss stops improving.

Requirements

To run this project, you need the following Python libraries:

	•	pandas
	•	tensorflow
	•	scikit-learn
	•	joblib
	•	numpy

You can install the dependencies by running:
pip install pandas tensorflow scikit-learn joblib numpy

Data Preprocessing

	1.	The project starts by loading daily stock data from the technology_stocks_daily_data_sorted.csv file.
	2.	The data is separated by ticker, with each ticker’s data reindexed to cover a uniform date range, and missing values are replaced with 0.
	3.	Features (open, high, low, close, adjusted_close, volume) are normalized using MinMaxScaler for each ticker.
	4.	The normalized data is then combined back into a single dataset for training.

Sequence Creation

The function create_sequences is responsible for creating sequences of length seq_length (in this case, 60) for the input data. Each sequence consists of stock price and volume data, and the target label is the close price of the stock at the end of each sequence.

LSTM Model Architecture

The model architecture is as follows:

	•	Input Layer: Sequences of stock data with a sequence length of 60.
	•	LSTM Layer 1: LSTM with 50 units, returning sequences.
	•	Dropout Layer 1: Dropout with a rate of 0.2 to prevent overfitting.
	•	LSTM Layer 2: LSTM with 50 units.
	•	Dropout Layer 2: Dropout with a rate of 0.2 to further prevent overfitting.
	•	Dense Output Layer: A fully connected layer with one unit to predict the stock’s closing price.

Model Compilation

The model is compiled using:

	•	Optimizer: adam
	•	Loss Function: Mean Squared Error (mse)

Training

	•	The model is trained for 100 epochs with a batch size of 32, and 20% of the data is used for validation.
	•	Early Stopping is applied to halt training if the validation loss doesn’t improve for 10 consecutive epochs. This helps prevent overfitting and reduces training time.

Model Saving

	•	The trained model is saved as tech_model.keras.
	•	The MinMaxScaler used for normalization is saved as scaler.save using joblib.

 Extending the Baseline Model

You can modify this baseline model by changing various parameters, such as:

	•	Batch size: The number of samples processed before the model is updated.
	•	Number of LSTM units: The number of memory units in each LSTM layer.
	•	Dropout rate: The fraction of neurons dropped during training to prevent overfitting.
	•	Number of layers: You can add more LSTM or Dense layers to the model architecture.

Files

	•	technology_stocks_daily_data_sorted.csv: The input data file containing daily stock prices for various tickers.
	•	tech_model.keras: The trained LSTM model.
	•	scaler.save: The MinMaxScaler used to normalize the data.
