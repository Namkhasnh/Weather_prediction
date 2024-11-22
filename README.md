<p align="center">
 <h1 align="center">Weather_Forecasting_by_MachineLearning</h1>
</p>

## Introduction
This project uses a **Decision Tree,** **K-Nearest Classifier,** **Logistic Regression** model to predict the weather based on features such as: date, precipitation, high temperature, low temperature, and wind speed. You can input new weather data and get the predicted weather result.

## Model Accuracy
The current Decision Tree model has an accuracy of 83% on the test data. While this model works well, it is still a work in progress and can be further optimized.

## Dataset
The dataset used for training my model can be found at [Weather Prediction dataset](https://www.kaggle.com/datasets/ananthr1/weather-prediction/data).

This dataset contains weather data for Seattle, USA, recorded on a daily basis for approximately 4 years. It includes the following weather conditions:

- Observed date
- High and low temperatures
- Precipitation
- Wind speed
- Weather conditions for each day based on these factors

The data is used to predict weather conditions based on these features.

You can download this in my file "dataset.csv"


## Project Structure
The project includes the following files:
- **weather_data_cleaning_and_modeling.py**: File for data processing and building the model.
- **predict_weather_from_input.py**: Library for numerical operations.
- **weather_app_tkinter.py**: Simple user interface (GUI) using Tkinter where you can input data and get a weather prediction result.
- **dataset.csv**: Input data containing weather-related information (needs to be present before running the program).

## Additional Instructions
To enhance the application, you can add features like displaying detailed predictions or saving the results.

## Requirements
* **pandas**
* **numpy**
* **sklearn** 
* **matplotlib**
* **seaborn**
* **tkinter**

## Model Status & Feedback
While the current model works well, it’s not fully optimized yet. I am constantly looking for ways to improve its performance, such as experimenting with different algorithms, feature engineering, or hyperparameter tuning.

I would love to hear your feedback or suggestions! Feel free to contribute, whether it’s by improving the model, optimizing the code, or suggesting new ideas.
