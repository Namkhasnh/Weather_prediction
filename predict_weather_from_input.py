import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def process_inputs(inputs):
    # Process the inputs
    result = 1

    # Assuming you have a new input as a list
    new_input = [
        inputs[0],
        float(inputs[1]),
        float(inputs[2]),
        float(inputs[3]),
        float(inputs[4])
    ]

    data = pd.read_csv("/Users/namkhanh/Desktop/Data Analysis/weather forcast/dataset.csv")
    lc = LabelEncoder()
    data["weather_encoded"] = lc.fit_transform(data["weather"])

    data_copy = data.copy()
    data_copy=data_copy.drop("weather" ,axis=1)

    date_col = data_copy['date']
    data_cols = data_copy.drop(columns=['date'])

    Q1_date = data_cols.quantile(0.25)
    Q3_date = data_cols.quantile(0.75)
    IQR_date = Q3_date - Q1_date

    outlier_condition = ~((data_cols < (Q1_date - 1.5 * IQR_date)) | (data_cols > (Q3_date + 1.5 * IQR_date))).any(axis=1)

    data = data_copy[outlier_condition]

    data.date = pd.to_datetime(data.date)
    
    x_data = ((data.loc[:,data.columns!="weather_encoded"]).astype(np.int64)).values[:,0:]
    y_data = data["weather_encoded"].values


    x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.1,random_state=2)
    dec = DecisionTreeClassifier(max_depth=4, max_leaf_nodes=15, random_state=0)
    dec.fit(x_train, y_train)

    df_new = pd.DataFrame([new_input], columns=["date", "precipitation", "high_temperature", "low_temperature", "wind"])

    df_new["date"] = pd.to_datetime(df_new["date"])

    df_new["date"] = df_new["date"].apply(lambda x: int(x.timestamp()))


    x_new = df_new.loc[:, df_new.columns != "weather"].astype(np.int64).values


    predictions = dec.predict(x_new)


    predicted_weather = lc.inverse_transform(predictions)

    result = predicted_weather[0]

    return result
