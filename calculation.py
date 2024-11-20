import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
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

    # Load the CSV file into a DataFrame
    df3 = pd.read_csv("/Users/namkhanh/Desktop/Data Analysis/weather forcast/dataset.csv")
    lc = LabelEncoder()
    df3["weather_encoded"] = lc.fit_transform(df3["weather"])


    df3_copy = df3.copy()
    df3_copy=df3_copy.drop("weather" ,axis=1)
    #*remove outlier points and infinite values
    date_col = df3_copy['date']
    data3_cols = df3_copy.drop(columns=['date'])

    # Tính toán phần trăm thứ nhất và thứ ba, cũng như IQR
    Q11_date = data3_cols.quantile(0.25)
    Q33_date = data3_cols.quantile(0.75)
    IQR3_date = Q33_date - Q11_date

    # Xác định giá trị ngoại lai
    outlier_condition3 = ~((data3_cols < (Q11_date - 1.5 * IQR3_date)) | (data3_cols > (Q33_date + 1.5 * IQR3_date))).any(axis=1)

    # Lọc ra giá trị ngoại lai và giữ lại cột month
    df3 = df3_copy[outlier_condition3]
    df3.precipitation=np.sqrt(df3.precipitation)
    df3.wind=np.sqrt(df3.wind)

    df3.date = pd.to_datetime(df3.date)
    # print(df3)

    x_df3 = ((df3.loc[:,df3.columns!="weather_encoded"]).astype(np.int64)).values[:,0:]
    y_df3 = df3["weather_encoded"].values


    x_train_df3,x_test_df3,y_train_df3,y_test_df3 = train_test_split(x_df3,y_df3,test_size=0.1,random_state=2)
    dec_df3 = DecisionTreeClassifier(max_depth=4, max_leaf_nodes=15, random_state=0)
    dec_df3.fit(x_train_df3, y_train_df3)
    dec_score_df3 = dec_df3.score(x_test_df3, y_test_df3)
      
    
    
    df_new = pd.DataFrame([new_input], columns=["date", "precipitation", "high_temperature", "low_temperature", "wind"])

    df_new["precipitation"] = np.sqrt(df_new["precipitation"])
    df_new["wind"] = np.sqrt(df_new["wind"])
    df_new["date"] = pd.to_datetime(df_new["date"])


    df_new["date"] = df_new["date"].apply(lambda x: int(x.timestamp()))


    x_new = df_new.loc[:, df_new.columns != "weather"].astype(np.int64).values


    predictions = dec_df3.predict(x_new)


    predicted_weather = lc.inverse_transform(predictions)

    result = predicted_weather[0]

    return result