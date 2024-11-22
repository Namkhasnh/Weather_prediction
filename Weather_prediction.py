# Import libraries
import pandas as pd 
import itertools 
import numpy as np
import matplotlib as plt
import re
import scipy 
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import pearsonr, ttest_ind 
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

# Upload and understand the dataset
data = pd.read_csv("/Users/namkhanh/Desktop/Data Analysis/weather forcast/dataset.csv")
print(data.head())
print(data.shape)

# There are 6 columns with a total of 1461 rows according to the observations in the dataset



from sklearn.preprocessing import LabelEncoder
# Create a label encoder object
lc = LabelEncoder()
# Encode weather categorical column in data into integer form
data["weather_encoded"]=lc.fit_transform(data["weather"])
# Create a data cope
data_copy = data.copy()
data_copy=data_copy.drop("weather" ,axis=1)
#remove outlier points and infinite values
date_col = data_copy['date']
data_cols = data_copy.drop(columns=['date'])

Q1_date = data_cols.quantile(0.25)
Q3_date = data_cols.quantile(0.75)
IQR_date = Q3_date - Q1_date

outlier_condition = ~((data_cols < (Q1_date - 1.5 * IQR_date)) | (data_cols > (Q3_date + 1.5 * IQR_date))).any(axis=1)

df = data_copy[outlier_condition]
df.precipitation=np.sqrt(data.precipitation)
df.wind=np.sqrt(data.wind)

data.date = pd.to_datetime(data.date)
print(data)

# Extract the feature and target variables from the DataFrame
# Convert the features to integers and exclude the "weather" column
x_data = ((data.loc[:,data.columns!="weather_encoded"]).astype(np.int64)).values[:,0:]
# Get the target variable as an array of values
y_data = data["weather_encoded"].values

#split data test and data train
x_train_df3,x_test_df3,y_train_df3,y_test_df3 = train_test_split(x_df3,y_df3,test_size=0.1,random_state=2)

