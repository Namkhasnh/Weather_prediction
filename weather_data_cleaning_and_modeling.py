# Import libraries
import pandas as pd 
import numpy as np
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt
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


# Create a dictionary that maps the encoded values to the actual names
weather_names = dict(zip(lc.classes_, lc.transform(lc.classes_)))
# Plot the count of each unique value in the weather column with actual names on the labels
sns.countplot(x='weather_encoded', data=data, palette='hls', hue='weather_encoded', legend=False)
plt.xticks(ticks=range(len(weather_names)), labels=list(weather_names.values()))
plt.show()
# Get the value counts of each unique value in the weather column
weather_counts = data['weather'].value_counts()
# Print the percentage of each unique value in the weather column
for weather, count in weather_counts.items():
    percent = (count / len(data)) * 100
    print(f"Percent of {weather.capitalize()}: {percent:.2f}%")
# From the above graph and analysis, we can see that our dataset contains mostly rain and sun weather conditions
# It is approximately the same when accounting for 43.3% of the dataset.

# Use a context manager to apply the default style to the plot
with plt.style.context('default'):
    
    # Create a figure with the specified size and an axis object
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot a boxplot with the given data, using the specified x and y variables, color palette, and axis object
    sns.boxplot(x="precipitation", y="weather", data=data, palette="winter", ax=ax)
    
    # Optional: set axis labels and title if desired
    ax.set(xlabel='Precipitation', ylabel='Weather', title='Boxplot of Weather vs. Precipitation') 
plt.show()
#From the boxplot between weather and precipitation above, the value of rain has many positive outliers,
#You can try with all case between (weather, temp_max),(weather, temp_min),(weather,wind)

# Handing null values
null_count = data.isnull().sum()
print(null_count)
# By looking above details, we can conclude that there are no NULL values ​


# Create a data copy
data_copy = data.copy()
data_copy=data_copy.drop("weather" ,axis=1)
#remove outlier points and infinite values
date_col = data_copy['date']
data_cols = data_copy.drop(columns=['date'])

Q1_date = data_cols.quantile(0.25)
Q3_date = data_cols.quantile(0.75)
IQR_date = Q3_date - Q1_date

outlier_condition = ~((data_cols < (Q1_date - 1.5 * IQR_date)) | (data_cols > (Q3_date + 1.5 * IQR_date))).any(axis=1)

data = data_copy[outlier_condition]

data.date = pd.to_datetime(data.date)
print(data)

# Extract the feature and target variables from the DataFrame
# Convert the features to integers and exclude the "weather" column
x_data = ((data.loc[:,data.columns!="weather_encoded"]).astype(np.int64)).values[:,0:]
# Get the target variable as an array of values
y_data = data["weather_encoded"].values

#split data test and data train
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.1,random_state=2)

### Model training

# K_nearest neighbor classifier (k =7)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7, p = 2, weights = 'distance' )
knn.fit(x_train, y_train)
knn_score = knn.score(x_test, y_test)
print("KNN Accuracy:", knn_score)
# The K-Neighbor Nearest Classifier model has reduced the accuracy to only 0.7 

# Decision Tree

from sklearn.tree import DecisionTreeClassifier
# Create a list of max depth values to try
max_depth_range = list(range(1, 8))
# Train and evaluate a decision tree model with varying max depth values
for depth in max_depth_range:
    dec = DecisionTreeClassifier(max_depth=depth, max_leaf_nodes=15, random_state=0)
    dec.fit(x_train, y_train)
    dec_score = dec.score(x_test, y_test)
    print("Decision Tree Accuracy: ", dec_score)
# Decision Tree model with confidence 0.83 with parameter max_depth = 4.
# This is the model with the best reliability among them. the results we have.

# Logistic regression

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(x_train, y_train)
lg_score = lg.score(x_test, y_test)
print("Logistic Accuracy : ", lg_score)
# The model above only gives 0.6129 accuracy, which is an extremely low result.


# In this script, we have built and evaluated several models.
# We selected the Decision Tree model with max_depth = 4 as it achieved the best accuracy (0.83).
# In the next script, "predict_weather_from_input.py", we will utilize this trained model to make predictions on new inputs.
