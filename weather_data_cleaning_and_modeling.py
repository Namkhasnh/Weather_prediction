# Import libraries
import pandas as pd 
import numpy as np
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

# Upload and understand the dataset
data = pd.read_csv("/Users/namkhanh/Desktop/Data Analysis/weather forcast/dataset.csv")


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
plt.title('Original data')
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
    ax.set(xlabel='Precipitation', ylabel='Weather', title='Boxplot of Weather vs Precipitation') 

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
print(data.info())

# Extract the feature and target variables from the DataFrame
# Convert the features to integers and exclude the "weather" column
x_data = ((data.loc[:,data.columns!="weather_encoded"]).astype(np.int64)).values[:,0:]
# Get the target variable as an array of values
y_data = data["weather_encoded"].values
print("Phân phối nhãn ban đầu:", Counter(y_data))

plt.figure(figsize=(7, 5), dpi=100)  
sns.countplot(x=y_data, palette="pastel", order=sorted(set(y_data)))
plt.xlabel('Weather_encoded', fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Data after preprocessing")
plt.show()
#split data test and data train

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2, random_state=42, stratify=y_data)

print("Phân phối nhãn ban đầu trong train:", Counter(y_train))
print("Phân phối nhãn ban đầu trong test:", Counter(y_test))

plt.figure(figsize=(7, 5), dpi=100)  
sns.countplot(x=y_train, palette="hls", order=sorted(set(y_train)))
plt.xlabel('Weather_encoded', fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Training Data")
plt.show()

plt.figure(figsize=(7, 5), dpi=100)  # DPI = 100 đảm bảo mỗi inch = 100px
sns.countplot(x=y_test, palette="husl", order=sorted(set(y_test)))
plt.xlabel('Weather_encoded', fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Testing Data")
plt.show()

oversampling_strategy = {0: 200, 1: 250, 3: 150 }
undersampling_strategy = {4: 400, }

oversampler = SMOTE(sampling_strategy=oversampling_strategy, random_state=42)
undersampler = RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=42)
pipeline = Pipeline(steps=[('oversample', oversampler), ('undersample', undersampler)])
x_train, y_train = pipeline.fit_resample(x_train, y_train)

sns.countplot(x=y_train, palette="hls", order=sorted(set(y_train)))
plt.xlabel('Weather_encoded', fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("New Training Data")
plt.show()

print("Phân phối nhãn sau cân bằng:", Counter(y_train))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Huấn luyện mô hình
rfc = RandomForestClassifier(random_state=42)
rfc.fit(x_train, y_train)
y_pred_rfc = rfc.predict(x_test)
# Dự đoán trên tập test
rfc_score = rfc.score(x_test, y_test)
print("RFC Accuracy:", rfc_score)
# Đánh giá hiệu quả mô hình
print(classification_report(y_test, y_pred_rfc, digits=4))


# K_nearest neighbor classifier (k =7)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Huấn luyện KNN với dữ liệu chuẩn hóa
knn = KNeighborsClassifier(n_neighbors=7, p=2, weights='distance')
knn.fit(x_train_scaled, y_train)
y_pred_knn = knn.predict(x_test_scaled)

# Đánh giá
knn_score = knn.score(x_test_scaled, y_test)
print("KNN Accuracy:", knn_score)
print(classification_report(y_test, y_pred_knn, digits=4))
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
    print(f"Decision Tree (max_depth={depth}) Accuracy: {dec_score}")

dec = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=15, random_state=0)
dec.fit(x_train, y_train)
dec_score = dec.score(x_test, y_test)
dec_predict = dec.predict(x_test)
print("Decision Tree Accuracy: ", dec_score)
print(classification_report(y_test, dec_predict, digits=4))

dec = DecisionTreeClassifier(max_depth=7, max_leaf_nodes=15, random_state=0)
dec.fit(x_train, y_train)
dec_score = dec.score(x_test, y_test)
dec_predict = dec.predict(x_test)
print("Decision Tree Accuracy: ", dec_score)
print(classification_report(y_test, dec_predict, digits=4))
# Logistic regression

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(x_train, y_train)
lg_score = lg.score(x_test, y_test)
print("Logistic Accuracy : ", lg_score)
# The model above only gives 0.6129 accuracy, which is an extremely low result.

from xgboost import XGBClassifier
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=10,
    
    random_state=42
)

xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# In this script, we have built and evaluated several models.
# We selected the Decision Tree model with max_depth = 4 as it achieved the best accuracy (0.83).
# In the next script, "predict_weather_from_input.py", we will utilize this trained model to make predictions on new inputs.
