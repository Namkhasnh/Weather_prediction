#! Import Library

import itertools 
import pandas as pd
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


data = pd.read_csv("/Users/namkhanh/Desktop/Data Analysis/weather forcast/dataset.csv")

# print(data.head())

# print(data.shape)
#! Visualizing the dataset
#*Label

le = LabelEncoder()
data['weather_encoded'] = le.fit_transform(data['weather'])

weather_names= dict(zip(le.classes_,le.transform(le.classes_)))

# sns.countplot(x='weather_encoded', data=data, palette='hls', hue='weather_encoded', legend=False)
# # Cập nhật lại các nhãn trên trục x
# plt.xticks(ticks=range(len(weather_names)), labels=list(weather_names.values()))


weather_counts = data['weather'].value_counts()

for weather,count in weather_counts.items():
    percent = (count/ len (data))* 100
    # print(f"Percent of {weather.capitalize ()}:{percent:.2f}%")

data_copy = data.copy()
# *Figures
# sns.set(style="darkgrid")

# vartables =["precipitation", "temp_max", "temp_min", "wind"]
# colors=["green","red","yellow","skyblue"]

# fig,axs = plt.subplots(2,2,figsize=(10,8))

# for i,var in enumerate(vartables):
#     sns.histplot(data=data , x=var, kde = True, ax= axs[i//2,i%2],color=colors[i])
    


#!Find  external value and divility 

# with plt.style.context('default'):
#     fig,ax = plt.subplots(figsize=(12,6))
#     sns.boxplot(x="precipitation",y="weather",data=data, palette="winter",ax=ax)
#     ax.set(xlabel='precipitation',ylabel="Weather",title="")
    
# with plt.style.context('default'):
#     fig,ax = plt.subplots(figsize=(12,6))
#     sns.boxplot(x="temp_max",y="weather",data=data, palette="summer",ax=ax)
#     ax.set(xlabel='temp_max',ylabel="Weather",title="")

# with plt.style.context('default'):
#     fig,ax = plt.subplots(figsize=(12,6))
#     sns.boxplot(x="temp_min",y="weather",data=data, palette="spring",ax=ax)
#     ax.set(xlabel='temp_min',ylabel="Weather",title="") 

# with plt.style.context('default'):
#     fig,ax = plt.subplots(figsize=(12,6))
#     sns.boxplot(x="wind",y="weather",data=data, palette="autumn",ax=ax)
#     ax.set(xlabel='wind',ylabel="Weather",title="")


# *Correlation and t test p test
# corr= data["precipitation"].corr(data["temp_max"])
# ttest,pvalue= stats.ttest_ind(data["precipitation"],data["temp_max"])

# with plt.style.context('default'):
#     ax= data.plot("precipitation","temp_max",style="o")
#     ax.set_title("Scatter Plot of Precipitation vs Maximum Temperature")
#     ax.set_xlabel("Precipitation")
#     ax.set_ylabel("Maximum Temperature")

#     textstr = f'Pearson Correlation {corr:.2f}\nT-Test P-Value:{pvalue:.2e}'
#     ax.text(0.05,0.95,textstr,transform=ax.transAxes, fontsize=12,verticalalignment="top",bbox=dict(facecolor="white",edgecolor="none",alpha=0.8))



# fig, ax = plt.subplots(figsize=(8,6))
# ax.scatter(x=data["wind"],y=data["temp_max"],marker='o',s=50,alpha=0.8,color="blue")

# corr,p_value = np.corrcoef(data["wind"],data["temp_max"])[0,1],np.mean(np.abs(stats.ttest_ind(data["wind"],data["temp_max"])[1]))
# ax.text(0.95,0.95,f"Pearson correlation:{corr:.2f}\nT Test and P value:{p_value:.2f}",transform =ax.transAxes,ha="right",va='top',fontsize=12)
# ax.set(xlabel="Wind",ylabel="Maximum Temperature")
# ax.set(title="Scatter plot of Wind vs Maximum Temperature")



# fig, ax = plt.subplots(figsize=(8,6))
# ax.scatter(x=data["temp_max"],y=data["temp_min"],marker='o',s=50,alpha=0.8,color="orange")

# corr,p_value = np.corrcoef(data["temp_max"],data["temp_min"])[0,1],np.mean(np.abs(np.subtract(data["temp_max"],data["temp_min"])))
# ax.text(0.45,0.95,f"Pearson correlation:{corr:.2f}\nT Test and P value:{p_value:.2f}",transform =ax.transAxes,ha="right",va='top',fontsize=12)
# ax.set(xlabel="Maximum Temperature",ylabel="Minimum Temperature")
# ax.set(title="Scatter plot of Maximum vs Minimum Temperature")



#*Handing null values
# null_count =data.isnull().sum()
# print(null_count)

#!data processing and cleaning
# Xóa cột đầu tiên bằng cách sử dụng chỉ mục
data_copy = data_copy.drop(data.columns[0], axis=1)


df=data_copy.drop("weather" ,axis=1)


#*remove outlier points and infinite values
# Bước 1: Tính toán Q1, Q3 và IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1


# Sử dụng điều kiện để xác định các giá trị nằm ngoài phạm vi và đặt chúng là NaN
df_outlier_filtered = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]


df = df_outlier_filtered.dropna()

#*Handing different distributions________
df["precipitation"] = np.sqrt(df["precipitation"])
df["wind"] = np.sqrt(df["wind"])

sns.set(style="darkgrid")

vartables =["precipitation", "temp_max", "temp_min", "wind"]
colors=["green","red","yellow","skyblue"]

# fig,axs = plt.subplots(2,2,figsize=(10,8))

# for i,var in enumerate(vartables):
#     sns.histplot(data=df , x=var, kde = True, ax= axs[i//2,i%2],color=colors[i])



x = ((df.loc[:,df.columns!="weather_encoded"]).astype(int)).values[:,0:]
y = df["weather_encoded"].values
# print(df.weather_encoded.unique())

#^split data test and data train


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)


#!Model Training
#*K-Nearest Neighbor classifier

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

knn_score = knn.score(x_test,y_test)
print(f"KNN Score: {knn_score:.2f}")
#^create a confusion matrix
y_pred_knn = knn.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred_knn)
# print(f"Confusion Matrix:\n{conf_matrix}")

# print('KNN classification report\n')
# print(classification_report(y_test, y_pred_knn,zero_division=0))


#*Decision Tree 

from sklearn.tree import DecisionTreeClassifier
max_depth_range = range(1,8)

for depth in max_depth_range:
    dec = DecisionTreeClassifier(max_depth=depth,max_leaf_nodes=15,random_state=0)
    dec.fit(x_train, y_train)
    dec_score = dec.score(x_test,y_test)
    # print(f'Decision Tree {depth}:{dec_score}')

y_pred_dec = dec.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred_dec)


# print(f'Decision Tree classification report\n')
# print(classification_report(y_test, y_pred_dec, zero_division=0))

#*Logistic Regression

from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()
lg.fit(x_train, y_train)
lg_score = lg.score(x_test,y_test)
# print(f'Logistic Regression Score: {lg_score}')

y_pred_lg = lg.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred_lg)

# print(f'Logistic Regression classification report\n')
# print('Logistic Regression\n',classification_report(y_test,y_pred_lg, zero_division=0))

#*Model reliability comparison chart
# models=["KNN","DECISION TREE","LOGISTIC REGRESSION"]
# accuracies =[knn_score,dec_score,lg_score]
# sns.set_style("darkgrid")
# plt.figure(figsize=(22,8))
# ax=sns.barplot(x=models,y=accuracies,palette="mako",saturation=1.5)
# plt.xlabel("Models",fontsize=20)
# plt.ylabel("Accuracy",fontsize=20)
# plt.title("Model Reliability Comparison",fontsize=20)
# plt.xticks(fontsize=11,horizontalalignment="center",rotation=8)
# plt.yticks(fontsize=13)

# for p in ax.patches:
#     ax.annotate(f'{p.get_height():.2%}',(p.get_x()+ p.get_width()/2,p.get_height()*1.02),ha="center",fontsize='x-large')

# # plt.show()

#*build model in case of keeping date
df_date = pd.read_csv("/Users/namkhanh/Desktop/Data Analysis/weather forcast/dataset.csv")
# print(data.head())
# print(df_date.info())
lc_date = LabelEncoder()

# Encode the "weather" column of the DataFrame and replace it with the encoded values
df_date["weather_encoded"] = lc_date.fit_transform(df_date["weather"])
#^ rename date to month
df_date.date = pd.to_datetime(df_date.date).dt.month
df_date = df_date.rename(columns={'date':'month'})
# print(df_date.head())

# sns.set(style='darkgrid')
# fig,axs = plt.subplots(figsize=(10,8))
# plot= sns.histplot(data=df_date, x="month", kde=True,color='green')
# plt.show()
df_date_copy = df_date.copy()
df_date_copy=df_date_copy.drop("weather" ,axis=1)

#*remove outlier points and infinite values
month_col = df_date_copy['month']
data_cols = df_date_copy.drop(columns=['month'])

# Tính toán phần trăm thứ nhất và thứ ba, cũng như IQR
Q1_date = data_cols.quantile(0.25)
Q3_date = data_cols.quantile(0.75)
IQR_date = Q3_date - Q1_date

# Xác định giá trị ngoại lai
outlier_condition = ~((data_cols < (Q1_date - 1.5 * IQR_date)) | (data_cols > (Q3_date + 1.5 * IQR_date))).any(axis=1)

# Lọc ra giá trị ngoại lai và giữ lại cột month
df_date = df_date_copy[outlier_condition]

# Kết quả cuối cùng
# print(df_date)

df_date.precipitation=np.sqrt(df_date.precipitation)
df_date.wind=np.sqrt(df_date.wind)
# sns.set(style="darkgrid")
# fig, axs = plt.subplots(2, 3, figsize=(10, 10))
# print(df_date.columns)

plots = ["month", "precipitation", "temp_max", "temp_min", "wind"]

# for i, plot in enumerate(plots):
    # sns.histplot(data=df_date, x=plot, kde=True, ax=axs[i//3, i%3], color=["blue", "green", "red", "skyblue", "orange"][i])

x_date = ((df_date.loc[:,df_date.columns!="weather_encoded"]).astype(int)).values[:,0:]
y_date = df["weather_encoded"].values
# print(df.weather_encoded.unique())

#^split data test and data train


x_train_date, x_test_date, y_train_date, y_test_date = train_test_split(x_date, y, test_size=0.1, random_state=2)

#!Model Training
#*K-Nearest Neighbor classifier
knn_date = KNeighborsClassifier()
knn_date.fit(x_train_date, y_train_date)

knn_date_score = knn_date.score(x_test_date, y_test_date)
# print("KNN Accuracy (with month column):", knn_date_score)
y_pred_knn_date = knn_date.predict(x_test_date)

# Compute the confusion matrix for the KNN model predictions
conf_matrix_knn_date = confusion_matrix(y_test_date, y_pred_knn_date)

# Print the confusion matrix to the console
# print("Confusion Matrix (with month column)")
# print(conf_matrix_knn_date)
# print('KNN (with month column)\n',classification_report(y_test_date,y_pred_knn_date, zero_division=0))


#*Decision Tree
max_depth_range_date = list(range(1, 8))


for depth in max_depth_range_date:
    dec_date = DecisionTreeClassifier(max_depth=depth, max_leaf_nodes=15, random_state=0)
    dec_date.fit(x_train_date, y_train_date)
    dec_date_score = dec_date.score(x_test_date, y_test_date)
    # print("Decision Tree Accuracy (with month column) for max_depth=", depth, ": ", dec_date_score)


y_pred_dec_date = dec_date.predict(x_test_date)

conf_matrix_dec_date = confusion_matrix(y_test_date, y_pred_dec_date)


# print("Confusion Matrix (with month column)")
# print(conf_matrix_dec_date)

# print('Decision Tree (with month column)\n',classification_report(y_test_date,y_pred_dec_date, zero_division=0))

#*Logistic regression
lg_date = LogisticRegression()
lg_date.fit(x_train_date, y_train_date)
lg_date_score = lg_date.score(x_test_date, y_test_date)

# print("Logistic Accuracy (with month column): ", lg_date_score)

y_pred_lg_date = lg_date.predict(x_test_date)
conf_matrix_date = confusion_matrix(y_test_date, y_pred_lg_date)
# print("Confusion Matrix (with month column)")
# print(conf_matrix_date)
# print('Logistic Regression (with month column)\n',classification_report(y_test_date,y_pred_lg_date, zero_division=0))


#*Model reliability comparison chart
# models=["KNN","DECISION TREE","LOGISTIC REGRESSION"]
# accuracies =[knn_date_score,dec_date_score,lg_date_score]
# sns.set_style("darkgrid")
# plt.figure(figsize=(22,8))
# ax=sns.barplot(x=models,y=accuracies,palette="mako",saturation=1.5)
# plt.xlabel("Models",fontsize=20)
# plt.ylabel("Accuracy",fontsize=20)
# plt.title("Model Reliability Comparison 2",fontsize=20)
# plt.xticks(fontsize=11,horizontalalignment="center",rotation=8)
# plt.yticks(fontsize=13)

# for p in ax.patches:
#     ax.annotate(f'{p.get_height():.2%}',(p.get_x()+ p.get_width()/2,p.get_height()*1.02),ha="center",fontsize='x-large')

# plt.show()

#*build model in case of keeping date variables as YYYYMMDD

df3 = pd.read_csv("/Users/namkhanh/Desktop/Data Analysis/weather forcast/dataset.csv")
lc = LabelEncoder()

df3["weather_encoded"]=lc.fit_transform(df3["weather"])

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

#^split data test and data train
x_train_df3,x_test_df3,y_train_df3,y_test_df3 = train_test_split(x_df3,y_df3,test_size=0.1,random_state=2)



#*K-Nearest Neighbor classifier

knn_df3 = KNeighborsClassifier()
knn_df3.fit(x_train_df3, y_train_df3)
knn_score_df3 = knn_df3.score(x_test_df3, y_test_df3)
# print("KNN Accuracy:", knn_score_df3)

#*Decision Tree
# Create a list of max depth values to try
max_depth_range3 = list(range(1, 8))

# Train and evaluate a decision tree model with varying max depth values
for depth in max_depth_range3:
    
    # Create a decision tree classifier with the current max depth value and other parameters
    dec_df3 = DecisionTreeClassifier(max_depth=depth, max_leaf_nodes=15, random_state=0)
    
    # Train the decision tree model on the training data
    dec_df3.fit(x_train_df3, y_train_df3)
    
    # Compute the accuracy of the decision tree model on the testing data
    dec_score_df3 = dec_df3.score(x_test_df3, y_test_df3)
    
    # Print the accuracy score to the console
    # print("Decision Tree Accuracy: ", dec_score_df3)
    # print("Decision Tree Accuracy (with month column) for max_depth=", depth, ": ", dec_score_df3)

#*Logistic regression
lg_df3 = LogisticRegression()
lg_df3.fit(x_train_df3, y_train_df3)

lg_score_df3 = lg_df3.score(x_test_df3, y_test_df3)

# print("Logistic Accuracy : ", lg_score_df3)


#!Model testing
dec_df3 = DecisionTreeClassifier(max_depth=4, max_leaf_nodes=15, random_state=0)
dec_df3.fit(x_train_df3, y_train_df3)
dec_score_df3 = dec_df3.score(x_test_df3, y_test_df3)
print("Decision Tree Accuracy: ", dec_score_df3)


