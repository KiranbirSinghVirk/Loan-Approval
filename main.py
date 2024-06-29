import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import preprocessing 

data = pd.read_csv("LoanApprovalPrediction.csv") 

# obj = (data.dtypes == 'object') 
# print("Categorical variables:",len(list(obj[obj].index)))
# Dropping Loan_ID column 
# Dropping Loan_ID column 
data.drop(['Loan_ID'],axis=1,inplace=True)


obj = (data.dtypes == 'object') 
index = 1
object_cols = list(obj[obj].index) 
plt.figure(figsize=(20,40)) 
index=1

  
for col in object_cols: 
  
  y = data[col].value_counts() 
  plt.subplot(4,4,index) 
  plt.xticks(rotation=90) 
  sns.barplot(x=list(y.index), y=y) 
  plt.ylabel(col)  # Replace with your specific label

  index +=1
plt.show() 

label_encoder = preprocessing.LabelEncoder() 
for col in list(obj[obj].index): 
  data[col] = label_encoder.fit_transform(data[col])
plt.figure(figsize=(12,6)) 
  
sns.heatmap(data.corr(),cmap='BrBG',fmt='.2f', linewidths=2,annot=True)  
plt.show()
