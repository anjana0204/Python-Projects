# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:43:03 2020

@author: Akshay
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datetime import datetime
from datetime import timedelta
import seaborn as sn
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
pip install imbalanced-learn
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV


train_data=pd.read_csv("C:/Users/Akshay/Desktop/Anjana/Machine Learning/Janta Hack/train_8wry4cB.csv")

X=train_data.drop(["gender"],axis=1)
y=train_data["gender"]
y=pd.DataFrame(y,columns=["gender"])
y = y['gender'].map({'female':1,'male':0})

""" Extracting Product Details"""

def extract_details(str):
    categories=[]
    sub_categories=[]
    sub_sub_categories=[]
    products=[]
    if(";" in str):
        Items_List=str.split(";")
        Category1=Items_List[0].split("/")[0]
        Sub_Category1=Items_List[0].split("/")[1]
        for  item in Items_List:
            categories.append(item.split("/")[0])
            sub_categories.append(item.split("/")[1])
            sub_sub_categories.append(item.split("/")[2])
            products.append(item.split("/")[3])
            Number_of_Categories=len(categories)
            Number_of_Subcategories=len(sub_categories)
            Number_of_Subsubcategories=len(sub_sub_categories)
            Number_of_Products=len(products)
            Number_Of_Items=len(Items_List)
            Max_Bought_Category=max(categories)
            Max_Bought_SubCategory=max(sub_categories)
            Max_Bought_Product=max(products)
    else:
        Items_List2=str.split("/")
        Category1=Items_List2[0]
        Sub_Category1=Items_List2[1]
        categories.append(Items_List2[0])
        sub_categories.append(Items_List2[1])
        sub_sub_categories.append(Items_List2[2])
        products.append(Items_List2[3])
        Number_of_Categories=len(categories)
        Number_of_Subcategories=len(sub_categories)
        Number_of_Subsubcategories=len(sub_sub_categories)
        Number_of_Products=len(products)
        Number_Of_Items=len(Items_List2)
        Max_Bought_Category=Category1
        Max_Bought_SubCategory=Items_List2[2]
        Max_Bought_Product=Items_List2[3]
        
        
    return(Number_Of_Items,Category1,Sub_Category1,Number_of_Categories,Number_of_Subcategories,Max_Bought_Category,Number_of_Subsubcategories,Number_of_Products,Max_Bought_SubCategory,Max_Bought_Product)

New_features_series=X["ProductList"].apply(lambda x: extract_details(x))
New_col_names=["Number_of_items","Category1","SubCategory1",
            "Number_Of_Categories","Number_Of_SubCategories","Most_Bought_Category",
            "Number_of_Subsubcategories","Number_of_Products","Max_Bought_SubCategory","Max_Bought_Product"]
New_features_data=New_features_series.to_list()
New_Features_DataFrame=pd.DataFrame(New_features_data,columns=New_col_names)


""" Calculating time taken to shop"""
X["startTime"]=pd.to_datetime(X["startTime"])
X["endTime"]=pd.to_datetime(X["endTime"])
X["Duration of shopping in minutes"]=X["endTime"]-X["startTime"]
X["Duration of shopping in minutes"]=X["Duration of shopping in minutes"]/np.timedelta64(1,'m')
X['weekday'] = X['startTime'].dt.dayofweek


"""Dropping unnecessary varaiables"""
X=X.drop(['session_id', 'startTime', 'endTime', 'ProductList'],axis=1)
Indep_Var=pd.concat([New_Features_DataFrame.reset_index(drop=True),X.reset_index(drop=True)],axis=1)
Target_Var=y

"""Dummy Variables"""
data1=pd.get_dummies(Indep_Var,drop_first=True)
scaler=MinMaxScaler()

X_train1,X_test,y_train1,y_test=train_test_split(data1,Target_Var,test_size=0.25,random_state=42,stratify=Target_Var)
X_train,X_val,y_train,y_val=train_test_split(X_train1,y_train1,test_size=0.25,random_state=42,stratify=y_train1)

"""Scaling data"""
X_train=scaler.fit_transform(X_train)
X_val=scaler.fit_transform(X_val)
X_test=scaler.fit_transform(X_test)

"""There is a huge imbalance in classification, so we use SMOTE algorithm"""
"""SMOTE"""
sm=SMOTE(random_state=42)
X_train,y_train=sm.fit_sample(X=X_train,y=y_train)

"""Fitting the model"""
lr=LogisticRegression()

model=lr.fit(X_train,y_train)
train_predict=model.predict(X_train)
Cross_score=cross_val_score(lr,X_val,y_val,cv=5)
Cross_score
""" This gives an array of validation scores which are good"""
"""Finding scores for training set"""

print("FI Score:"+ str(f1_score(train_predict,y_train)))
print("Accuracy Score:"+str(accuracy_score(train_predict,y_train)))
print("Recall_score:"+str(recall_score(train_predict,y_train)))
""" This shows great scores on evaluation metrics"""


""" Predicting on test data set"""

test_predict=lr.predict(X_test)
print("FI Score:"+ str(f1_score(test_predict,y_test)))
print("Accuracy Score:"+str(accuracy_score(test_predict,y_test)))
print("Recall_score:"+str(recall_score(test_predict,y_test)))

"""
FI Score:0.9043521022866978

Accuracy Score:0.8518095238095238

Recall_score:0.9108469539375929

"""

