# load libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from functions_model import pred_test, fit_overview, diff_overview, mape_stores

# load and filter data

train =pd.read_csv("data/train_df.csv", parse_dates=[0])
train =train[(train["item_category"] =="daily total") & (train["store_name"] !="KaDeWe")] 
train =train.dropna()

test =pd.read_csv("data/test_df.csv", parse_dates=[0])
test =test[(test["item_category"] =="daily total") & (test["store_name"] !="KaDeWe")] 


# define features 

numfeat =["days_back","temperature_2m_mean","sunshine_duration","precipitation_hours"]
catfeat =["store_name","month","year","weekday","hol_pub","hol_school",
"valentines_day","nye","halloween",
"street_market","public_space"]

# transform features

num_tr =Pipeline(
  steps=[
    ("scaling", RobustScaler()),
    ("polyint",PolynomialFeatures(3,include_bias=False))
])

cat_tr =Pipeline(steps=[
  ("ohe", OneHotEncoder(drop='first',sparse_output=False))
])

lag_tr =Pipeline(
    steps=[("scaling", RobustScaler())]
)

prepro =ColumnTransformer(
  transformers=[
    ("num", num_tr, numfeat),
    ("cat",cat_tr,catfeat),
    ("lag_tr", lag_tr,['lag1'])
])

# train data
# select features
train =train.sample(frac=1, random_state=21).reset_index(drop=True)
xtrain =train[numfeat +catfeat +['lag1']]
ytrain =train['total_amount']
ytest =test['total_amount']

# model fit and prediction 
lr =Pipeline(
  steps=[
    ("prepro", prepro),
    ("lr",LinearRegression())])
lr.fit(xtrain,ytrain)
ytrainpred =lr.predict(xtrain)


# evaluate model with test data
# predict
ytestpred =pred_test(train,test,lr,numfeat,catfeat)

# fir statistics for train and test
fit_overview(ytrain,ytrainpred,ytest,ytestpred)