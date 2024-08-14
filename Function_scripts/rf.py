# load libraries and data

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV
import datetime as dt
import time
from functions_model import pred_test, fit_overview, diff_overview, mape_stores, create_val_set

# load data
train =pd.read_csv("train_df.csv", parse_dates=[0])
train =train[train["item_category"] =="daily total"]
train =train.dropna().reset_index(drop=True)

def create_val(df):
  last_val_date =df.date.max()
  cv =[]
  for n in range(5):
    first_val_date = last_val_date - dt.timedelta(days=6)
    train_index = df[df.date<first_val_date].index
    val_index = df[(df.date>=first_val_date) & (df.date<=last_val_date)].index
    cv.append((train_index,val_index))
    last_val_date = first_val_date - dt.timedelta(days=1)
    cv =cv[::-1]
    return cv


test =pd.read_csv("test_df.csv", parse_dates=[0])
test =test[test["item_category"] =="daily total"]

# define features 
numfeat =["days_back","temperature_2m_mean","sunshine_duration","precipitation_hours"]
catfeat =["store_name","month","year","weekday","hol_pub","hol_school",
"valentines_day","nye","halloween",
"street_market","public_space"]

# transform features


cat_tr =Pipeline(steps=[
  ("ohe", OneHotEncoder(drop='first',sparse_output=False))
])

prepro =ColumnTransformer(
  transformers=[
    ("cat",cat_tr,catfeat)],remainder ='passthrough')


# featrue selection
xtrain =train[numfeat +catfeat +["lag1","lag2"]]
ytrain =train['total_amount']

# model fit & predict
seed =21
rf =Pipeline(
  steps=[
    ("prepro", prepro),
    ('rf',RandomForestRegressor(random_state=seed,
      n_jobs=6, verbose =1,criterion="poisson"))
    ])


cvsplit =create_val(train)

param={
  "rf__max_depth":np.arange(5,50,1),
  "rf__n_estimators":np.arange(50,500,10),
  "rf__min_samples_split":np.arange(4,10,1),
  "rf__min_samples_leaf":np.arange(2,10,1)
  }

rfh =RandomizedSearchCV(estimator=rf, param_distributions=param,
      scoring ='neg_mean_absolute_percentage_error', cv=cvsplit, n_jobs=6, n_iter =200)
rfh.fit(xtrain, ytrain)
rf_best =rfh.best_estimator_
ytrainpred =rf_best.predict(xtrain)

with open('rf_best1.pkl', mode='wb') as file:
  pickle.dump(rf_best,file)

# validation
ytest =test['total_amount']
test,ytestpred =pred_test(train,test,rf_best,numfeat,catfeat)
# fir statistics
print(f"Best validation MAPE: {-rfh.best_score_:.6f}")
fit_overview(ytrain,ytrainpred,ytest,ytestpred)
print(rfh.best_params_)