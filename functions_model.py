# functions for model testing and evaluation 
# import packages
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import datetime as dt


# adjusted r-squared
# arguments: r-squared, dataframe with features 

def adj_r2(r2, x):
    adjr2 =round(1 - ((1 -r2) * (len(x) - 1) / (len(x) - x.shape[1] - 1)),3)
    return adjr2 


# Advanced Dickey-Fuller test to check for stationarity

def adf_test_p_values (df):
    p_vals = []
    for store in df["store_name"].unique():
        store_df = df[df["store_name"] == store]
        p = adfuller(store_df["total_amount"], autolag="AIC")[1]
        p_vals.append((store, p))
    return p_vals



# Create validation set

def create_val_set(df):
    df = df.set_index("date")
    days = np.sort(df.index.unique())
    tscv = TimeSeriesSplit(n_splits=2, test_size = 7)

    for train_index, val_index in tscv.split(days):
        train_days, val_days = days[train_index], days[val_index]
        train, val = df.loc[train_days], df.loc[val_days]

    train = train.sort_values("date", ascending = False)
    val = val.sort_values("date", ascending = False)

    train = train.reset_index()
    val = val.reset_index()

    return train, val


# Create validation folds

def create_train_validation_folds(df):
    last_val_date = df.date.max()
    cv = []
    for n in range(5):
        first_val_date = last_val_date - dt.timedelta(days=6)
        train_index = df[df.date<first_val_date].index
        val_index = df[(df.date>=first_val_date) & (df.date<=last_val_date)].index
        cv.append((train_index,val_index))
        last_val_date = first_val_date - dt.timedelta(days=1)
    cv = cv[::-1]
    return cv




# mean average percentage error by store
# arguments:
# train / test dataset
# ytrainpred / ytestpred

def mape_stores(data, pred):
  sum_ =pd.DataFrame({
    'Store name': data["store_name"],
    'Observed': data["total_amount"],
    'Predicted': pred})
  sum_['mape'] =abs((sum_['Observed'] -sum_['Predicted'])/sum_['Observed'])*100
  mape_stores =sum_.groupby('Store name')['mape'].mean().reset_index()
  mape_stores.columns =['Store name', 'MAPE']
  return mape_stores

# observed, predicted and error for each store and date
# arguments:
# train / test dataset
# ytrainpred / ytestpred
# list of store names or the word all (without quotation marks) for every store in the dataset

def diff_overview(data,pred,stores):
  scaler = StandardScaler()

  sum_ =pd.DataFrame({
    'Store name': data["store_name"],
    'Date': data["date"],
    'Observed': data["total_amount"],
    'Predicted': pred,
    'Difference': (pred - data['total_amount'])})
  sum_["Stand_resid"] = scaler.fit_transform(np.array(sum_["Difference"]).reshape(-1,1))
  if stores ==all:
    return sum_
  else:
    return sum_[sum_['Store name'].isin(stores)]


# fit statistics for train and test data
# arguments:
# target and prediction for train and test data

def fit_overview(ytrain, ytrainpred, ytest, ytestpred):
  r2_train =round(r2_score(ytrain, ytrainpred),3)
  r2_test =round(r2_score(ytest, ytestpred),3)
  print("R-squared train: ", r2_train)
  print("Mean absolute percentage error train: ", 100*(mean_absolute_percentage_error(ytrain, ytrainpred).round(2)),"\n")
  
  print("R-squared test ", r2_test)
  print("Mean absolute percentage error test: ", 100*(mean_absolute_percentage_error(ytest, ytestpred).round(2)))




# prections for test dataset
# arguments:
# train and test data
# model which was already instantiated
# catfeat, numfeat: list of categorical and numerical features
  
def pred_test(train,test,model,numfeat,catfeat):
  
  def forecast_amount(store, model):

    start_date = test.date.min()
    end_date = test.date.max()
    date_range = pd.date_range(start = start_date, end = end_date)

    test_store = test[(test.store_name==store)]

    train_store_last_day = train[(train.store_name==store) & (train.date == train.date.max())]
    train_store_two_days_before = train[(train.store_name==store) & (train.date==train.date.max() - pd.Timedelta(days = 1))]
    train_store_three_days_before = train[(train.store_name==store) & (train.date==train.date.max() - pd.Timedelta(days = 2))]
    train_store_four_days_before = train[(train.store_name==store) & (train.date==train.date.max() - pd.Timedelta(days = 3))]
    train_store_five_days_before = train[(train.store_name==store) & (train.date==train.date.max() - pd.Timedelta(days = 4))]
    train_store_six_days_before = train[(train.store_name==store) & (train.date==train.date.max() - pd.Timedelta(days = 5))]
    train_store_week_before = train[(train.store_name==store) & (train.date==train.date.max() - pd.Timedelta(days = 6))]

    lag_value = train_store_last_day['total_amount'].iloc[0]
    lag_value2 = train_store_two_days_before["total_amount"].iloc[0]
    lag_value3 = train_store_three_days_before["total_amount"].iloc[0]
    lag_value4 = train_store_four_days_before["total_amount"].iloc[0]
    lag_value5 = train_store_five_days_before["total_amount"].iloc[0]
    lag_value6 = train_store_six_days_before["total_amount"].iloc[0]
    lag_value7 = train_store_week_before['total_amount'].iloc[0]

    pred_daily_amount = {}


    for date in date_range:

        x = test_store[test_store.date == date][catfeat + numfeat]
        x['lag1'] = lag_value
        x["lag2"] = lag_value2
        x["lag3"] = lag_value3
        x["lag4"] = lag_value4
        x["lag5"] = lag_value5
        x["lag6"] = lag_value6
        x["lag7"] = lag_value7

        pred_amount = model.predict(x)[0]
        pred_daily_amount[date] = [pred_amount, lag_value, lag_value2, lag_value7]
    
        lag_value = pred_amount
        lag_value2 = x["lag1"].iloc[0]
        lag_value3 = x["lag2"].iloc[0]
        lag_value4 = x["lag3"].iloc[0]
        lag_value5 = x["lag4"].iloc[0]
        lag_value6 = x["lag5"].iloc[0]
        lag_value7 = x["lag6"].iloc[0]

    return pred_daily_amount

  storewise_daily_forecast = {store:forecast_amount(store, model) for store in test.store_name.unique()}

  test['pred_total_amount'] = test.apply(lambda x: storewise_daily_forecast[x.store_name][x.date], axis=1)

  test["all"]  = test.apply(lambda x: storewise_daily_forecast[x.store_name][x.date], axis=1)

  test[['pred_total_amount',"lag1","lag2","lag7"]]  = test["all"].apply(pd.Series)

  test = test.drop("all", axis = 1)


  ytestpred = test['pred_total_amount']

  return test, ytestpred  