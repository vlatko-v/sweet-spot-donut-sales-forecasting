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

from sktime.split import ExpandingWindowSplitter

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



# creating window splitter function for sktime's time-series cv plot

def window_splitter_prep(train, validation_length, num_folds):

    def get_initial_window_size(store_group, validation_length, num_folds):
        total_periods = len(store_group.index.get_level_values("date").unique())
        initial_window = total_periods - validation_length * num_folds
        return initial_window

    cv_list = []

    for (store,item), group in train.groupby(["store_name","item_category"]):

        # Calculate the initial window size for the current store
        initial_window = get_initial_window_size(group, validation_length, num_folds)

        # Initialize the ExpandingWindowSplitter for the current group
        splitter = ExpandingWindowSplitter(
            initial_window=initial_window,
            step_length=7,
            fh=list(range(1, 7 + 1))
        )

        cv_list.append(({f"{store}"+"_"+f"{item}":splitter},
                            {f"{store}"+"_"+f"{item}":group}))

    return cv_list




# mean average percentage error by store
# arguments:
# train / test dataset
# ytrainpred / ytestpred

def mape_stores(data, pred, breakdown:str):
  sum_ = pd.DataFrame({
    'Store name': data.reset_index()["store_name"],
    "Product category": data.reset_index()["item_category"],
    'Observed': data.reset_index()["total_amount"],
    'Predicted': pred.reset_index(drop = True)
    })
  
  sum_['mape'] = abs((sum_['Observed'] - sum_['Predicted'])/sum_['Observed'])*100
  
  if breakdown == "stores":
    mape_stores = sum_.groupby('Store name')['mape'].mean().reset_index()
    mape_stores.columns = ['Store name', 'MAPE']

  elif breakdown == "stores_products":
    mape_stores = sum_.groupby(['Store name',"Product category"])['mape'].mean().reset_index()
    mape_stores.columns = ['Store name', "Product Category", 'MAPE']

  return mape_stores



# observed, predicted and error for each store and date
# arguments:
# train / test dataset
# ytrainpred / ytestpred
# list of store names or the word all (without quotation marks) for every store in the dataset

def diff_overview(data,pred):
  scaler = StandardScaler()

  sum_ = pd.DataFrame({
    'Store name': data.reset_index()["store_name"],
    "Product category": data.reset_index()["item_category"],
    'Date': data.reset_index()["date"],
    'Observed': data.reset_index()["total_amount"],
    'Predicted': pred.reset_index(drop = True),
    'Difference': (pred.reset_index(drop = True) - data.reset_index()['total_amount'])
    })
  sum_["Stand_resid"] = scaler.fit_transform(np.array(sum_["Difference"]).reshape(-1,1))

  return sum_


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

def pred_test(train,test,model):
  
  def forecast_amount(store, category, model):

    start_date = test.index.get_level_values("date").min()
    end_date = test.index.get_level_values("date").max()
    date_range = pd.date_range(start = start_date, end = end_date)

    # Training data for the specific store and product category
    train_store_category = train[(train.store_name == store) & (train.item_category == category)]

    # Initialize the lag values
    lag_value = train_store_category['total_amount'].iloc[-1]
    lag_value2 = train_store_category['total_amount'].iloc[-2]
    lag_value3 = train_store_category['total_amount'].iloc[-3]

    # Initialize the moving averages and moving standard deviation

    moving_avg_7 = train_store_category.tail(7)['total_amount'].mean()
    moving_avg_15 = train_store_category.tail(15)['total_amount'].mean()
    moving_avg_30 = train_store_category.tail(30)['total_amount'].mean()
    moving_std_4 = train_store_category.tail(4)['total_amount'].std()
    

    pred_daily_amount = {}

    for date in date_range:

        x = test[(test.store_name == store) & (test.item_category == category) & (test.index.get_level_values("date") == date)]
        
        x.loc[:, "total_amount_lag_1"] = lag_value
        x.loc[:, "total_amount_lag_2"] = lag_value2
        x.loc[:, "total_amount_lag_3"] = lag_value3
        x.loc[:, "total_amount_mean_1_7"] = moving_avg_7
        x.loc[:, "total_amount_mean_1_15"] = moving_avg_15
        x.loc[:, "total_amount_mean_1_30"] = moving_avg_30
        x.loc[:, "total_amount_std_1_4"] = moving_std_4

        x = x[train.drop("total_amount", axis = 1).columns]

        pred_amount = model.predict(x)[0]

        pred_daily_amount[date] = [pred_amount, lag_value, lag_value2, lag_value3, moving_avg_7, moving_avg_15, moving_avg_30, moving_std_4]

        # Updating lag values
        lag_value = pred_amount
        lag_value2 = pred_daily_amount[date][1]
        lag_value3 = pred_daily_amount[date][2]


        # Updating the moving averages and standard deviation by shifting the windows
        train_store_category = pd.concat([train_store_category.iloc[1:], pd.DataFrame([{"total_amount": pred_amount}])], ignore_index=True)
        
        moving_avg_7 = train_store_category.tail(7)['total_amount'].mean()
        moving_avg_15 = train_store_category.tail(15)['total_amount'].mean()
        moving_avg_30 = train_store_category.tail(30)['total_amount'].mean()
        moving_std_4 = train_store_category.tail(4)['total_amount'].std()

    return pred_daily_amount

  daily_forecast_store_cat = {(store, category): forecast_amount(store, category, model)
                               for store in test.store_name.unique()
                               for category in test.item_category.unique()}

  test = pd.concat([test.reset_index(), 
           test.reset_index().apply(lambda x: daily_forecast_store_cat[(x["store_name"], x["item_category"])][x["date"]], axis = 1)], axis =1,
           )

  test[['pred_total_amount',"total_amount_lag_1","total_amount_lag_2","total_amount_lag_3", "total_amount_mean_7",
        "total_amount_mean_15","total_amount_mean_30","total_amount_std_4"]]  = test[0].apply(pd.Series)

  test = test.drop(0, axis = 1)
  test = test.set_index("date")

  ytestpred = test['pred_total_amount']

  return test, ytestpred