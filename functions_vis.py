

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as md
import seaborn as sns
import pandas as pd
import numpy as np

import scipy.stats as ss







# Time series plot by store

def ts_lineplot (df):

    df["year_month"] = pd.to_datetime(df['year'].astype(str) + '-'+ df['month'].astype(str), format= '%Y-%m')
    df = df.dropna().set_index("date")

    fig, axes = plt.subplots(4,2, figsize = (12,10), sharey = False, sharex = True)
    axes = axes.flatten()

    for i, store in enumerate(df["store_name"].unique()):
      store_df = df[df["store_name"] == store]
      store_grouped = store_df.groupby("year_month").agg({"total_amount":"sum"})

      sns.lineplot(data = store_grouped, x = store_grouped.index, y = "total_amount", linewidth = 2, ax = axes[i], errorbar=("ci",False))

      axes[i].set_title(store, size = 24)
      axes[i].set_xlabel("")
      axes[i].set_ylabel("Amount sold", size = 18)
      axes[i].tick_params(axis='y', labelsize=16)
      axes[i].tick_params(axis='x', labelsize=16, rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()




# Time series plot stacked (Danziger, Maybachufer, Potsdamer)

def ts_lineplot_stacked (df):

    df["year_month"] = pd.to_datetime(df['year'].astype(str) + '-'+ df['month'].astype(str), format= '%Y-%m')
    df = df.dropna().set_index("date")

    df_grouped = df.groupby(["store_name","year_month"]).agg({"total_amount":"sum"}).reset_index()

    df_grouped = df_grouped[df_grouped["store_name"].isin(["Potsdamer","Maybachufer","Danziger"])].reset_index()

    df_grouped["store_name"] =  df_grouped["store_name"].map({"Danziger":"Store 1 (Residential Area)", "Maybachufer":"Store 2 (Residential Area)", "Potsdamer":"Store 3 (Tourist Location)"})

    plt.figure(figsize=(12,6))
    sns.lineplot(data = df_grouped, x = "year_month", y = "total_amount", linewidth = 2, errorbar=("ci",False), hue = "store_name")


    plt.title("Donut Sales", size = 24)
    plt.xlabel("")
    plt.ylabel("Amount sold", size = 18)
    plt.tick_params(axis='y', labelsize=16)
    plt.tick_params(axis='x', labelsize=16)

    plt.legend(loc = "upper left", fontsize = 14)

    plt.tight_layout()
    plt.show()






# Histogram of total sales


def vis_total_amount_hist (df):
    fig, axes = plt.subplots(4, 3, figsize = (15,20), sharey=False)
    axes = axes.flatten()
    df = df[df["item_category"] == "daily total"]

    for i, store in enumerate(df["store_name"].unique()):
        store_data = df[df["store_name"] == store]
        
        sns.histplot(data = store_data, x = "total_amount", 
                     hue = "year", palette = "plasma", 
                     multiple="stack", ax = axes[i])

        axes[i].set_title(store, size = 24)
        axes[i].set_ylabel('Count', size = 18)
        axes[i].set_xlabel('Amount Sold', size = 18)
        axes[i].tick_params(axis='y', labelsize=16)
        axes[i].tick_params(axis='x', labelsize=16)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()



# Visualisation of weather data and total amount sold

## Rainfall

def vis_rain (df):
    fig, axes = plt.subplots(4, 3, figsize = (15,20), sharey=True)
    axes = axes.flatten()
    df = df[df["item_category"] == "daily total"]

    for i, store in enumerate(df["store_name"].unique()):
        store_data = df[df["store_name"] == store]
        
        sns.regplot(data = store_data, x = "precipitation_hours",
        y = "total_amount", line_kws= {"color":"red"}, ax = axes[i],
        lowess=True)

        axes[i].set_title(store, size = 24)
        axes[i].set_ylabel('Amount Sold', size = 18)
        axes[i].set_xlabel('Precipitation (hrs)', size = 18)
        axes[i].tick_params(axis='y', labelsize=16)
        axes[i].tick_params(axis='x', labelsize=16)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()



def vis_rain_bin (df):
    fig, axes = plt.subplots(4, 3, figsize = (15,20), sharey=True)
    axes = axes.flatten()

    df = df[df["item_category"] == "daily total"]

    df["rainfall_bins"] = pd.Categorical(df["rainfall_bins"], 
                                        categories=["0-4 hrs", "4-8 hrs" ,"8-12 hrs", "12-16 hrs", "> 16 hrs"], ordered=True)

    for i, store in enumerate(df["store_name"].unique()):
        store_data = df[df["store_name"] == store]

        sns.stripplot(data = df, x = "rainfall_bins",
        y = "total_amount", linewidth=0.5, alpha=0.6, color="#0175DB", ax = axes[i])
        
        sns.barplot(data = store_data, x = "rainfall_bins",
        y = "total_amount", errorbar=("ci",False), color="#0175DB",  alpha = 0.7,  ax = axes[i])

        axes[i].set_title(store, size = 24)
        axes[i].set_ylabel('Amount Sold', size = 18)
        axes[i].set_xticklabels(["0-4","4-8","8-12","12-16",">16"], size = 16)
        axes[i].set_xlabel('Precipitation (hrs)', size = 18)
        axes[i].tick_params(axis='y', labelsize=16)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()



# # Visualisation of rainfall and sales over months 

def rainfall_month_sales (df):
    fig, axes1 = plt.subplots(4,2, figsize = (12,15), sharey = True)
    axes1 = axes1.flatten()

    df = df[df["item_category"] == "daily total"]

    for i, store in enumerate(df["store_name"].unique()):
        store_df = df[df["store_name"] == store]

        # Sales

        sns.barplot(data = store_df, x = "month", y = "total_amount", errorbar=("ci",False), 
                    alpha = 0.5, color = "#3578FF", ax = axes1[i])

        axes1[i].set_title(store, size = 24)
        axes1[i].set_xlabel('')
        axes1[i].set_ylabel('Total Amount', size = 18)
        axes1[i].set_xticklabels(axes1[i].get_xticklabels(), rotation=45, size = 13)
        #axes1[i].set_yticklabels(axes1[i].get_yticklabels(), size = 16)

        axes2 = axes1[i].twinx()

        # Temperature

        sns.lineplot(data = store_df, x = "month", y = "precipitation_hours", errorbar=("ci",False), color = "red", ax = axes2)

        axes2.set_ylabel('Precipitation (hours)', size = 18)
        #axes2.set_yticklabels(axes2.get_yticklabels(), size = 16)

    plt.tight_layout()





## Sunshine duration

def vis_sunshine (df):
    fig, axes = plt.subplots(4, 3, figsize = (15,20), sharey=True)
    axes = axes.flatten()
    df = df[df["item_category"] == "daily total"]

    for i, store in enumerate(df["store_name"].unique()):
        store_data = df[df["store_name"] == store]
        
        sns.regplot(data = store_data, x = "sunshine_duration",
        y = "total_amount", line_kws= {"color":"red"}, ax = axes[i],
        lowess=True)

        axes[i].set_title(store, size = 24)
        axes[i].set_ylabel('Amount Sold', size = 18)
        axes[i].set_xlabel('Sunshine duration (seconds)', size = 18)
        axes[i].tick_params(axis='y', labelsize=16)
        axes[i].tick_params(axis='x', labelsize=16)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()


def vis_sunshine_bin (df):
    fig, axes = plt.subplots(4, 3, figsize = (15,20), sharey=True)
    axes = axes.flatten()

    df = df[df["item_category"] == "daily total"]
    df["sunshine_bins"] = pd.Categorical(df["sunshine_bins"], 
                                        categories=["0-4 hrs", "4-8 hrs", "8-12 hrs", "> 12 hrs"], ordered=True)
    
    for i, store in enumerate(df["store_name"].unique()):
        store_data = df[df["store_name"] == store]

        sns.stripplot(data = df, x = "sunshine_bins",
        y = "total_amount", linewidth=0.5, alpha=0.6, color="#FFE810", ax = axes[i])
        
        sns.barplot(data = store_data, x = "sunshine_bins",
        y = "total_amount", errorbar=("ci",False), color="#FFE810",  alpha = 0.7,  ax = axes[i])

        axes[i].set_title(store, size = 24)
        axes[i].set_ylabel('Amount Sold', size = 18)
        axes[i].set_xticklabels(["0-4", "4-8", "8-12", "> 12"], size = 16)
        axes[i].set_xlabel('Sunshine duration (hrs)', size = 18)
        axes[i].tick_params(axis='y', labelsize=16)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()



## Temperature 

def vis_temp (df):
    fig, axes = plt.subplots(4, 3, figsize = (15,20), sharey=True)
    axes = axes.flatten()
    df = df[df["item_category"] == "daily total"]

    for i, store in enumerate(df["store_name"].unique()):
        store_data = df[df["store_name"] == store]
        
        sns.regplot(data = store_data, x = "temperature_2m_mean",
        y = "total_amount", line_kws= {"color":"red"}, ax = axes[i],
        lowess=True)

        axes[i].set_title(store, size = 24)
        axes[i].set_ylabel('Amount Sold', size = 18)
        axes[i].set_xlabel('Temperature (°C)', size = 18)
        axes[i].tick_params(axis='y', labelsize=16)
        axes[i].tick_params(axis='x', labelsize=16)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()



def vis_temp_bin (df):
    fig, axes = plt.subplots(4, 3, figsize = (15,20), sharey=True)
    axes = axes.flatten()

    df = df[df["item_category"] == "daily total"]
    df["temp_bins"] = pd.Categorical(df["temp_bins"], 
                                     categories=['< 0°C', '0 - 10°C','10 - 15°C', '15 - 20°C', '20 - 25°C',   '> 25°C'], ordered=True)
    
    for i, store in enumerate(df["store_name"].unique()):
        store_data = df[df["store_name"] == store]

        sns.stripplot(data = df, x = "temp_bins",
        y = "total_amount", linewidth=0.5, alpha=0.6, color="#DF0404", ax = axes[i])
        
        sns.barplot(data = store_data, x = "temp_bins",
        y = "total_amount", errorbar=("ci",False), color="#DF0404",  alpha = 0.7,  ax = axes[i])


        axes[i].set_title(store, size = 24)
        axes[i].set_ylabel('Amount Sold', size = 18)
        axes[i].set_xticklabels(["< 0","0-10","10-15","15-20","20-25",">25"], size = 16)
        axes[i].set_xlabel('Temperature (°C)', size = 18)
        axes[i].tick_params(axis='y', labelsize=16)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()


# Visualisation of temperature and sales over months 


def temp_month_sales (df):
    fig, axes1 = plt.subplots(4,2, figsize = (12,15), sharey = True)
    axes1 = axes1.flatten()

    df = df[df["item_category"] == "daily total"]

    for i, store in enumerate(df["store_name"].unique()):
        store_df = df[df["store_name"] == store]

        # Sales

        sns.barplot(data = store_df, x = "month", y = "total_amount", errorbar=("ci",False),  color = "#3578FF", ax = axes1[i])

        axes1[i].set_title(store, size = 24)
        axes1[i].set_xlabel('')
        axes1[i].set_ylabel('Total Amount', size = 18)
        axes1[i].set_xticklabels(axes1[i].get_xticklabels(), rotation=45, size = 13)
        #axes1[i].set_yticklabels(axes1[i].get_yticklabels(), size = 16)

        axes2 = axes1[i].twinx()

        # Temperature

        sns.lineplot(data = store_df, x = "month", y = "temperature_2m_mean", errorbar=("ci",False), color = "red", ax = axes2)

        axes2.set_ylabel('Temperature (°C)', size = 18)
        #axes2.set_yticklabels(axes2.get_yticklabels(), size = 16)

    plt.tight_layout()




# Visualisation of holidays and total amount sold

def vis_pub_hol(df):
    fig, axes = plt.subplots(4, 3, figsize = (15,15), sharey=False)
    axes = axes.flatten()
    df = df[df["item_category"] == "daily total"]
    
    for i, store in enumerate(df["store_name"].unique()):
        store_data = df[df["store_name"] == store]
        
        sns.boxplot(data = store_data, x = "hol_pub",
        y = "total_amount", ax = axes[i])

        axes[i].set_title(store, size = 24)
        axes[i].set_xticklabels(["No Holiday","Public Holiday"], size = 18)
        axes[i].set_ylabel('Amount Sold', size = 18)
        axes[i].tick_params(axis='y', labelsize=16)
        axes[i].set_xlabel('')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()



def vis_school_hol(df):
    fig, axes = plt.subplots(4, 3, figsize = (15,15), sharey=False)
    axes = axes.flatten()
    df = df[df["item_category"] == "daily total"]

    for i, store in enumerate(df["store_name"].unique()):
        store_data = df[df["store_name"] == store]
        
        sns.boxplot(data = store_data, x = "hol_school",
        y = "total_amount", ax = axes[i])

        axes[i].set_title(store, size = 24)
        axes[i].set_xticklabels(["No Holiday","School Holiday"], size = 18)
        axes[i].set_ylabel('Amount Sold', size = 18)
        axes[i].tick_params(axis='y', labelsize=16)
        axes[i].set_xlabel('')
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()






# Visualisation of public spaces

def vis_pub_spaces(df):
    fig, axes = plt.subplots(3, 3, figsize = (15,15), sharey=False)
    axes = axes.flatten()
    df = df[(df["item_category"] == "daily total") & 
       (df["date"] >= pd.to_datetime("2018-09-14"))]

    for i, year in enumerate(df["year"].unique()):
        year_data = df[df["year"] == year]

        sns.boxplot(data = year_data, x = "public_space",
        y = "total_amount", ax = axes[i])

        axes[i].set_title(year, size = 24)
        axes[i].set_xticks([0,1],["Residential","Public Space"], size = 18)
        axes[i].set_ylabel('Amount Sold', size = 18)
        axes[i].tick_params(axis='y', labelsize=16) 
        axes[i].set_xlabel('')
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()






## Time variables

### Week of year

def vis_weeks(df):
    fig, axes = plt.subplots(4, 3, figsize = (10,15), sharey=True)
    axes = axes.flatten()
    df = df[df["item_category"] == "daily total"]

    for i, store in enumerate(df["store_name"].unique()):
        store_data = df[df["store_name"] == store]

        store_data["special_events"] = store_data.apply(lambda x: True if x["nye"] == 1 or x["valentines_day"] == 1 or x["halloween"] == 1 else False, axis = 1)

        weekly_sales = store_data.groupby("week_year").agg({"total_amount":"sum", "special_events":"max"}).reset_index()

        weekly_sales['color'] = weekly_sales['special_events'].apply(lambda x: 'red' if x == True else '#00B4F6')

        sns.barplot(data = weekly_sales, x = "week_year",
        y = "total_amount", errorbar = ("ci",False), palette=weekly_sales['color'], ax = axes[i])

        axes[i].set_title(store, size = 24)
        axes[i].set_ylabel('Amount Sold', size = 18)
        axes[i].set_xlabel('Week of the Year', size = 18)
        axes[i].set_xticklabels("")
        axes[i].tick_params(axis='y', labelsize=16) 

        legend_elements = [
            Patch(facecolor='red', edgecolor='red', label='Special Event Week')]

        axes[i].legend(handles=legend_elements, title='')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()



### Weekday

def vis_weekday(df):
    fig, axes = plt.subplots(4, 3, figsize = (10, 15), sharey=True)
    axes = axes.flatten()
    df = df[df["item_category"] == "daily total"]

    for i, store in enumerate(df["store_name"].unique()):
        store_data = df[df["store_name"] == store]
        
        sns.barplot(data = store_data, x = "weekday",
        y = "total_amount", errorbar=("ci",False), color = "#B60192", ax = axes[i])

        axes[i].set_title(store, size = 24)
        axes[i].set_ylabel('Amount Sold', size = 18)
        axes[i].set_xlabel('')
        axes[i].set_xticks([0,1,2,3,4,5,6],["Mo","Tu","We","Th","Fr","Sa","Su"], size = 16)
        axes[i].tick_params(axis='y', labelsize=16) 

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()



### Weekend

def vis_weekend(df):
    fig, axes = plt.subplots(4, 3, figsize = (10,15), sharey=False)
    axes = axes.flatten()
    df = df[df["item_category"] == "daily total"]

    for i, store in enumerate(df["store_name"].unique()):
        store_data = df[df["store_name"] == store]
        
        sns.boxplot(data = store_data, x = "weekend",
        y = "total_amount", ax = axes[i])

        axes[i].set_title(store, size = 24)
        axes[i].set_ylabel('Total Amount Sold', size = 18)
        axes[i].set_xlabel('')
        axes[i].set_xticks([0,1],["Workday","Weekend"], size = 14)
        axes[i].tick_params(axis='y', labelsize=16) 

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()



#### Predictions ####

# Residual plot

def residual_plot(df):

    df = df.copy()
    
    fig, ax = plt.subplots(1, 2, figsize = (15,6))

    # No daily sales

    sns.scatterplot(data = df[df["Product category"] != "daily total"], 
                    x = "Predicted", y = "Stand_resid", 
                    hue = "Product category", ax = ax[0])

    ax[0].axhline(y=0, color='r', linestyle='--')
    ax[0].set_xlabel("Predicted Sales (Daily total)")
    ax[0].set_ylabel("Standardized Residuals")
    ax[0].set_title("Residual Plot (Excluding Total Daily Sales)")


    # Only daily sales

    custom_labels = {
        "Altona": "Store 1",
        "Danziger": "Store 2",
        "Jungfernstieg": "Store 3",
        "Maybachufer": "Store 4",
        "Mitte": "Store 4",
        "Neuer Kamp": "Store 6",
        "Potsdamer": "Store 7",
        "Warschauer": "Store 8"
    }

    df["Store name"] = df["Store name"].map(custom_labels)

    sns.scatterplot(data = df[df["Product category"] == "daily total"], 
                    x = "Predicted", y = "Stand_resid", 
                    hue = "Store name", ax = ax[1])

    ax[1].axhline(y=0, color='r', linestyle='--')
    ax[1].set_xlabel("Predicted Sales (Daily total)")
    ax[1].set_ylabel("Standardized Residuals")
    ax[1].set_title("Residual Plot (Only Total Daily Sales)")

    plt.show()



# Final timeseries prediction

def ts_predicted (df):
    fig, axes = plt.subplots(4,2, figsize = (15,18))
    axes = axes.flatten()

    for i, store in enumerate(df["store_name"].unique()):
        store_df = df[(df["store_name"] == store)]
        cutoff_date = pd.to_datetime("2024-05-25")

        sns.lineplot(data = store_df, x = "date", y = "total_amount", ax = axes[i], errorbar = ("ci", False), marker = "o", markersize = 10,linewidth = 4, label = "Observed")

        sns.lineplot(data = store_df[store_df["date"] >= cutoff_date], x = "date", y = "Predicted", errorbar = ("ci", False), label = "Predicted", marker = "o", markersize = 10, linewidth = 4, ax = axes[i])

        axes[i].axvline(x = cutoff_date, color='black', linestyle='--', linewidth = 3)

        axes[i].set_title(store, size = 24)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Amount Sold", size = 18)
        axes[i].xaxis.set_major_formatter(md.DateFormatter('%m-%d'))
        axes[i].tick_params(axis='y', labelsize=16)
        axes[i].tick_params(axis='x', labelsize=16, rotation=45)

        axes[i].legend(fontsize = 16, loc = "upper right")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()







# Correlations

## total amount with variable, by store

def corr_total_amount_by_store (df, x:str):
    print(x, "\n")
    df = df[df["item_category"] == "daily total"]
    for store in df["store_name"].unique():
        store_data = df[df["store_name"] == store]
        store_data = store_data.groupby("date").agg({x:"mean", "total_amount":"sum"})
        print (store + ": ", round(store_data["total_amount"].corr(store_data[x]),2))




## Correlation between categorical variables

def cramers_corrected_stat(df,cat_col1,cat_col2):
    """
    This function spits out corrected Cramer's correlation statistic
    between two categorical columns of a dataframe 
    """
    df = df[df["item_category"] == "daily total"]
    crosstab = pd.crosstab(df[cat_col1],df[cat_col2])
    chi_sqr = ss.chi2_contingency(crosstab)[0]
    n = crosstab.sum().sum()
    r,k = crosstab.shape
    phi_sqr_corr = max(0, chi_sqr/n - ((k-1)*(r-1))/(n-1))    
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    
    result = np.sqrt(phi_sqr_corr / min( (k_corr-1), (r_corr-1)))
    return round(result,3)



def anova_pvalue(df,cat_col,num_col):
    """
    This function spits out the anova p-value (probability of no correlation) 
    between a categorical column and a numerical column of a dataframe
    """
    print(cat_col, "\n")
    df = df[df["item_category"] == "daily total"]
    for store in df["store_name"].unique():
        store_data = df[df["store_name"] == store]
        if isinstance(cat_col, int) or isinstance(cat_col, float):
            store_data = store_data.groupby("date").agg({cat_col:"mean", "total_amount":"sum"})
        else:
            store_data = store_data.groupby(["date",cat_col]).agg({"total_amount":"sum"}).reset_index()
        CategoryGroupLists = store_data.groupby(cat_col)[num_col].apply(list)
        AnovaResults = ss.f_oneway(*CategoryGroupLists)
        p_value = round(AnovaResults[1],3)
        print(store + ": ", p_value)



def anova_pvalue_allstores(df,cat_col,num_col):
    
    print(cat_col, "\n")
    df = df[df["item_category"] == "daily total"]
    if isinstance(cat_col, int) or isinstance(cat_col, float):
        df = df.groupby("date").agg({cat_col:"mean", "total_amount":"sum"})
    else:
        df = df.groupby(["date",cat_col]).agg({"total_amount":"sum"}).reset_index()
    CategoryGroupLists = df.groupby(cat_col)[num_col].apply(list)
    AnovaResults = ss.f_oneway(*CategoryGroupLists)
    p_value = round(AnovaResults[1],3)
    print(p_value)