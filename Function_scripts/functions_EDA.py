
import pandas as pd
import numpy as np
import holidays
from datetime import datetime, timedelta
import requests
from matplotlib import ticker
import matplotlib.pyplot as plt

# function to bar-plot the total amount per category, excluding daily_total
def vis_total_categories(d):

    # Filter out rows where item_category is 'daily total'
    filtered_df = d[d['item_category'] != 'daily total']

    # Count occurrences of each item category and sum the total_amount
    category_counts = filtered_df.groupby('item_category')['total_amount'].sum().sort_values(ascending = False)

    # Create bar plot
    ax =category_counts.plot(kind='bar', color='skyblue')

    # Add labels and title
    plt.title('Total Amount per Item Category')
    plt.xlabel('Item Category')
    plt.ylabel('Total Amount (in Millions)')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / 1e6:.2f}'))

    # Add the value of each bar on top of the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height() / 1e6:.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points')
    # Show plot
    plt.show()

# function to add column street_market as a dummy

def dummy_street_market(df):
    df['street_market'] = df.apply(lambda row: 1 if row['store_name'] == 'Maybachufer' and row['weekday'] in [1, 4] else 0, axis=1)
    return df

# function to create total amount per day

def daily_total(d): 
    td =d.groupby(["date","store_name","Location_name",
    "location_id","store_id"])["total_amount"].sum().reset_index() 
    td['item_category'] ='daily total'
    td["item_name"] = "daily total"
    td['type_id'] ='0'
    td['type_name'] ='daily total'
    td['item_id'] ='0'
    td['amount'] =td['total_amount']
    d_new =pd.concat([d, td]).sort_values(by = "date", ascending=False).reset_index(drop=True)
    d_new = d_new[d_new["item_category"] != "old"].reset_index(drop=True)
    return d_new

# function to create lagged variables

def lag(d):
    dl =d.loc[d['item_category'] =='daily total',
    ["date","store_name","item_category","total_amount"]]
    dl['lag1'] =dl.groupby("store_name")["total_amount"].shift(periods = -1)
    dl['lag2'] =dl.groupby("store_name")["total_amount"].shift(periods = -2)
    dl = dl.drop("total_amount", axis =1)
    d_new =pd.merge(d,dl,on=["date","store_name","item_category"], how ='left')
    return d_new


# function to transform weather variables


def weather_data(df:pd.DataFrame):
    weather_URL = "https://archive-api.open-meteo.com/v1/archive"

    params_BE = {
	"latitude": 52.52,
	"longitude": 13.41,
	"start_date": "2017-07-01",
	"end_date": "2024-06-05",
	"daily": ["temperature_2m_mean", "sunshine_duration", "precipitation_hours"],
	"timezone": "Europe/Berlin"}

    params_HH = {
	"latitude": 53.5507,
	"longitude": 9.993,
	"start_date": "2017-07-01",
	"end_date": "2024-06-05",
	"daily": ["temperature_2m_mean", "sunshine_duration", "precipitation_hours"],
	"timezone": "Europe/Berlin"}

    weather_BE = requests.get(url =weather_URL,params=params_BE).json()
    weather_BE = pd.DataFrame(weather_BE.get("daily"))
    weather_BE = weather_BE.rename(columns = {"time":"date"})
    weather_BE["date"] = pd.to_datetime(weather_BE["date"])

    weather_HH = requests.get(url =weather_URL,params=params_HH).json()
    weather_HH = pd.DataFrame(weather_HH.get("daily"))
    weather_HH = weather_HH.rename(columns = {"time":"date"})
    weather_HH["date"] = pd.to_datetime(weather_HH["date"])

    df_new = df.copy().reset_index()

    merged_BE = pd.merge(left = df_new[df_new["Location_name"] == "Berlin"], right = weather_BE, on = "date")
    merged_HH = pd.merge(left = df_new[df_new["Location_name"] == "Hamburg"], right = weather_HH, on = "date")

    df_new = pd.concat([merged_BE, merged_HH]).sort_values(by = "index").reset_index(drop = True).drop("index", axis = 1)

    return df_new





# OLD weather transformations


def transform_weather_temp(df, variable:str):
    df = df.iloc[:,[1,3]]
    df["date"] = df["MESS_DATUM"].apply(lambda x: pd.to_datetime(str(x), format = "%Y%m%d%H"))
    df[(df["date"].dt.date >= pd.to_datetime('2017-07-01').date()) &
        (df["date"].dt.time >= pd.to_datetime("08:00:00").time()) &
        (df["date"].dt.time <= pd.to_datetime("20:00:00").time()) ]
    df["date"] = df["date"].dt.date
    df = df.groupby("date").mean().reset_index().loc[:,["date",variable]]
    df["date"] = df["date"].apply(lambda x: pd.Timestamp(x))
    return df


def transform_weather_prec_sunshine(df, variable:str):
    df = df.iloc[:,[1,3]]
    df["date"] = df["MESS_DATUM"].apply(lambda x: pd.to_datetime(str(x), format = "%Y%m%d%H"))
    df[(df["date"].dt.date >= pd.to_datetime('2017-07-01').date()) &
        (df["date"].dt.time >= pd.to_datetime("08:00:00").time()) &
        (df["date"].dt.time <= pd.to_datetime("20:00:00").time()) ]
    df["date"] = df["date"].dt.date
    df = df.groupby("date").sum().reset_index().loc[:,["date",variable]]
    df["date"] = df["date"].apply(lambda x: pd.Timestamp(x))
    return df




# school holidays

def hol_school (df,hs_b,hs_h):
    df['date'] =pd.to_datetime(df['date'])
    start_b =pd.to_datetime(hs_b["Beginn"], format ="%d.%m.%Y").to_frame()
    end_b =pd.to_datetime(hs_b["Ende"], format ="%d.%m.%Y").to_frame()
    hs_b =pd.concat([start_b,end_b], axis =1)
    start_h =pd.to_datetime(hs_h["Beginn"], format ="%d.%m.%Y").to_frame()
    end_h =pd.to_datetime(hs_h["Ende"], format ="%d.%m.%Y").to_frame()
    hs_h =pd.concat([start_h,end_h], axis =1)
    def gen_range(row):
        return pd.date_range(start=row['Beginn'], end=row['Ende'])
    hs_b['date'] =hs_b.apply(gen_range, axis=1)
    hs_b_exp =hs_b.explode('date').reset_index(drop=True).drop(["Beginn","Ende"], axis =1)
    hs_b_exp["hol_school"] =1
    hs_h['date'] =hs_h.apply(gen_range, axis=1)
    hs_h_exp =hs_h.explode('date').reset_index(drop=True).drop(["Beginn","Ende"], axis =1)
    hs_h_exp["hol_school"] =1
    df_new =df.copy().reset_index()
    merged_b =pd.merge(left =df_new[df_new["Location_name"] == "Berlin"], right =hs_b_exp, how ="left", on ="date").fillna(0)
    merged_h =pd.merge(left =df_new[df_new["Location_name"] == "Hamburg"], right =hs_h_exp, how ="left", on = "date").fillna(0)
    df_new =pd.concat([merged_b, merged_h]).sort_values(by = "index").reset_index(drop = True).drop("index", axis = 1)
    return df_new


# public holidays 

def hol_pub (df):
    df['date'] =pd.to_datetime(df['date'])
    hol_b =holidays.Germany(years=range(2017, 2025), prov='BE')
    holdates_b =sorted(hol_b.keys())
    hp_b =pd.DataFrame(holdates_b, columns=['date'])
    hp_b['date'] =pd.to_datetime(hp_b['date'])
    hp_b["hol_pub"] =1
    hol_h =holidays.Germany(years=range(2017, 2025), prov='HH')
    holdates_h =sorted(hol_h.keys())
    hp_h =pd.DataFrame(holdates_h, columns=['date'])
    hp_h['date'] =pd.to_datetime(hp_h['date'])
    hp_h["hol_pub"] =1
    df_new =df.copy().reset_index()
    merged_b =pd.merge(left =df_new[df_new["Location_name"] == "Berlin"], right =hp_b, how ="left", on = "date").fillna(0)
    merged_h =pd.merge(left =df_new[df_new["Location_name"] =="Hamburg"], right =hp_h, how ="left", on ="date").fillna(0)
    df_new =pd.concat([merged_b, merged_h]).sort_values(by = "index").reset_index(drop = True).drop("index", axis = 1)
    return df_new


# calculating total amount

def calculate_total_amount(dataframe):
    item_name = dataframe['item_name']
    amount = dataframe['amount']
    
    if any(str(i) in item_name for i in [4, 6, 12]):
        if '4' in item_name:
            return amount * 4
        elif '6' in item_name:
            return amount * 6
        elif '12' in item_name:
            return amount * 12
    elif any(month in item_name for month in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']):
        return amount * 6
    elif 'box' in item_name.lower():
        return amount * 4
    else:
        return amount * 1


# dropping duplicates

def drop_duplicates(d):
    ind_drop =d[(d["item_name"] =="donuts sold (old)") &
(((d["date"] >"2021-07-12") & (d["store_name"] =="Mitte")) | 
((d["date"] >"2021-07-12") & (d["store_name"] =="Potsdamer")) |
((d["date"] >"2021-07-14") & (d["store_name"] =="Danziger")) |  
((d["date"] >"2021-07-14") & (d["store_name"] =="Maybachufer")) |  
((d["date"] >"2021-07-14") & (d["store_name"] =="Warschauer")) |  
((d["date"] >"2021-07-15") & (d["store_name"] =="KaDeWe")) |
((d["date"] >"2021-07-15") & (d["store_name"] =="Schöneberg")) |
((d["date"] >"2021-07-12") & (d["store_name"] =="Jungfernstieg")) |
((d["date"] >"2021-07-12") & (d["store_name"] =="Altona")) |
((d["date"] >"2021-07-12") & (d["store_name"] =="Neuer Kamp")) |
((d["date"] >"2021-07-12") & (d["store_name"] =="Eppendorf")) |
((d["date"] >"2021-07-12") & (d["store_name"] =="Hamburg Hauptbahnhof")) |
((d["date"] >"2021-07-12") & (d["store_name"] =="Hauptbahnhof")))].index
    d_new =d.drop(ind_drop)
    return d_new


# creating time variables

def date_info(d):
    d['date'] =pd.to_datetime(d['date'])
    d['weekday'] =d['date'].dt.dayofweek
    d['day'] =d['date'].dt.day
    d['month'] =d['date'].dt.month
    d['year'] =d['date'].dt.year
    d['week_year'] =d['date'].dt.isocalendar().week
    last_date =datetime(2024, 5, 31)
    d['days_back'] =(last_date -d['date']).dt.days
    return d

# categorization of items

classics = [
    'Strawberry Sprinkles', 'White Choc & Strawberries', 'Choc Sprinkles', 
    'Cinnamon Sugar', 'Boston Cream', 'Classic Donut', 'Chocolate Peanut Fudge', 'Salted Caramel Hazelnut', 'Salted Caramel', 
    "Bram's Favourites 12 Box", "Bram's Favourites 6 Box", "Bram's 12 Favorites", "Bram's 6 Favorites", "Bram's Favourites (12 Box)"]

specials = [
    'Blueberry Lemon Cheesecake', 'Chocolate Bomboloni', 'Apple Pie', 
    'Bienenstich', 'Special Donut', 'Halloween Haunt Box', 'Halloween Box', "Valentine's Day Box", 'Valentines Day Special Box','New Years Eve Special: 6 box',
 'New Years Eve Special (4)',
 'New Years Eve Special (6)','NYE 4 Box',
 'NYE 6 Box',
 'New Years Eve Special: 4 box',
 "Cookies & Cream", "Weekend special", "Lotus Ring",
    "White Nougat Pistachio", "Pink Sprinkles", "Chocolate Bombolini", "Pecan Pie", "Affogato", "Vanilla Glazed",
    "Peanut Butter Cup", "Cranberry Walnut"
]

monthly_specials = [
    'Passionfruit', 'Strawberry Shortcake', 'Strawberries & Cream', 
    'Lemon Tart', 'Pistachio Dream', 'May Donut Box',
 'April Donut Box',
 'December Donut Box',
 'March Donut Box',
 'October Donut Box',
 'February Donut Box',
 'January Donut Box',
 'June Donut Box',
 'November Donut Box',
 'September Donut Box',
 'August Donut Box',
 'July Donut Box']

different_products = ['Star Wars Day', 'Pizza Hawaii', 'Grilled Cheese', 'Pie Day', 'Bat', 'Cat Day', 'NYE: Tonka Cream', 'Zimtstern Donut', 'Free Donut Softeis ', 'Cookie Softie Sandwich', 'Choc Custard Filled', 'Free Donut Upgrade', 'Strawberry Bun', 'Fried Chicken & Donut Waffle', 'Waffle + Maple Syrup', 'Chicken Waffle Sriracha', 'Chicken Waffle Truffle', 'Waffle + Sriracha ', 'Chicken Waffle Maple', 'Grilled Cheese with Jalapeños', 'Waffle + Truffle3', 'Grilled Cheese + Jalapenos', 'Classic Hot Dog', 'Special Hot Dog', 'Waffle + Truffle', 'Free Berliner', 'Letter Donuts']

charity = ['Charity Donut', 'charity in box(duplicate)']

mixed = ['6 Donuts',
 '4 Donuts',
 '12 Donuts',
 '6 Box', '12 Box','Donut Drink Combo', '6x Donut Box Online', '6 Donuts + 50% Rabatt auf ein Nitro Flat White 0,25l','6 box + free Nitro can', 'Puzzle Deal'
]


def categorize_item(item_name):
    if item_name in classics:
        return 'classics'
    elif item_name in specials:
        return 'specials'
    elif item_name in monthly_specials:
        return 'monthly_specials'
    elif item_name in different_products:
        return 'not_donut'
    elif item_name == 'donuts sold (old)':
        return 'old'
    elif item_name in charity:
        return 'charity_donut'
    elif item_name in mixed:
        return 'mixed'
    else:
        return 'other'
    



# reclassify "other" category

def update_item_category(dataframe):
    # Ensure the 'date' column is in datetime format
    dataframe['date'] = pd.to_datetime(dataframe['date'])

    # Extract the year and month from the 'date' column
    dataframe['year'] = dataframe['date'].dt.year
    dataframe['month'] = dataframe['date'].dt.month

    # Filter only the rows where item_category is 'other'
    other_df = dataframe[dataframe['item_category'] == 'other']

    # Drop duplicate entries to get unique days per 'item_name', 'year', and 'month'
    unique_days = other_df[['item_name', 'year', 'month', 'date']].drop_duplicates()

    # Group by 'item_name' and 'year' and count unique dates
    unique_days_count = unique_days.groupby(['item_name', 'year']).size().reset_index(name='unique_days_count')

    # Aggregate the months in a comma-separated format
    months_aggregated = unique_days.groupby(['item_name', 'year'])['month'].apply(lambda x: ','.join(map(str, sorted(x.unique())))).reset_index(name='months')

    # Merge the aggregated months back into the result DataFrame
    result = pd.merge(unique_days_count, months_aggregated, on=['item_name', 'year'])

    # Count the number of months in the 'months' column
    result['months_count'] = result['months'].apply(lambda x: len(x.split(',')))

    # Classify into bins
    bins = ['3 or less', 'more than 3 and less than 6', '6 or more']
    result['bin'] = pd.cut(result['months_count'], bins=[0, 3, 6, float('inf')], labels=bins, right=False)

    # Identify the item_names in the '3 or less' bin
    items_3_or_less = result[result['bin'] == '3 or less']['item_name'].unique()

    # Identify the item_names with 10 or more months in the same year
    items_10_or_more = result[result['months_count'] >= 10]['item_name'].unique()

    # Identify continuous sales of 10 months or more
    continuous_sales_items = []
    for item_name in unique_days['item_name'].unique():
        item_sales = unique_days[unique_days['item_name'] == item_name].sort_values(by=['year', 'month'])
        item_sales['time'] = item_sales['year'] * 12 + item_sales['month']
        item_sales['diff'] = item_sales['time'].diff().fillna(1)
        continuous_sales = (item_sales['diff'] <= 1).astype(int).groupby(item_sales['diff'].ne(1).cumsum()).cumsum()
        if continuous_sales.max() >= 10:
            continuous_sales_items.append(item_name)

    # List of item_names that should remain 'other'
    items_keep_other = [
        'Weekend special', 'donuts in boxes (wolt)', 'Letter donuts',
        'Softeis', 'Softi - Cup', 'Oatly Softeis', 'Drinking Bottle', 'Oatly Softeis', 'Softeis', 'Softi - Cup', 'Softi - Dount', 'Special Softi', 'donuts in boxes (wolt)'
    ]

    # Function to update item_category based on additional rules
    def update_item_category_row(row):
        if row['item_name'] in items_keep_other:
            return 'other'
        if row['item_name'] in items_10_or_more:
            return 'specials'
        if row['item_name'] in continuous_sales_items:
            return 'specials'
        if row['item_name'] in items_3_or_less:
            return 'monthly_specials'
        else:
            "specials"
        return row['item_category']

    # Apply the function to update the item_category column only for 'other' category rows
    dataframe.loc[dataframe['item_category'] == 'other', 'item_category'] = dataframe[dataframe['item_category'] == 'other'].apply(update_item_category_row, axis=1)

    return dataframe


