from utils import db_connect
engine = db_connect()


import pandas as pd

data_adquiered = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")
data_adquiered.to_csv("../data/raw/total_data.csv", index = False)
data_adquiered.head()


data_adquiered.shape
data_adquiered.info()

print(f"duplicated Name records is: {data_adquiered['name'].duplicated().sum()}")
print(f"duplicated Host ID records is: {data_adquiered['host_id'].duplicated().sum()}")
print(f"duplicated ID records is: {data_adquiered['id'].duplicated().sum()}")

data_adquiered.drop(["id", "name", "host_name", "last_review", "reviews_per_month"], axis = 1, inplace = True)
data_adquiered.head()

#variables plots

import matplotlib.pyplot as plt 
import seaborn as sns

fig, axis = plt.subplots(2, 3, figsize=(10, 7))

# Create Histogram
sns.histplot(ax = axis[0,0], data = data_adquiered, x = "host_id")
sns.histplot(ax = axis[0,1], data = data_adquiered, x = "neighbourhood_group").set_xticks([])
sns.histplot(ax = axis[0,2], data = data_adquiered, x = "neighbourhood").set_xticks([])
sns.histplot(ax = axis[1,0], data = data_adquiered, x = "room_type")
sns.histplot(ax = axis[1,1], data = data_adquiered, x = "availability_365")
fig.delaxes(axis[1, 2])

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()



#Numeric Variables
fig, axis = plt.subplots(4, 2, figsize = (10, 14), gridspec_kw = {"height_ratios": [6, 1, 6, 1]})

sns.histplot(ax = axis[0, 0], data = data_adquiered, x = "price")
sns.boxplot(ax = axis[1, 0], data = data_adquiered, x = "price")

sns.histplot(ax = axis[0, 1], data = data_adquiered, x = "minimum_nights").set_xlim(0, 200)
sns.boxplot(ax = axis[1, 1], data = data_adquiered, x = "minimum_nights")

sns.histplot(ax = axis[2, 0], data = data_adquiered, x = "number_of_reviews")
sns.boxplot(ax = axis[3, 0], data = data_adquiered, x = "number_of_reviews")

sns.histplot(ax = axis[2,1], data = data_adquiered, x = "calculated_host_listings_count")
sns.boxplot(ax = axis[3, 1], data = data_adquiered, x = "calculated_host_listings_count")

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# Numerical - Numerical Analysis - multivariable

# Create subplot canvas
fig, axis = plt.subplots(4, 2, figsize = (10, 16))

# Create Plates 
sns.regplot(ax = axis[0, 0], data = data_adquiered, x = "minimum_nights", y = "price")
sns.heatmap(data_adquiered[["price", "minimum_nights"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)

sns.regplot(ax = axis[0, 1], data = data_adquiered, x = "number_of_reviews", y = "price").set(ylabel = None)
sns.heatmap(data_adquiered[["price", "number_of_reviews"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])

sns.regplot(ax = axis[2, 0], data = data_adquiered, x = "calculated_host_listings_count", y = "price").set(ylabel = None)
sns.heatmap(data_adquiered[["price", "calculated_host_listings_count"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 0]).set(ylabel = None)
fig.delaxes(axis[2, 1])
fig.delaxes(axis[3, 1])

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

#Categorical data

fig, axis = plt.subplots(figsize = (5, 4))

sns.countplot(data = data_adquiered, x = "room_type", hue = "neighbourhood_group")

# Show the plot
plt.show()

#Correlation analysis for categorical

# Factorize the Room Type and Neighborhood Data
data_adquiered["room_type"] = pd.factorize(data_adquiered["room_type"])[0]
data_adquiered["neighbourhood_group"] = pd.factorize(data_adquiered["neighbourhood_group"])[0]
data_adquiered["neighbourhood"] = pd.factorize(data_adquiered["neighbourhood"])[0]

fig, axes = plt.subplots(figsize=(15, 15))

sns.heatmap(data_adquiered[["neighbourhood_group", "neighbourhood", "room_type", "price", "minimum_nights",	
                        "number_of_reviews", "calculated_host_listings_count", "availability_365"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

# Draw Plot
plt.show()

#Outliners and patterns 

sns.pairplot(data = data_adquiered)
data_adquiered.describe()

fig, axes = plt.subplots(3, 3, figsize = (15, 15))

sns.boxplot(ax = axes[0, 0], data = data_adquiered, y = "neighbourhood_group")
sns.boxplot(ax = axes[0, 1], data = data_adquiered, y = "price")
sns.boxplot(ax = axes[0, 2], data = data_adquiered, y = "minimum_nights")
sns.boxplot(ax = axes[1, 0], data = data_adquiered, y = "number_of_reviews")
sns.boxplot(ax = axes[1, 1], data = data_adquiered, y = "calculated_host_listings_count")
sns.boxplot(ax = axes[1, 2], data = data_adquiered, y = "availability_365")
sns.boxplot(ax = axes[2, 0], data = data_adquiered, y = "room_type")

plt.tight_layout()

plt.show()

# Stats for Price
price_stats = data_adquiered["price"].describe()
price_stats

# IQR for Price

price_iqr = price_stats["75%"] - price_stats["25%"]
upper_limit = price_stats["75%"] + 1.5 * price_iqr
lower_limit = price_stats["25%"] - 1.5 * price_iqr

print(f"The upper and lower limits for finding outliers are {round(upper_limit, 2)} and {round(lower_limit, 2)}, with an interquartile range of {round(price_iqr, 2)}")

# Clean the outliers

total_data = data_adquiered[data_adquiered["price"] > 0]

#Night stays

nights_stats = data_adquiered["minimum_nights"].describe()
nights_stats

# IQR for minimum_nights
nights_iqr = nights_stats["75%"] - nights_stats["25%"]

upper_limit = nights_stats["75%"] + 1.5 * nights_iqr
lower_limit = nights_stats["25%"] - 1.5 * nights_iqr

print(f"The upper and lower limits for finding outliers are {round(upper_limit, 2)} and {round(lower_limit, 2)}, with an interquartile range of {round(nights_iqr, 2)}")

# Clean the outliers

total_data = total_data[total_data["minimum_nights"] <= 15]

# Stats for number_of_reviews

review_stats = total_data["number_of_reviews"].describe()
review_stats

# IQR for number_of_reviews

review_iqr = review_stats["75%"] - review_stats["25%"]

upper_limit = review_stats["75%"] + 1.5 * review_iqr
lower_limit = review_stats["25%"] - 1.5 * review_iqr

print(f"The upper and lower limits for finding outliers are {round(upper_limit, 2)} and {round(lower_limit, 2)}, with an interquartile range of {round(review_iqr, 2)}")

# Stats for calculated_host_listings_count

hostlist_stats = total_data["calculated_host_listings_count"].describe()
hostlist_stats

# IQR for calculated_host_listings_count

hostlist_iqr = hostlist_stats["75%"] - hostlist_stats["25%"]

upper_limit = hostlist_stats["75%"] + 1.5 * hostlist_iqr
lower_limit = hostlist_stats["25%"] - 1.5 * hostlist_iqr

print(f"The upper and lower limits for finding outliers are {round(upper_limit, 2)} and {round(lower_limit, 2)}, with an interquartile range of {round(hostlist_iqr, 2)}")

count_04 = sum(1 for x in total_data["calculated_host_listings_count"] if x in range(0, 5))
count_1 = total_data[total_data["calculated_host_listings_count"] == 1].shape[0]
count_2 = total_data[total_data["calculated_host_listings_count"] == 2].shape[0]

print("Count of 0: ", count_04)
print("Count of 1: ", count_1)
print("Count of 2: ", count_2)

# Clean the outliers

total_data = total_data[total_data["calculated_host_listings_count"] > 4]

# Count NaN
total_data.isnull().sum().sort_values(ascending = False)

#scaling

from sklearn.preprocessing import MinMaxScaler

num_variables = ["number_of_reviews", "minimum_nights", "calculated_host_listings_count", 
                 "availability_365", "neighbourhood_group", "room_type"]
scaler = MinMaxScaler()
scal_features = scaler.fit_transform(total_data[num_variables])
df_scal = pd.DataFrame(scal_features, index = total_data.index, columns = num_variables)
df_scal["price"] = total_data["price"]
df_scal.head()


#Feature selection

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split

X = df_scal.drop("price", axis = 1)
y = df_scal["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


selection_model = SelectKBest(chi2, k = 4)
selection_model.fit(X_train, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix])

X_train_sel.head()

