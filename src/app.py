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