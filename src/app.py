from utils import db_connect
engine = db_connect()

import pandas as pd

total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")
total_data.head()

import pandas as pd

data_adquiered = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")
data_adquiered.to_csv("../data/raw/total_data.csv", index = False)
data_adquiered.head()