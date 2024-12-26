import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

features = ["MSSubClass", "LotArea", "LotShape",
            "Utilities", "OverallQual", "OverallCond",
            "YearBuilt", "YearRemodAdd", "ExterQual",
            "ExterCond", "BsmtQual", "BsmtCond",
            "TotalBsmtSF", "HeatingQC",
            "KitchenAbvGr", "KitchenQual",
            "TotRmsAbvGrd", "Functional",
            "GarageArea", "GarageQual", "GarageCond",
            "PoolArea", "YrSold"]

changesDictionary = {
    'LotShape':  {"Reg": 0, "IR1": 1, "IR2": 2, "IR3": 3},
    'Utilities': {"AllPub": 0, "NoSewr": -1, "NoSeWa": -2, "ELO": -3},
    'ExterQual': {"Ex": 5, "Gd": 4,"TA": 3, "Fa": 2, "Po": 1},
    'ExterCond': {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
    'BsmtQual': {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, None: -1},
    'BsmtCond': {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, None: -1},
    'HeatingQC': {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
    'KitchenQual': {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
    'Functional': {"Typ": 0, "Min1": -1, "Min2": -2, "Mod": -3, "Maj1": -4, "Maj2": -5, "Sev": -6, "Sal": -7},
    'GarageQual': {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, None: -1},
    'GarageCond': {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, None: -1},
}




with pd.option_context('future.no_silent_downcasting', True):
    train_data.replace(changesDictionary, inplace=True)
    test_data.replace(changesDictionary,  inplace=True)

train_data[features] = train_data[features].apply(pd.to_numeric)
test_data[features] = test_data[features].apply(pd.to_numeric)

train_y = train_data.SalePrice
train_X = train_data[features]

test_X = test_data[features]

model = LinearRegression()
model.fit(train_X, train_y)

predict = model.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': predict})
output.to_csv('submission.csv', index=False)
