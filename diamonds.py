import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("diamonds.csv")

cut_att_dict = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
clarity_att_dict = {"I3": 1, "I2": 2, "I1": 3, "SI2": 4, "SI1": 5, "VS2": 6, "VS1": 7, "VVS2": 8, "VVS1": 9, "IF": 10,
                    "FL": 11}
color_att_dict = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}

data['cut'] = data['cut'].map(cut_att_dict)
data['clarity'] = data['clarity'].map(clarity_att_dict)
data['color'] = data['color'].map(color_att_dict)

columns = ['carat', 'x', 'y']
X = data[columns]
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Random Forest

random_forest = RandomForestRegressor()
random_forest.fit(X_train, y_train)
print(random_forest.score(X_train, y_train))
print(random_forest.score(X_test, y_test))
