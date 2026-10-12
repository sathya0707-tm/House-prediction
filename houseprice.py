import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,f1_score
import matplotlib.pyplot as plt

#load dataset
data=pd.read_csv("data.csv")

#understand the data
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())
print(data.shape)
print(data.duplicated().sum())

#exploratory data analysis
data.hist(figsize=(10,8))
plt.tight_layout()
plt.show()

#prepare the data for modelling
x = data[
    [
        "sqft_living",
        "bedrooms",
        "bathrooms",
        "floors",
        "grade",
        "condition",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "lat",
        "long"
    ]
]
y=data["price"]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

#train the model
model=LinearRegression()
model.fit(x_train,y_train)

#prediction
prediction=model.predict(x_test)
print("MAE: ",mean_absolute_error(y_test,prediction))
print("MSE: ",mean_squared_error(y_test,prediction))
print("r2_score:",r2_score(y_test,prediction))
plt.scatter(y_test, prediction, alpha=0.4)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()


# Get house details from the user
sqft_living = float(input("Enter living area in sqft: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = float(input("Enter number of bathrooms: "))
floors = float(input("Enter number of floors: "))
grade = int(input("Enter house grade (1-13): "))
condition = int(input("Enter house condition (1-5): "))
sqft_above = float(input("Enter above-ground area in sqft: "))
sqft_basement = float(input("Enter basement area in sqft: "))
yr_built = int(input("Enter year built: "))
lat = float(input("Enter latitude: "))
long = float(input("Enter longitude: "))

# Arrange the input in the same order as X
new_house = pd.DataFrame([{
    "sqft_living": sqft_living,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "floors": floors,
    "grade": grade,
    "condition": condition,
    "sqft_above": sqft_above,
    "sqft_basement": sqft_basement,
    "yr_built": yr_built,
    "lat": lat,
    "long": long
}])

# Predict the price
predicted_price = model.predict(new_house)

print("Predicted house price:", round(predicted_price[0], 2))
