#House price prediction using Simple LinearRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


df = pd.read_csv('house_sqft.csv')

X = df[['Square_Footage']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 9)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

new_pred_value = 1300

new_pred = model.predict(pd.DataFrame([[new_pred_value]], columns=['Square_Footage']))

print(f"Prediction for {new_pred_value}sqft : {new_pred}$")

plt.scatter(X_train, y_train, color = 'blue', label = 'Train points')
plt.scatter(X_test, y_test, color = 'green', label = 'Test points')
plt.plot(X_train, model.predict(X_train), color = 'red', label = 'Regression line')
plt.xlabel('Square Footage')
plt.ylabel('Prices $')
plt.legend()
plt.show()