from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([12000, 15000, 10050, 14600, 16000])

fig, axs = plt.subplots(2, 2, figsize=(14, 7))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

model = LinearRegression()

model.fit(X_train, y_train)

new_X = [[8]]

y_pred = model.predict(new_X)

print(y_pred[0])

nx = [X[-1][0], new_X[0][0]]
ny = [y[-1], y_pred[0]]

axs[0,0].plot(X, y, color='blue', linestyle='--',marker='o', label='Original Data')
axs[0,0].plot(nx, ny, color='red', linestyle='--',marker='o', label="Connection between prediction")
axs[0,0].scatter(new_X, y_pred[0], color='green', marker='o', s=100, label='New Prediction', zorder=5)
axs[0,0].set_title("Graph Representation")
axs[0,0].set_xlabel("House No.")
axs[0,0].set_ylabel("House Price")

X = X.reshape(-1)
narrx = np.append(X, new_X)
narry = np.append(y, y_pred[0])

axs[0,1].bar(narrx, narry ,color='blue')
axs[0,1].set_title("Bar Representation")
axs[0,1].set_xlabel("House No.")
axs[0,1].set_ylabel("House Price")

axs[1,0].barh(narrx, narry ,color='lightblue')
axs[1,0].set_title("HBar Representation")
axs[1,0].set_xlabel("House Price")
axs[1,0].set_ylabel("House No.")

axs[1,1].pie(y, labels=X, autopct='%1.1f%%', colors=['yellow', 'red', 'blue', 'green', 'grey', 'purple'])
axs[1,1].set_title("Pie Representation")

plt.tight_layout()

plt.show()