import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'keyword_presence': [1, 0, 1, 0, 1, 0, 1, 0],  # 1 if keyword present, 0 otherwise
    'length': [100, 150, 120, 130, 110, 140, 95, 160],  # Length of email
    'suspicious_links': [1, 0, 1, 0, 1, 0, 1, 0],  # 1 if suspicious link present, 0 otherwise
    'is_scam': [1, 0, 1, 0, 1, 0, 1, 0]  # Target variable: 1 for scam, 0 for non-scam
}

db = pd.DataFrame(data)

X = db[['keyword_presence', 'length', 'suspicious_links']]
y = db['is_scam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

new_email = pd.DataFrame([[1, 120, 1]], columns=['keyword_presence', 'length', 'suspicious_links'])

prediction_probability = model.predict_proba(new_email)[0][1]
prediction_class = model.predict(new_email)

percentage_probability = round(prediction_probability, 2) * 100
print(f"Probability of email be scam : {percentage_probability:.0f}%")
print(f"Predicted class (scam = 1 , not scam = 0): {prediction_class[0]}")

# Data for the pie chart
labels = ['Scam', 'Not Scam']
sizes = [percentage_probability, 100 - percentage_probability]
colors = ['red', 'green']
explode = (0.1, 0)  # explode the 'Scam' slice

# Plotting the pie chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True)
plt.title(f"Probability of Email Being a Scam: {percentage_probability:.0f}%")
plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.

# Show the plot
plt.show()