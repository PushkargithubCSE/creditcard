import pandas as pd

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Display the first few rows of the dataset
print(data.head())

# Display summary information about the dataset
print(data.info())

# Check class distribution
class_distribution = data['Class'].value_counts()
print("Class Distribution:\n", class_distribution)

from sklearn.model_selection import train_test_split

X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import precision_score, recall_score, f1_score

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
