from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the dataset
pima = pd.read_csv("diabetes.csv", header=0)

# Define features and labels
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = pima[feature_cols]
y = pima['Outcome']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=2)

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)  
knn.fit(X_train, y_train)

y_pred=knn.predict(X_test)

print("accuracy of knn:",accuracy_score(y_pred,y_test))

# # Prompt user for feature values
print("Enter feature values for a single instance:")
instance = []
for feature in feature_cols:
    value = float(input(f"{feature}: "))
    instance.append(value)

# Make prediction for the single instance
prediction = knn.predict([instance])

# Print the prediction
if prediction[0] == 0:
    print("Prediction: Non-Diabetic")
else:
    print("Prediction: Diabetic")
