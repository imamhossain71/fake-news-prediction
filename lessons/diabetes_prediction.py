
#  Import Libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



#  Load Dataset

df = pd.read_csv("data/synthetic_diabetes_dataset_2000.csv")

print(df.head())   # for 1st 5 rows
print(df.info())   # for dataset structure



#  Data Cleaning

# some clum 0 means missing value (invalid)
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

for col in cols:
    df[col] = df[col].replace(0, np.nan)   # 0 → NaN 
    df[col].fillna(df[col].median(), inplace=True)  #  fill by median



# Feature & Target

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



#  Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



#  Train Model (Random Forest)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)



#  Prediction

y_pred = model.predict(X_test)



#  Evaluation

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))



# user input for custom prediction
print("\n🔹 Enter Patient Details:")

pregnancies = int(input("Pregnancies: "))
glucose = float(input("Glucose: "))
blood_pressure = float(input("Blood Pressure: "))
skin_thickness = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
#dpf = float(input("Diabetes Pedigree Function: "))
age = int(input("Age: "))

user_data = [[
    pregnancies, glucose, blood_pressure,
    skin_thickness, insulin, bmi, age
]]

user_data_scaled = scaler.transform(user_data)

prediction = model.predict(user_data_scaled)

if prediction[0] == 1:
    print("\n Result: Diabetes Positive")
else:
    print("\nResult: No Diabetes")


#  Custom Prediction

sample = [[120, 70, 20, 79, 25.0, 0.5, 30, 45]]

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

print("\nPrediction:", "Diabetes Positive " if prediction[0] == 1 else "No Diabetes ")