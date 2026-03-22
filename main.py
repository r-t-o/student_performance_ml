import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score
import numpy as np
import joblib
'''
data = {
    "study_hours":[1,2,3,4,5,6,7,8,2,3,5,6,7,8,4],
    "sleep_hours":[8,7,6,6,5,5,4,4,7,6,5,5,4,4,6],
    "attendance":[60,65,70,75,80,85,90,95,68,72,78,88,92,96,74],
    "performance":[50,55,60,65,70,75,80,85,58,62,72,78,82,88,66]
}
'''
np.random.seed(42)
study_hours = np.random.randint(1,10,100)
sleep_hours = np.random.randint(4,9,100)
attendance = np.random.randint(50,100,100)
performance = (
    study_hours * 5 + attendance * 0.5 - sleep_hours * 2 + np.random.randint(-5,5,100)
)
df = pd.DataFrame({
    "study_hours": study_hours,
    "sleep_hours": sleep_hours,
    "attendance": attendance,
    "performance": performance
})
print(df.head())
print(df.shape)
X = df.drop("performance",axis =1)
y = df["performance"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(y_pred)
error = mean_absolute_error(y_test,y_pred)
print("mean absolute error:",error)
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train,y_train)
dt_pred = dt_model.predict(X_test)
print(dt_pred)
rf_model = RandomForestRegressor()
rf_model.fit(X_train,y_train)
rf_pred = rf_model.predict(X_test)
print(rf_pred)
print("Linear Regression:", mean_absolute_error(y_test, y_pred))
print("Linear Regression R2:", r2_score(y_test, y_pred))
print("Decision Tree:", mean_absolute_error(y_test, dt_pred))
print("Decision Tree R2:", r2_score(y_test, dt_pred))
print("Random Forest:", mean_absolute_error(y_test, rf_pred))
print("random forest r2:",r2_score(y_test,rf_pred))
models = {
    "Linear Regression": r2_score(y_test, y_pred),
    "Decision Tree": r2_score(y_test, dt_pred),
    "Random Forest": r2_score(y_test, rf_pred)
}
best_model = max(models, key=models.get)
print("Best Model:", best_model)
joblib.dump(model, "model.pkl")