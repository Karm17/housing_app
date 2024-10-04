import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# load the dataset
cal= fetch_california_housing()
df =pd.DataFrame(data=cal.data, columns= cal.feature_names)
df['price']= cal.target
df.head()
#tittle of the app
st.title("california House Price Predictyion for Brokerage")
# data overview
st.write("overview")
st.dataframe(df.head(10))

#split the data into train and test
X = df.drop(['price'], axis=1) #input variables
y=df['price']#target variable
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

#standarize the data
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Model Selection
st.subheader("## Select a Model")

model = st.selectbox("Choose a model", ["Linear Regression","Ridge", "Lasso", "ElasticNet"])

#Initialize the model
models={"Linear Regression": LinearRegression(),
        "Ridge":Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet()}

#Train the selected Model
selected_model=models[model]

#train the model
selected_model.fit(X_train_sc, y_train)

#predict the values
y_pred = selected_model.predict(X_test_sc)

#evaluate the model using metrics
test_mse=mean_absolute_error(y_test, y_pred)
test_mae=mean_absolute_error(y_test, y_pred)
test_rmse=np.sqrt(test_mse)
test_r2=r2_score(y_test, y_pred)

# display the metrics for selected model
st.write("Test MSE:", test_mse)
st.write("Test MAE:", test_mae)
st.write("Test RMSE:", test_rmse)
st.write("Test R@:", test_r2)

#Prompt the user to enter 
st.write("Enter the input values to predict the house price:")

user_input = {}

for feature in X.columns:
    user_input[feature]= st.number_input(feature)

#convert the dictionary to a DataFrame
user_input_df= pd.DataFrame([user_input])

#scale the user input
user_input_sc = scaler.transform(user_input_df)

#predict the house price
predicted_price= selected_model.predict(user_input_sc)

#display the predicted house price
st.write(f"Predicted House Price is {predicted_price[0] * 100000}")