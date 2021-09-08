# The idea of this project is to help new data scientist understand basic data science concepts. **Streamlit** will be used to create interactive visuals.

# Import streamlit!
import streamlit as st

# Title
st.title("Data Science Concepts")

# Import libraries
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

# Some information
'''
Text Text Text Text Text Text Text 
Text Text Text Text Text Text Text
Text Text Text Text Text Text Text 

'''

# What to write here

# Load data to be used in the streamlit app.
# @st.cache
# def load_data(path, num_rows):
#     df = pd.read_csv(path, nrows=num_rows, index_col=0)
#     df = df.rename(columns={'Start Station Latitude': 'lat',
#         'Start Station Longitude': 'lon'})
#     df['Start Time'] = pd.to_datetime(df['Start Time'])
#     df['Stop Time'] = pd.to_datetime(df['Stop Time'])
#     return df

# Load cached data with a function defined.
# df = load_data('data/NYC_bikes.csv', 10000)

##############################

# Set the random seed for reproduceable output.
np.random.seed(123)

# Create synthetic data to to use for illustration

# Creates a range from -2 to 2, with 100 steps.
x = np.linspace(-1, 1, 100)*2

# Creates the target variable
y = 5*x**3 + 2.5*x**2 + 11 + np.random.randn(100)*5.0 


# This the function above
def f(t):
    return 5*t**3 + 2.5*t**2 + 11

# Plot
fig = plt.figure()
plt.scatter(x, y, c="cornflowerblue", label = "data") # Plot the points
plt.plot(x, f(x), 'b--', label="function") # Plot the driving function (no error)
plt.xlabel("x (independent variable)") 
plt.ylabel("y (target)")
plt.title("Plotted Synthetic Data")
plt.legend()
st.write(fig)

##################################

##################################

##################################
@st.cache(allow_output_mutation=True)
def load_model(model_type):
    if model_type == 1:
        model = joblib.load('saved_models/overfitting_underfit_model.joblib') 
    elif model_type == 2:
        model = tf.keras.models.load_model('saved_models/overfitting_right_model')
    if model_type == 3:
        model = tf.keras.models.load_model('saved_models/overfitting_overfit_model_50000')
    if model_type == 4:
        model = tf.keras.models.load_model('saved_models/overfitting_overfit_model_75000')
    if model_type == 5:
        model = tf.keras.models.load_model('saved_models/overfitting_overfit_model_100000')
    if model_type == 6:
        model = tf.keras.models.load_model('saved_models/overfitting_overfit_model_125000')

    return model
##################################

##################################
# @st.cache
# def plot_model(x, y, model_type):

#     # if model_type == 1:
#     #     x = x.reshape(-1,1)

#     # # Make predictions
#     # y_pred = model.predict(x)

#     # Plot and fit curve

#     fig = plt.figure()
#     plt.scatter(x, y, label='True labels', c="cornflowerblue") # Plot the data
#     # plt.plot(x, f(x), 'b--', label="Function", alpha=0.5) # Plot the driving function (no error)
#     # plt.plot(x, y_pred, linewidth=2, color='r', linestyle='--', label='Model') # Plot the predictions
#     plt.legend()
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('Curve fitting')

#     # Show
#     return fig
##################################

##################################

## Create a main sidebar

menu_list = ('Overfitting', 'Regularization', 'Neural network', 'Difference between svm and logistic regression',
 'Accuracy Vs Recall/ etc and the confusion matrix', 'ROC and AUC', 'Profitability curves', 'Lift')


match menu_list:
        case 'Overfitting':

            st.sidebar.title('Topics')  
            st.sidebar.selectbox('Choose a topic', menu_list)

            ## Create a sidebar
            st.sidebar.title('Filters')
            st.sidebar.markdown('''Choose model type''')

            # A checkbox in sidebar to toggle roundtrip.
            model_type = st.sidebar.radio('pick a model type', \
                ('Underfit', 'Fit (about right)', 'Overfit_50K', 'Overfit_75K', 'Overfit_100K', 'Overfit_125K'))
            # round trip condition based on checkbox
            if model_type == 'Underfit':
                model = load_model(1)
                # fig1 = plot_model(x, y, 1)
                x_reshaped = x.reshape(-1,1)

                # # Make predictions
                y_pred = model.predict(x_reshaped)
                fig1 = plt.figure()
                plt.scatter(x, y, label='True labels', c="cornflowerblue") # Plot the data
                plt.plot(x, f(x), 'b--', label="Function", alpha=0.5) # Plot the driving function (no error)
                plt.plot(x, y_pred, linewidth=2, color='r', linestyle='--', label='Model') # Plot the predictions
                plt.legend()
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Curve fitting')
                st.write(fig1)

            elif model_type == 'Fit (about right)':
                model = load_model(2)
                # # Make predictions
                y_pred = model.predict(x)
                fig1 = plt.figure()
                plt.scatter(x, y, label='True labels', c="cornflowerblue") # Plot the data
                plt.plot(x, f(x), 'b--', label="Function", alpha=0.5) # Plot the driving function (no error)
                plt.plot(x, y_pred, linewidth=2, color='r', linestyle='--', label='Model') # Plot the predictions
                plt.legend()
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Curve fitting')
                st.write(fig1)

            elif model_type == 'Overfit_50K':
                model = load_model(3)
                y_pred = model.predict(x)
                fig1 = plt.figure()
                plt.scatter(x, y, label='True labels', c="cornflowerblue") # Plot the data
                plt.plot(x, f(x), 'b--', label="Function", alpha=0.5) # Plot the driving function (no error)
                plt.plot(x, y_pred, linewidth=2, color='r', linestyle='--', label='Model') # Plot the predictions
                plt.legend()
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Curve fitting')
                st.write(fig1)

            elif model_type == 'Overfit_75K':
                model = load_model(4)
                y_pred = model.predict(x)
                fig1 = plt.figure()
                plt.scatter(x, y, label='True labels', c="cornflowerblue") # Plot the data
                plt.plot(x, f(x), 'b--', label="Function", alpha=0.5) # Plot the driving function (no error)
                plt.plot(x, y_pred, linewidth=2, color='r', linestyle='--', label='Model') # Plot the predictions
                plt.legend()
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Curve fitting')
                st.write(fig1)

            elif model_type == 'Overfit_100K':
                model = load_model(5)
                y_pred = model.predict(x)
                fig1 = plt.figure()
                plt.scatter(x, y, label='True labels', c="cornflowerblue") # Plot the data
                plt.plot(x, f(x), 'b--', label="Function", alpha=0.5) # Plot the driving function (no error)
                plt.plot(x, y_pred, linewidth=2, color='r', linestyle='--', label='Model') # Plot the predictions
                plt.legend()
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Curve fitting')
                st.write(fig1)

            elif model_type == 'Overfit_125K':
                model = load_model(6)
                y_pred = model.predict(x)
                fig1 = plt.figure()
                plt.scatter(x, y, label='True labels', c="cornflowerblue") # Plot the data
                plt.plot(x, f(x), 'b--', label="Function", alpha=0.5) # Plot the driving function (no error)
                plt.plot(x, y_pred, linewidth=2, color='r', linestyle='--', label='Model') # Plot the predictions
                plt.legend()
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Curve fitting')
                st.write(fig1)

# # A checkbox in sidebar to toggle day of the week slider.
# use_day = st.sidebar.checkbox('Filter By Day')
# # A slider in sidebar to change day of the week.
# if use_day == True:
#     day_filter = st.sidebar.slider('Day of the week', 1,7,1)
#     df = df[df['Start Time'].dt.weekday == (day_filter -1)]

# # A checkbox in sidebar to toggle hour slider.
# use_hour = st.sidebar.checkbox('Set hour')
# if use_hour == True:
#     # Have the sidebar determine an hour of the day.
#     hour = st.sidebar.slider('Hour', 0, 24, 12)
#     df = df[df['Start Time'].dt.hour == hour]


# Create a plot to show the distribution of birth years.
# fig = plt.figure()
# plt.hist(df['Birth Year'], color="cornflowerblue", bins = 50)
# plt.title("Distribution of Birth Year of Customers")
# plt.xlabel("Year")
# plt.ylabel("Frequency")
# st.write(fig)