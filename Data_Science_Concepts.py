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

# This function prints markdown txt
def msg(message, font_size):

    msg_str = f"""
    <style>
    p.a {{
    font: bold {font_size}px Courier;
    }}
    </style>
    <p class="a">{message}</p>
    """

    st.markdown(msg_str, unsafe_allow_html=True)

##############################
##############################

# Some information
message = 'Hi Hi'
msg(message, 15)

##############################
##############################

# Create synthetic data to to use for illustration

# Creates a range from -2 to 2, with 100 steps.
x = np.linspace(-1, 1, 100)*2

# Creates the target variable
y = 5*x**3 + 2.5*x**2 + 11 + np.random.randn(100)*5.0 

# This function above
def f(t):
    return 5*t**3 + 2.5*t**2 + 11

##################################

# This function prints markdown txt
def msg(message, font_size):

    msg_str = f"""
    <style>
    p.a {{
    font: bold {font_size}px Courier;
    }}
    </style>
    <p class="a">{message}</p>
    """

    st.markdown(msg_str, unsafe_allow_html=True)

##################################

# This function displays Overfitting plots
def plot_overfitting(x, y, label_1, fx, label_2, y_pred, xlabel, ylabel, title):
    fig1 = plt.figure()
    plt.scatter(x, y, label=label_1, c="cornflowerblue") # Plot the data
    plt.plot(x, fx, 'b--', label="Function", alpha=0.5) # Plot the driving function (no error)
    plt.plot(x, y_pred, linewidth=2, color='r', linestyle='--', label='Model') # Plot the predictions
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    st.write(fig1)

##################################
##################################

# This function loads all saved models
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

## Create a main sidebar

menu_list = ('Select', 'Overfitting', 'SVM Vs Logistic regression', 'Regularization',
 'Accuracy Vs Recall/ etc and the confusion matrix', 
 'ROC and AUC', 'Neural networks', 'Profitability curves', 'Lift')

st.sidebar.title('Topics')  
topics_drop_down = st.sidebar.selectbox('Choose a topic', menu_list, index = 0)

if topics_drop_down == 'Overfitting':

    # # Plot
    # fig = plt.figure()
    # plt.scatter(x, y, c="cornflowerblue", label = "data") # Plot the points
    # plt.plot(x, f(x), 'b--', label="function") # Plot the driving function (no error)
    # plt.xlabel("x (independent variable)") 
    # plt.ylabel("y (target)")
    # plt.title("Plotted Synthetic Data")
    # plt.legend()
    # st.write(fig)

    ## Create a sidebar
    st.sidebar.title('Filters')
    st.sidebar.markdown('''Choose model type''')

    # A checkbox in sidebar to toggle roundtrip.
    topics_Overfitting = st.sidebar.radio('pick a model type', \
        ('Underfit', 'Fit (about right)', 'Overfit_50K', 'Overfit_75K', 'Overfit_100K', 'Overfit_125K'))
    # round trip condition based on checkbox

    # Load and predict all models to speed togglling between options
    x_reshaped = x.reshape(-1,1)
    model_1 = load_model(1)
    y_pred_1 = model_1.predict(x_reshaped)
    model_2 = load_model(2)
    y_pred_2 = model_2.predict(x_reshaped)
    model_3 = load_model(3)
    y_pred_3 = model_3.predict(x_reshaped)
    model_4 = load_model(4)
    y_pred_4 = model_4.predict(x_reshaped)
    model_5 = load_model(5)
    y_pred_5 = model_5.predict(x_reshaped)
    model_6 = load_model(6)
    y_pred_6 = model_6.predict(x_reshaped)

    if topics_Overfitting == 'Underfit':

        plot_overfitting(x, y, 'Data', f(x),'Function', 
                        y_pred_1, 'X', 'Y', 'Underfit Model')

    elif topics_Overfitting == 'Fit (about right)':

        plot_overfitting(x, y, 'Data', f(x),'Function', 
                        y_pred_2, 'X', 'Y', 'Fit (about right) Model')

    elif topics_Overfitting == 'Overfit_50K':

        plot_overfitting(x, y, 'Data', f(x),'Function', 
                        y_pred_3, 'X', 'Y', 'Overfit_50K epochs Model')

    elif topics_Overfitting == 'Overfit_75K':

        plot_overfitting(x, y, 'Data', f(x),'Function', 
                        y_pred_4, 'X', 'Y', 'Overfit_75K epochs Model')

    elif topics_Overfitting == 'Overfit_100K':
        
        plot_overfitting(x, y, 'Data', f(x),'Function', 
                        y_pred_5, 'X', 'Y', 'Overfit_100K epochs Model')

    elif topics_Overfitting == 'Overfit_125K':

         plot_overfitting(x, y, 'Data', f(x),'Function', 
                        y_pred_6, 'X', 'Y', 'Overfit_125K epochs Model')

else:
    if topics_drop_down != menu_list[0]:
        msg(topics_drop_down +  ' is coming soon!', 30)