
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline


# Main content
st.title("Exploratory Data Analysis")

# File upload
uploaded_file = st.file_uploader("Upload the Data:", type=["csv"])

replace_na_option = st.checkbox("Replace NaN with 0", value=False)


# Read data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Replace NaN with 0 if the checkbox is selected
    if replace_na_option:
        df = df.fillna(0)

    # Display first few rows of the dataset
    st.subheader("First few rows of the dataset:")
    st.dataframe(df.head())

    # Display numerical columns
    st.subheader("Numerical Columns:")
    numerical_columns = df.select_dtypes(include='number').columns
    st.write(numerical_columns)

    # Display categorical columns
    st.subheader("Categorical Columns:")
    categorical_columns = df.select_dtypes(exclude='number').columns
    st.write(categorical_columns)
    df.fillna(0, inplace=True)
    cat = st.multiselect(
        "Choose categorical variables",
        df.select_dtypes(include=['category', 'object', 'int64', 'float64']).columns)

    for col in cat:
        df[col] = df[col].astype('category')
    
    st.write(df[cat])
    
    remaining_columns = list(set(df.columns) - set(cat))

    for col in remaining_columns:
        if df[col].dtype != 'object':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        else:
            df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')

    st.subheader("Numerical Columns:")      
    st.write(df[remaining_columns])




    numerical_columns=remaining_columns
    categorical_columns=cat
        


    # Display summary statistics for numerical columns

    st.subheader("Summary Statistics for Numerical Columns:")
    st.dataframe(df[numerical_columns].describe())

    # Data Visualization
    st.subheader("Data Visualization:")
    
    # Selectbox for x-axis and y-axis
    x_axis = st.selectbox('Select X-axis', options=df.columns)
    y_axis = st.selectbox('Select Y-axis', options=df.columns)

    # Plotting the interactive chart
    fig = px.scatter(df, x=x_axis, y=y_axis, title=f'{x_axis} vs {y_axis}')
    st.plotly_chart(fig)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Correlation plot 
    st.subheader("Correlation Plot:")
    # Calculate correlation matrix
    correlation_matrix = df[numerical_columns].corr()
    # Plot correlation matrix heatmap
    fig_size = st.slider('Select Plot Size', min_value=8, max_value=30, value=12)
    plt.figure(figsize=(fig_size+15, fig_size))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(f'Correlation Matrix for {numerical_columns}')
    st.pyplot()

    
else:
    st.warning("Please upload a CSV file.")








   
