import streamlit as st
import pandas as pd
from utils import read_config
from io import BytesIO
import base64


st.set_page_config(
    page_title="Home",
    page_icon="📊",
    layout='centered'
)
st.image("./Westernacher_Logo_3.png")


st.write("# Welcome to use this extra ML tool")
# st.sidebar.success("Select the step")

st.markdown("""
    ### Flexible interface with SAP IBP CI-DS
    
    We only provide upload data manually now
            
    Send a request to IBP and get data from API
            """)

# st.image("https://github.com/tamakiseda/streamlit-/blob/main/pic3.png?raw=true")
st.image("./Westernacher_Logo_5.png")




file_path = "test_data.csv"
exampledata = pd.read_csv(file_path)
csv_file = exampledata.to_csv(index=False)
st.markdown("""
    ###### Download Example Data
            """)
st.download_button(label='Click to Download CSV file', 
    data=csv_file, file_name='example.csv', mime='text/csv', key='download_button')

st.write("Note: After clicking the 'Click to Download CSV file' button, your browser may prompt you to download the file.")


# main page
tbls = st.tabs(['Upload Manually','By IBP (In Developing)'])
# parameters

with tbls[0]:

    uploaded_file = st.file_uploader(
        'File uploader', type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=False)
    if uploaded_file is not None:
        st.success('File successfully uploaded!')


with tbls[1]:
    st.markdown("""
    ### Main Contains:
    - Planning Level: The attributes of selected Key figure.
    - Key Figure: The key figure you want to analysis.
    - Time Range: The time range of the data.
    """)
    
    select_columns_pl = st.multiselect('Choose Planning Level',
                                        options=['Product ID','Customer ID','Location ID'])
    select_columns_kf = st.multiselect('Choose Key Figure',
                                    options=['Actuals Shipment Qty','Open Sales Order'])
    central_variable = st.selectbox("Choose Start Period", options=['M1 2023','M2 2023','M3 2023','M4 2023'])
    constant_value = st.number_input("Number of the forward periods", value=5, step=1,
                                        key='constant_value')
    button2_clicked = st.button("Import data from IBP")
    if button2_clicked:
        st.write("Data Imported")






st.markdown(
    """

    ## Main Steps

    ### Export data from IBP
    - Upload data from Home page manually
    - Get data form IBP directly 

    ### Pre-processing 
    - Data Standardization
    - Categorical Variable
    - Dimensionality Reduction

    ### Select Algorithms
    - Cluster
    - Regression
    - Classification


    ### Select Parameters 
    - Number of clusters
    - Iteration times
    
    ### Modeling
    - To run the model


    ### Evalutate
    - Silhouette Score
    - Davies-Bouldin Index	
    - Calinski-Harabasz Index	

"""
)

# Use Pandas to read the file into a DataFrame
if uploaded_file is None:
    st.stop()
elif uploaded_file.name.endswith(('.xls', '.xlsx')):
    df_ori = pd.read_excel(uploaded_file)
else:
    df_ori = pd.read_csv(uploaded_file)

st.markdown("""
    ### Do you want to learn more❔
    - Check [Machine Learing](https://scikit-learn.org/stable/index.html)
    - Check [Doc of Streamlit](https://docs.streamlit.io)
            """)
st.session_state['data'] = df_ori

# read the configs
if 'configs' not in st.session_state:
    print('load saved confings')
    st.session_state['configs'] = read_config('configs.json')
if 'ori_config' not in st.session_state:
    st.session_state['ori_config'] = {}
