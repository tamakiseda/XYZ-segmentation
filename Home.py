import streamlit as st
import pandas as pd
from utils import read_config

st.set_page_config(
    page_title="Home",
    page_icon="📊",
    layout='centered'
)

st.write("# Welcome to use this extra ML tool")
# st.sidebar.success("Select the step")

st.markdown("""
    ### Flexible interface with SAP IBP CI-DS
    
    We only provide upload data manually now
            
    Send a request to IBP and get data from API
            """)

# st.image("https://github.com/tamakiseda/streamlit-/blob/main/pic3.png?raw=true")
st.image("./overall_process.png")

#example data
file_path = "test_data.csv"
exampledata = pd.read_csv(file_path)
st.markdown("""
    ##### Download Example Data
            """)

# Create a button to trigger the download
if st.button('Download Example'):
    # Prepare the CSV file
    csv_file = exampledata.to_csv(index=False)
    # Create a download link
    st.download_button(label='Download CSV', data=csv_file, file_name='example.csv', mime='text/csv', key='download_button')



uploaded_file = st.file_uploader(
    'File uploader', type=['xlsx', 'xls', 'csv'],
    accept_multiple_files=False)
if uploaded_file is not None:
    st.success('File successfully uploaded!')

st.markdown(
    """
        Features of our tiny platform

    **👈 Try to explore from the sidebar**

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

# 读取之前的配置参数
if 'configs' not in st.session_state:
    print('load saved confings')
    st.session_state['configs'] = read_config('configs.json')
if 'ori_config' not in st.session_state:
    st.session_state['ori_config'] = {}
