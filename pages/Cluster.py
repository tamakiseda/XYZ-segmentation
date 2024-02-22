import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
import altair as alt
import base64
from utils import read_config, write_config
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from copy import deepcopy
import warnings
import datetime
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px




TASK = 'cluster'


def recover():
    st.session_state.configs.setdefault(TASK, {})[model_name] = {}
    st.session_state.ori_config = deepcopy(model_map[model_name]['params'])
    write_config('configs.json', st.session_state.configs)


def change_model():
    st.session_state.ori_config[TASK][model_name] = configs_all[TASK][model_name]


# check data
if 'data' not in st.session_state:
    st.warning('Please upload data in Home page')
    st.stop()

# 
if 'regression' in st.session_state.ori_config:
    del st.session_state.ori_config['regression']
if 'classification' in st.session_state.ori_config:
    del st.session_state.ori_config['classification']

# Main content
st.title("Build the Cluster Model")

df_ori = st.session_state.data
df = df_ori.copy()

model_map = {
    'K-Means': {
        'model': KMeans,
        'params': dict(
            n_clusters=dict(
                options=list(range(2,3+1)),
                default=3
            ),
            init=dict(
                options=['k-means++', 'random'],
                default='k-means++'
            ),
            max_iter=300
            #random_state=42,
            # copy_x=True,
            # algorithm=dict(
            #     options=["lloyd", "elkan", "auto", "full"],
            #     default='auto'
            # ),
        )
    }

}

#init_options mapping 
init_options_mapping = {
    'random': 'K-Means',
    'k-means++':'K-Means++'
    }

#mapping of the parameters
name_map = {
    'K-Means': {
        'n_clusters': 'Number of K',
        'init': 'Initialization',
        'max_iter': 'Max Iteration'
    },
    'MiniBatchKMeans': {
        'n_clusters': 'Number of K'
    }
}
# choose model
model_name = st.selectbox(
    'Choose Model', options=tuple(model_map), on_change=change_model
)

configs_all = st.session_state['configs']

# set parameters
params = deepcopy(model_map[model_name]['params'])
configs = configs_all.get(TASK, {}).get(model_name, {})
params.update(configs)
if model_name not in st.session_state.ori_config.get(TASK, {}):
    print('save ori cluster confings')
    st.session_state.ori_config.setdefault(TASK, {})[model_name] = params
ori_config = st.session_state.ori_config[TASK][model_name]


# main page
tbls = st.tabs(['Configs', 'Main'])
# parameters
with tbls[0]:
    st.markdown("""
    ### Main parameters:
    - Number of K: K of K-Means algorithm, number of the clusters
    - Initialization: Methods of initialize the centroid
    - Max Iteration: Max of iteration time
    - Random State: Determines random number generation for centroid initialization
    """)

    with st.expander('Parameters', expanded=True):
        cols = st.columns(2)
        idx = 0
        new_configs = {}
        for name, values in params.items():
            if values is None or isinstance(values, str):
                new_name = name_map.get(model_name, {}).get(name, name)
                value = cols[idx].text_input(new_name, value=ori_config[name])
            elif isinstance(values, bool):
                new_name = name_map.get(model_name, {}).get(name, name)
                value = cols[idx].selectbox(new_name, options=[True, False], index=ori_config[name])
            elif isinstance(values, (int, float)):
                new_name = name_map.get(model_name, {}).get(name, name)
                value = cols[idx].number_input(new_name, value=ori_config[name])
            elif isinstance(values, (tuple, list)):
                new_name = name_map.get(model_name, {}).get(name, name)
                value = cols[idx].selectbox(new_name, options=ori_config[name])
            elif isinstance(values, dict):
                init_index = ori_config[name]['options'].index(
                    ori_config[name]['default'])
                new_name = name_map.get(model_name, {}).get(name, name)
                value = cols[idx].selectbox(new_name, options=ori_config[name]['options'],
                                            index=init_index, format_func=lambda x: init_options_mapping.get(x,x))
                value = {**values, **{'default': value}}

            idx = (idx + 1) % 2
            new_configs[name] = value

    if configs != new_configs:
        configs_all.setdefault(TASK, {})[model_name] = new_configs
        write_config('configs.json', configs_all)

    # st.write(st.session_state.ori_config)

    # documents
    #with st.expander(f'Document of {model_name}', expanded=False):
        #st.markdown(model_map[model_name]['model'].__doc__, unsafe_allow_html=True)

with tbls[1]:
    #Display the raw data
    st.subheader('Raw data')
    st.dataframe(df_ori, use_container_width=True, height=300)

    df.fillna(0, inplace=True)
    ################delete#############

    ##################################################delete#############
    key_figure_column = "Key Figure"
    key_figure_index = df.columns.get_loc(key_figure_column)
    # Select columns before and after the "Key Figure" column
    columns_before = df.iloc[:, :key_figure_index].columns
    columns_after = df.iloc[:, key_figure_index + 1:].columns
    df_before_key_figure = df[columns_before]
    df_after_key_figure = df[columns_after]

    ################delete#############
    # st.subheader('df_before_key_figure')
    # st.dataframe(df_before_key_figure, use_container_width=True, height=300)

    # st.subheader('df_after_key_figure')
    # st.dataframe(df_after_key_figure, use_container_width=True, height=300)
    ##################################################delete#############
    for col in columns_before:
        df[col] = df[col].astype('object')
    for col in columns_after:
        if df[col].dtype != 'object':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        else:
            df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')

    try:
        cat = st.multiselect(
            "Choose Planning Level",
            df_before_key_figure.select_dtypes(
                include=['category', 'object', 'int64', 'float64']).columns
        )
        central_variable = st.selectbox("Choose Start Period", df_after_key_figure.columns,
                                        key='central_variable')
        constant_value = st.number_input("Number of the forward periods", value=5, step=1,
                                            key='constant_value')
        variable_index = df.columns.get_loc(central_variable)
        num = df.columns[variable_index: variable_index + constant_value]
        #dfg = df.groupby(df_before_key_figure)[num].sum().reset_index()
        dfg = pd.pivot_table(df,values =num,
               index=cat, 
               aggfunc='sum').reset_index()
        st.subheader('Selected Data')
        st.dataframe(dfg, use_container_width=True, height=150)
        ################


        def display_document(document_list):
            st.write("### Methods of Pre-processing")

            for item in document_list:
                st.write(f"- {item}")




        document_list = [
            "Remove Outlier: Replace by constant",
            "Drop NA Fila: Drop all NA filas"
            ]
        display_document(document_list)


        df=dfg
        df.fillna(0, inplace=True)

        def select_columns(df):
            return df[df.select_dtypes(include='number').columns]
        
        def drop_na(df):
            mask = df.select_dtypes(include=[float, int]).fillna(0).eq(0).all(axis=1)
            return df[~mask]

        def remove_outliers(df):
            df_tmp = df[df.select_dtypes(include='number').columns]

            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler())
            ])

            pipeline = Pipeline(steps=[
                ('preprocessor', numerical_transformer),
                ('outlier_detector', IsolationForest(contamination=0.05))
            ])
            df_tmp['outlier'] = pipeline.fit_predict(df_tmp)

            return df[df_tmp['outlier'] != 1]
        
        def scale_data(df, scaler):
            numeric_columns = df.select_dtypes(include='number').columns

            if scaler == 'StandardScaler':
                numeric_df = df[numeric_columns]
                scaled_df = pd.DataFrame(StandardScaler().fit_transform(numeric_df),
                                            columns=numeric_df.columns)
                df[numeric_columns] = scaled_df
            elif scaler == 'MinMaxScaler':
                df[numeric_columns] = pd.DataFrame(
                    MinMaxScaler().fit_transform(df[numeric_columns]),
                    columns=numeric_columns)
            return df


        # Preprocessing code
        select_process = st.multiselect(
            'Pre-processing',
            options=(
                'Drop NA Fila', 'Remove Outlier')
        )
        if not select_process:           
            pass

        #if 'Select Numercial Columns' in select_process:
            #df = select_columns(df)

        if 'Drop NA Fila' in select_process:
            df = drop_na(df)

        if 'Remove Outlier' in select_process:
            df = remove_outliers(df)

        #for scaler in ['StandardScaler', 'MinMaxScaler']:
            #if scaler in select_process:
                #df = scale_data(df, scaler)

        cv_per_row = df[num].std(axis=1) / df[num].mean(axis=1)
        cv_per_mean = df[num].mean(axis=1)
        # Add CV as a new column in the processed_df
        df['CV'] = cv_per_row
        df['Mean'] = cv_per_mean
        df['CV-squared'] = cv_per_row ** 2
        df.fillna(0, inplace=True)


        st.subheader('After-process data')
        st.dataframe(df, use_container_width=True, height=200)


        #################################################################
        # choose train data
        clean_data=df
        clean_data.fillna(0, inplace=True) 
        select_columns = st.multiselect('Choose train data',
                                        options=['CV'],
                                        default=['CV'])
        df_level = df[cat]
        df = df[select_columns]
        df.fillna(0, inplace=True)
        
        # Run
        args = {k: v if not isinstance(v, dict) else v['default'] for k, v in configs.items()}
        model = model_map[model_name]['model'](**args)
        #############

        print("here is the model")
        print(model.get_params())
    
        #############
        output = model.fit_predict(df)
        df[TASK] = output
        df = pd.concat([df_level, df], axis=1)


        cluster_stats = df.groupby('cluster')['CV'].agg(['min', 'max', 'mean']).reset_index()
        # Sort clusters based on mean values
        sorted_clusters = cluster_stats.sort_values(by='mean', ascending=False)['cluster'].tolist()
        # Create a mapping from sorted clusters to categories 'x', 'y', 'z'
        cluster_mapping = {cluster: category for cluster, category in
                            zip(sorted_clusters, ['z', 'y', 'x'])}
        df['category'] = df['cluster'].map(cluster_mapping)
        #attention, here is the df  !!!!
        st.subheader('Results Table')
        st.dataframe(df, use_container_width=True, height=200)


        grouped_df = df.groupby('cluster')['CV'].agg(['min', 'max', 'mean']).reset_index()
        grouped_df['count'] = df.groupby('cluster')['cluster'].count().values
        grouped_df['category'] = grouped_df['cluster'].map(
            cluster_mapping)  
        grouped_df.columns = ['Cluster', 'Min', 'Max', 'Mean', 'Count',
                                'Category']  
        st.subheader('Cluster Results')
        grouped_df=grouped_df.drop(['Cluster'],axis=1)
        st.table(grouped_df)


        import plotly.express as px
        import plotly.graph_objects as go
        cols = st.columns(2)
        
        allowed_columns = ['category', 'CV']
        #allowed_columns=allowed_columns.extend(cat)

        x_axis = cols[0].selectbox('Select X-axis variable:',[col for col in df.columns if col in allowed_columns],
                                    index=allowed_columns.index('CV'))

        y_axis = cols[1].selectbox('Select Y-axis variable:',[col for col in df.columns if col in allowed_columns],
                                    index=allowed_columns.index('category'))
        
        
        # Plot the scatter plot
        if pd.api.types.is_numeric_dtype(df[x_axis]):
            fig = px.scatter(df, x=x_axis, y=y_axis, labels={x_axis: 'X-axis', y_axis: 'Y-axis'})
        else:
            fig = px.scatter(df, x=x_axis, y=y_axis,color=x_axis, labels={x_axis: 'X-axis', y_axis: 'Y-axis'}, category_orders={x_axis: sorted(df[x_axis].unique())})
        

        centroid_df = df.groupby('category').mean().reset_index()


        fig.add_trace(go.Scatter(
            x=centroid_df[x_axis],
            y=centroid_df[y_axis],
            mode='markers',
            marker=dict(symbol='cross',size=10, color='green'),
            name='Centroids'
        ))

        fig.update_layout(legend=dict(title=x_axis))
        st.plotly_chart(fig, use_container_width=True)


        #############################
        # Create a pie plot
        pie_fig = px.pie(values=df['category'].value_counts(), names=df['category'].value_counts().index)
        st.plotly_chart(pie_fig, use_container_width=True)







        #############################

        st.markdown("""
            ### Evaluation
            - Silhouette Score: The best value is 1 and the worst value is -1. Values close to 0 indicate overlapping clusters better than 0.8
            - Davies-Bouldin Index: The smaller the better, value is between (0,1), 0="good" clusters and 1="bad" clusters
            - Calinski-Harabasz Index: The larger the better.

                """)

        # Evaluate the clustering results
        dato = df[select_columns]
        silhouette_avg = silhouette_score(dato, df['cluster'])
        db_index = davies_bouldin_score(dato, df['cluster'])
        ch_index = calinski_harabasz_score(dato, df['cluster'])
        data = {'Metric': ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index'],
                'Score': [silhouette_avg, db_index, ch_index]}
        results_df = pd.DataFrame(data)
        st.title("Score Table")
        st.table(results_df)


        #########
        try:
            st.title("Download Result")
            clean_data=pd.DataFrame(clean_data)
            clean_data['Category']=df['category']
        except KeyError:
            print('keyerror')
        except Exception as e:
            st.warning(f'{e}')
        

        st.dataframe(clean_data, use_container_width=True, height=300)
        clean_data = pd.DataFrame(clean_data)
        
        if st.button('Download CSV'):
            date_time=f"Cluster_Result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            csv_file = pd.ExcelWriter(date_time,engine='xlsxwriter')
            clean_data.to_excel(csv_file, sheet_name='Clustering Results', index=False)
            results_df.to_excel(csv_file, sheet_name='Score Table', index=False)
            csv_file.save()
            with open(date_time,"rb") as file:
                b64 = base64.b64encode(file.read()).decode()
                filename = date_time
            st.markdown(
        f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Data</a>',
        unsafe_allow_html=True)

    except Exception as e:
        st.error(e)
        pass



        


