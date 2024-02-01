import streamlit as st
import pandas as pd
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
from sklearn.impute import SimpleImputer


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
    'KMeans': {
        'model': KMeans,
        'params': dict(
            n_clusters=3,
            init=dict(
                options=['k-means++', 'random'],
                default='random'
            ),
            max_iter=300,
            random_state=42,
            # copy_x=True,
            # algorithm=dict(
            #     options=["lloyd", "elkan", "auto", "full"],
            #     default='auto'
            # ),
        )
    },
    'MiniBatchKMeans': {
        'model': MiniBatchKMeans,
        'params': dict(
            n_clusters=8,
            init=dict(
                options=['k-means++', 'random'],
                default='k-means++'
            ),
            n_init='auto',
            random_state=42,
            # max_iter=100,
            # batch_size=1024,
            # verbose=0,
            # compute_labels=True,
            # tol=0.0,
            # max_no_improvement=10,
            # init_size=3 * 8,
            # reassignment_ratio=0.01
        )
    },
    'DBSCAN': {
        'model': DBSCAN,
        'params': dict(
            eps=0.5,
            min_samples=5,
            metric="euclidean",
            leaf_size=30,
            # p=2,
            # n_jobs=1,
            # algorithm=dict(
            #     options=['auto', 'ball_tree', 'kd_tree', 'brute'],
            #     default='auto'
            # ),
        )
    }
}
#
name_map = {
    'KMeans': {
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
tbls = st.tabs(['configs', 'main'])
# parameters
with tbls[0]:
    st.markdown("""
    ### Main parameters:
    - n_cluster: K of k-means, number of the clusters
    - init: methods of initialize the centroid
    - max_inter: max of iteration time
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
                                            index=init_index)
                value = {**values, **{'default': value}}

            idx = (idx + 1) % 2
            new_configs[name] = value

    if configs != new_configs:
        configs_all.setdefault(TASK, {})[model_name] = new_configs
        write_config('configs.json', configs_all)

    # st.write(st.session_state.ori_config)

    # documents
    with st.expander(f'Document of {model_name}', expanded=False):
        st.markdown(model_map[model_name]['model'].__doc__, unsafe_allow_html=True)

with tbls[1]:
    st.subheader('Raw data')
    st.dataframe(df_ori, use_container_width=True, height=300)

    df.fillna(0, inplace=True)
    # Assuming your DataFrame is named df
    key_figure_column = "Key Figure"
    # Get the index of the "Key Figure" column
    key_figure_index = df.columns.get_loc(key_figure_column)
    # Select columns before and after the "Key Figure" column
    columns_before = df.iloc[:, :key_figure_index].columns
    columns_after = df.iloc[:, key_figure_index + 1:].columns
    # Create new DataFrames with selected columns
    df_before_key_figure = df[columns_before]
    df_after_key_figure = df[columns_after]

    try:
        # Create a multi-select box to choose categorical variables
        cat = st.multiselect(
            "Choose Planning Level",
            df_before_key_figure.select_dtypes(
                include=['category', 'object', 'int64', 'float64']).columns
        )

        # Convert selected categorical variables to 'category' type
        for col in cat:
            df[col] = df[col].astype('category')

        central_variable = st.selectbox("Choose Start Period", df_after_key_figure.columns,
                                        key='central_variable')
        constant_value = st.number_input("Number of the periods", value=5, step=1,
                                         key='constant_value')
        variable_index = df.columns.get_loc(central_variable)
        num = df.columns[variable_index: variable_index + constant_value]
        for col in num:
            if df[col].dtype != 'object':
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
            else:
                df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')

        dfg = df.groupby(cat)[num].sum().reset_index()
        df = dfg


        def display_document(document_list):
            st.write("### Methods of Pre-processing")

            for item in document_list:
                st.write(f"- {item}")


        document_list = [
            "StandardScaler: Data Standardization",
            "remove outlier: replace by constant",
            "dropna: Drop all NA filas",
            "LabelEncoder: Categorical Variable"]
        display_document(document_list)


        def select_columns(df):
            return df[df.select_dtypes(include='number').columns]


        def drop_na(df):
            return df.dropna(axis=0, how='all')


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
                'Drop NA Fila', 'Remove Outlier', 'StandardScaler', 'Select Numercial Columns',
                'LabelEncoder', 'MinMaxScaler')
        )
        if not select_process:
            pass
        if 'Select Numercial Columns' in select_process:
            df = select_columns(df)

        if 'Drop NA Fila' in select_process:
            df = drop_na(df)

        if 'Remove Outlier' in select_process:
            df = remove_outliers(df)

        for scaler in ['StandardScaler', 'MinMaxScaler']:
            if scaler in select_process:
                df = scale_data(df, scaler)

        cv_per_row = df[num].std(axis=1) / df[num].mean(axis=1)
        cv_per_mean = df[num].mean(axis=1)
        # Add CV as a new column in the processed_df
        df['CV'] = cv_per_row
        df['Mean'] = cv_per_mean
        df['CV-squared'] = cv_per_row ** 2

        st.subheader('After-process data')
        st.dataframe(df, use_container_width=True, height=300)
        clean_data = df

        # choose train data
        select_columns = st.multiselect('Choose train data',
                                        options=['CV', 'CV-squared', 'Mean'],
                                        default=['CV'])
        df_level = df[cat]
        df = df[select_columns]
        df.fillna(0, inplace=True)

        # Run
        args = {k: v if not isinstance(v, dict) else v['default'] for k, v in configs.items()}
        model = model_map[model_name]['model'](**args)
        output = model.fit_predict(df)
        df[TASK] = output

        df = pd.concat([df_level, df], axis=1)

        # Calculate min, max, mean of 'CV' for each cluster
        cluster_stats = df.groupby('cluster')['CV'].agg(['min', 'max', 'mean']).reset_index()
        # Sort clusters based on mean values
        sorted_clusters = cluster_stats.sort_values(by='mean', ascending=False)['cluster'].tolist()
        # Create a mapping from sorted clusters to categories 'x', 'y', 'z'
        cluster_mapping = {cluster: category for cluster, category in
                           zip(sorted_clusters, ['z', 'y', 'x'])}
        # Add a new column 'category' based on the mapping
        df['category'] = df['cluster'].map(cluster_mapping)
        st.dataframe(df, use_container_width=True, height=300)

        grouped_df = df.groupby('cluster')['CV'].agg(['min', 'max', 'mean']).reset_index()
        grouped_df['count'] = df.groupby('cluster')['cluster'].count().values
        grouped_df['category'] = grouped_df['cluster'].map(
            cluster_mapping)  # Add 'category' column based on the mapping
        grouped_df.columns = ['Cluster', 'Min', 'Max', 'Mean', 'Count',
                              'Category']  # Optional: Rename columns for clarity
        st.subheader('Cluster Results')
        st.table(grouped_df)

        # Scatter plot with Altair
        cols = st.columns(2)
        x_axis = cols[0].selectbox('Select X-axis variable:', df.columns,
                                   index=df.columns.get_loc('cluster'))
        y_axis = cols[1].selectbox('Select Y-axis variable:', df.columns,
                                   index=df.columns.get_loc('CV'))

        scatter_plot = alt.Chart(df).mark_circle().encode(
            x=x_axis,
            y=y_axis,
            color=alt.Color('category:N', title='Category', sort=['x', 'y', 'z']),
            # Specify sort order
            tooltip=list(df.columns)
        ).interactive()

        # Adding centroid points
        centroid_plot = alt.Chart(df.groupby('cluster').mean().reset_index()).mark_point(
            shape='cross',
            size=150,
            strokeWidth=2
        ).encode(
            x=x_axis,
            y=y_axis,
            color=alt.value('green'),
            tooltip=['cluster:N']
        )
        # Combine scatter plot and centroid plot
        final_plot = scatter_plot + centroid_plot
        st.subheader('Cluster Results Plot')
        st.altair_chart(final_plot, use_container_width=True)

        #############################

        #############################

        st.markdown("""
            ### Evaluation
            - Silhouette Score: The best value is 1 and the worst value is -1. Values close to 0 indicate overlapping clusters.
            - Davies-Bouldin Index: the smaller the better.
            - Calinski-Harabasz Index: the larger the better.
    
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
    except Exception as e:
        # st.error(e)

        pass
#######################
# download
try:
    st.title("Download Result")
    clean_data['Cluster'] = df['cluster']
except KeyError:
    print('keyerror')
except Exception as e:
    st.warning(f'{e}')

if st.button('Download CSV'):
    # Create a multi-sheet CSV file
    csv_file = pd.ExcelWriter(
        f"Cluster_Result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        engine='xlsxwriter')

    # Write 'clean_data' to the first sheet
    clean_data.to_excel(csv_file, sheet_name='Clustering Results', index=False)

    # Write 'results_df' to the second sheet
    results_df.to_excel(csv_file, sheet_name='Score Table', index=False)

    # Save the Excel file
    csv_file.save()

    # Prepare the file for download
    with open(f"Cluster_Result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
              "rb") as file:
        b64 = base64.b64encode(file.read()).decode()
        filename = f"Cluster_Result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    st.markdown(
        f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Data</a>',
        unsafe_allow_html=True)
