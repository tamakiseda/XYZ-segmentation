import os
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from copy import deepcopy
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
import base64
from utils import read_config, write_config
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

TASK = 'regression'


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
df_ori = st.session_state.data
df = df_ori.copy()

# 
if 'cluster' in st.session_state.ori_config:
    del st.session_state.ori_config['cluster']
if 'classification' in st.session_state.ori_config:
    del st.session_state.ori_config['classification']


# Main content
st.title("Build the Classification Model")

to_dict = lambda d: dict(options=d, default=d[0])

model_map = {
    'Lasso': {
        'model': Lasso,
        'params': dict(
            alpha=1.0,
            fit_intercept=True,
            precompute=False,
            copy_X=True,
            max_iter=1000,
            tol=1e-4,
            warm_start=False,
            positive=False,
            random_state=None,
            selection=to_dict(['cyclic', 'random']),
        )
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor,
        'params': dict(
            n_estimators=100,
            criterion=to_dict(["squared_error", "absolute_error", "friedman_mse", "poisson"]),
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
        )
    },
    'DecisionTreeRegressor': {
        'model': DecisionTreeRegressor,
        'params': dict(
            criterion=to_dict(["squared_error", "friedman_mse", "absolute_error", "poisson"]),
            splitter=to_dict(["best", "random"]),
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            random_state=42,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            ccp_alpha=0.0,
        )
    },
    'KNeighborsRegressor': {
        'model': KNeighborsRegressor,
        'params': dict(
            n_neighbors=5,
            weights=to_dict(['uniform', 'distance']),
            algorithm=to_dict(['auto', 'ball_tree', 'kd_tree', 'brute']),
            leaf_size=30,
            p=2,
            metric="minkowski",
            metric_params=None,
            n_jobs=None,
        )
    },
    'SVR': {
        'model': SVR,
        'params': dict(
            kernel=to_dict(['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']),
            degree=3,
            gamma=to_dict(['scale', 'auto']),
            coef0=0.0,
            tol=1e-3,
            C=1.0,
            epsilon=0.1,
            shrinking=True,
            cache_size=200,
            verbose=False,
            max_iter=-1,
        )
    },
}

# 模型选择
model_name = st.selectbox('Choose Model', options=tuple(model_map), on_change=change_model)
# 参数设置 动态生成
configs_all = st.session_state['configs']
# 参数设置 动态生成
params = deepcopy(model_map[model_name]['params'])
configs = configs_all.get(TASK, {}).get(model_name, {})
params.update(configs)
if model_name not in st.session_state.ori_config.get(TASK, {}):
    print('save ori regression confings')
    st.session_state.ori_config.setdefault(TASK, {})[model_name] = params
ori_config = st.session_state.ori_config[TASK][model_name]

# 两个页面
tbls = st.tabs(['configs', 'main'])
# 配置参数
with tbls[0]:
    with st.expander('Parameters', expanded=True):
        cols = st.columns(2)
        idx, new_configs = 0, {}
        for name, values in params.items():
            if values is None or isinstance(values, str):
                value = cols[idx].text_input(name, value=ori_config[name])
            elif isinstance(values, bool):
                value = cols[idx].selectbox(name, options=[True, False], index=ori_config[name])
            elif isinstance(values, (int, float)):
                value = cols[idx].number_input(name, value=ori_config[name])
            elif isinstance(values, (tuple, list)):
                value = cols[idx].selectbox(name, options=ori_config[name])
            elif isinstance(values, dict):
                init_index = ori_config[name]['options'].index(
                    ori_config[name]['default'])
                value = cols[idx].selectbox(name, options=ori_config[name]['options'],
                                            index=init_index)
                value = {**values, **{'default': value}}  # 序列解包

            idx = (idx + 1) % 2
            new_configs[name] = value

        # 恢复默认值
        # if st.button('恢复默认值', on_click=recover):
        #     st.warning('刷新页面更新🔄')

    # 保存参数
    if configs != new_configs:
        configs_all.setdefault(TASK, {})[model_name] = new_configs
        write_config('configs.json', configs_all)  # 及时保存    

    st.write(st.session_state.ori_config)
    
    # 算法的说明文档
    with st.expander(f'Document of {model_name}', expanded=False):
        st.markdown(model_map[model_name]['model'].__doc__, unsafe_allow_html=True)

with tbls[1]:

    df.fillna(0, inplace=True)
    for col in df.columns:
        df[col] = df[col].replace(',', '', regex=True)

    # Create a multi-select box to choose categorical variables
    selected_categorical_variables = st.multiselect(
        "Choose categorical variables",
        df.select_dtypes(include=['category', 'object', 'int64', 'float64']).columns
    )
    if not selected_categorical_variables:
        st.warning('Please select categorical variables')
        st.stop()
    # Convert selected categorical variables to 'category' type
    for var in selected_categorical_variables:
        df[var] = df[var].astype('category')

    # Create a multi-select box to choose categorical variables
    selected_numerical_variables = st.multiselect(
        "Choose numerical variables",
        df.select_dtypes(include=['category', 'object', 'int64', 'float64']).columns, key='ddd'
    )
    if not selected_numerical_variables:
        st.warning('Please select numerical variables')
        st.stop()
    # Convert selected categorical variables to 'category' type
    for var in selected_numerical_variables:
        df[var] = df[var].astype('float')

    #
    df = df.groupby(selected_categorical_variables)[
        selected_numerical_variables].sum().reset_index()
    df['cv'] = df.std(axis=1) / df.mean(axis=1)
    df['mean'] = df.mean(axis=1)

    # 数据预处理Preprocessing code
    select_process = st.multiselect(
        'Pre-processing',
        options=(
            'dropna', 'remove outlier', 'StandardScaler', 'select columns(number)',
            'MinMaxScaler')
    )
    if 'select columns(number)' in select_process:
        df = df[df.select_dtypes(include='number').columns]
    if 'dropna' in select_process:
        # df.dropna(axis=1, inplace=True)
        df = df[df.apply(lambda x: x.sum(), axis=1) != 0]
    if 'LabelEncoder' in select_process:
        pass
    if 'remove outlier' in select_process:
        df_tmp = df[df.select_dtypes(include='number').columns]

        # Create a transformer for numerical columns
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ])
        # Create a pipeline with preprocessor and outlier detection
        pipeline = Pipeline(steps=[
            ('preprocessor', numerical_transformer),
            ('outlier_detector', IsolationForest(contamination=0.05))
        ])
        # Apply the pipeline to your DataFrame
        df_tmp['outlier'] = pipeline.fit_predict(df_tmp)
        # Filter out the outliers
        df = df[df_tmp['outlier'] == 1]
    # df_index = df.index  # 记录
    if 'StandardScaler' in select_process:
        df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
    elif 'MinMaxScaler' in select_process:
        df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
    # label encoder
    cols = st.columns([0.2, 0.8])
    cols[0].markdown('')
    label_encode = cols[0].checkbox('LabelEncoder')
    if label_encode:
        df_tmp = df_ori.iloc[df.index, :].select_dtypes('object')
        if not df_tmp.empty:
            label_colums = cols[1].multiselect('Choose label',
                                               options=tuple(df_tmp.columns),
                                               default=tuple(df_tmp.columns))
            if label_colums:
                for column in label_colums:
                    df[column] = LabelEncoder().fit_transform(df_tmp[column])

    # 选择训练数据
    select_columns = st.multiselect('Choose train data', options=tuple(df.columns),
                                    default=tuple(df.select_dtypes('number').columns))
    df = df[select_columns]

    # 划分训练集和测试集
    cols = st.columns(2)
    train_rate = cols[0].number_input('train_rate', min_value=0.1, max_value=0.9, value=0.8)
    df_train, df_test = train_test_split(df, train_size=train_rate)

    # target y
    target_column = cols[1].selectbox('target y', options=tuple(df.columns))
    df_train_y = df_train.pop(target_column)
    df_train_x = df_train
    df_test_y = df_test.pop(target_column)
    df_test_x = df_test

    # show data
    st.subheader('Raw data')
    st.dataframe(df_ori, use_container_width=True, height=300)

    st.subheader('Preprocess data')
    st.dataframe(df, use_container_width=True, height=300)

    # 运行
    ## 初始化模型
    args = {k: v if not isinstance(v, dict) else v['default'] for k, v in configs.items()}
    model = model_map[model_name]['model'](**args)
    model.fit(df_train_x, df_train_y)
    predict_y = model.predict(df_test_x)
    new_target_y = f'predict {target_column}'
    df_test_x[target_column] = df_test_y
    df_test_x[new_target_y] = predict_y

    st.subheader('Classification Results')

    fig, ax = plt.subplots()
    plt.plot(df_test_x[target_column], '.', label='target')
    plt.plot(df_test_x[new_target_y], '.', label='predict')
    plt.legend()
    st.pyplot(fig)

    # 下载
    select_c = st.multiselect('Choose download variables', options=tuple(df_test_x.columns),
                              default=tuple(df_test_x.columns))
    download_button = st.button('Download test CSV')
    if download_button:
        # Create a CSV file for download
        # tmp = df_ori.loc[df_index, df_test_x.columns.values[:-1]]
        # tmp['class'] = df_test_x[new_target_y].values
        # csv_file = tmp.to_csv(index=False)
        csv_file = df_test_x[select_c].to_csv(index=False)

        # Prepare the file for download
        b64 = base64.b64encode(csv_file.encode()).decode()
        filename = f"selected_variables_data.csv"
        st.markdown(f"### Download Selected Variables Data")
        st.markdown(
            f"Click below to download selected variables data as a CSV file:")
        st.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Selected Variables Data</a>',
            unsafe_allow_html=True)
