import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy.linalg import matrix_power
from sklearn.linear_model import LinearRegression

def load_data(file_name):
    x_df = pd.read_excel(file_name, sheet_name='Arkusz1', index_col=0)
    y_df = pd.read_excel(file_name, sheet_name='Arkusz2', index_col=0)
    return x_df, y_df

def split_data(x_df, y_df, n=3):
    row_val = x_df.shape[0]
    train_split = []
    val_split = []  
    for row in range(0, row_val):
        if ((row % n) < n - 1):
            train_split.append(row)
        else:
            val_split.append(row)
    last_train = np.max(train_split)
    last_val = np.max(val_split)
    if (np.max(train_split) < np.max(val_split)):
        train_split.pop()
        val_split.pop()
        train_split.append(last_val)
        val_split.append(last_train)
    return train_split, val_split

def preprocess_data(x_df, y_df, train_split, val_split, variables):
    X_copy = x_df.copy()
    joined_mat = X_copy.join(y_df[y_df.columns.values[0]]).sort_values(by=y_df.columns.values[0], ascending=True)
    X_copy = joined_mat.iloc[:,:-1]
    y_copy = joined_mat.iloc[:,-1]
    Xt = X_copy.iloc[train_split]
    Xv = X_copy.iloc[val_split]
    yt = y_copy.iloc[train_split]
    yv = y_copy.iloc[val_split]

    scaling = StandardScaler()
    X_cols = Xt[variables].columns
    X_train_idx = Xt.index
    X_test_idx = Xv.index
    X_train_z = scaling.fit_transform(Xt[variables])
    X_train = pd.DataFrame(X_train_z, columns=X_cols, index=X_train_idx)
    X_test_z = scaling.transform(Xv[variables])
    X_test = pd.DataFrame(X_test_z, columns=X_cols, index=X_test_idx)
    
    return X_train, X_test, yt, yv, scaling

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def calculate_hk(X_train):
    return (3 * (int(len(X_train.columns)) + 1)) / len(X_train.index)

def calculate_hus(scaled_inputs, X_train):
    mat_dot = matrix_power((np.dot(X_train.T, X_train)), -1)
    xi = np.asarray(scaled_inputs)
    xi_t = xi.T
    h_us_idx = np.dot(xi, np.dot(mat_dot, xi_t))[0, 0]
    return h_us_idx

def scale_user_inputs_vp(input1, input2, input3, scaling, X_cols):
    user_inputs = pd.DataFrame({'F03CF': [input1], 'nDB': [input2], 'AAC': [input3]})  
    scaled_inputs = scaling.transform(user_inputs[X_cols])
    return scaled_inputs

def scale_user_inputs_ws(input1, input2, scaling, X_cols):
    user_inputs = pd.DataFrame({'TFF': [input1], 'SIC1': [input2]})  
    scaled_inputs = scaling.transform(user_inputs[X_cols])
    return scaled_inputs

def predict_solubility_pressure(scaled_inputs, model):
    prediction = model.predict(scaled_inputs)
    return prediction[0]



st.title("Predictions of Water Solubility and Vapor Pressure")
st.subheader("Coś tam coś tam")
st.header("Bla bla bla")
st.text("blablabla")
st.markdown("[Google](https://www.google.com)")

# Panel boczny z zakładkami
selected_tab = st.sidebar.selectbox("Select a enpoint:", ["Vapor Pressure", "Water Solubility"])

if selected_tab == "Vapor Pressure":
    x_df, y_df = load_data('VP_matrix.xlsx')
    train_split, val_split = split_data(x_df, y_df, 3)
    variables = ['F03CF', 'nDB', 'AAC']
    X_train, X_test, y_train, y_test, scaling = preprocess_data(x_df, y_df, train_split, val_split, variables)
    model = train_model(X_train, y_train)
    h_k = calculate_hk(X_train)

    F03CF = st.number_input("Enter F03[C-F] value", value=0)
    nDB = st.number_input("Enter nDB value", value=0)
    AAC = st.text_input("Enter AAC value", value=0)

    try:
        AAC = float(AAC)
    except ValueError:
        st.error("Please enter a floating point number")

    scaled_inputs = scale_user_inputs_vp(F03CF, nDB, AAC, scaling, variables)

    if st.button("Predict"):
        prediction = predict_solubility_pressure(scaled_inputs, model)
        h_us_idx = calculate_hus(scaled_inputs, X_train)
        if h_us_idx > h_k:
            st.markdown(f'<p style="background-color: red; padding: 10px; color: white;">Predicted Value: {prediction:.2f} is outside the applicability domain.</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="background-color: green; padding: 10px; color: white;">Predicted Value: {prediction:.2f} is inside applicability domain</p>', unsafe_allow_html=True)

if selected_tab == "Water Solubility":
    x_df, y_df = load_data('WS-matrix.xlsx')
    train_split, val_split = split_data(x_df, y_df, 4)
    variables = ['TFF', 'SIC1']
    X_train, X_test, y_train, y_test, scaling = preprocess_data(x_df, y_df, train_split, val_split, variables)
    model = train_model(X_train, y_train)
    h_k = calculate_hk(X_train)

    TFF = st.number_input("Enter T(F..F) value", value=0)
    SIC1 = st.text_input("Enter SIC1 value", value="0.0")

    try:
        SIC1 = float(SIC1)
    except ValueError:
        st.error("Please enter a floating point number")

    scaled_inputs = scale_user_inputs_ws(TFF, SIC1, scaling, variables)

    if st.button("Predict"):
        prediction = predict_solubility_pressure(scaled_inputs, model)
        h_us_idx = calculate_hus(scaled_inputs, X_train)
        if h_us_idx > h_k:
            st.markdown(f'<p style="background-color: red; padding: 10px; color: white;">Predicted Value: {prediction:.2f} is outside the applicability domain.</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="background-color: green; padding: 10px; color: white;">Predicted Value: {prediction:.2f} is inside applicability domain</p>', unsafe_allow_html=True)
