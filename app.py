import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from numpy.linalg import matrix_power


st.title("Predictions of Water Solubility and Vapor Pressure")
st.subheader("Coś tam coś tam")
st.header("Bla bla bla")
st.text("blablabla")
st.markdown("[Google](https://www.google.com)")

# Panel boczny z zakładkami
selected_tab = st.sidebar.selectbox("Select a enpoint:", ["Vapor Pressure", "Water Solubility"])




if selected_tab == "Vapor Pressure":


    x_df = pd.read_excel('VP_matrix.xlsx', sheet_name= 'Arkusz1', index_col=0)
    y_df = pd.read_excel('VP_matrix.xlsx', sheet_name='Arkusz2', index_col=0)
    var_names = x_df.columns.values

    def one_to_x(X, n=3):
        row_val = X.shape[0]
        train_split = []
        val_split = []  
        for row in range(0, row_val):
            if ((row%n) < n-1):
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

    X_copy = x_df.copy()
    joined_mat = X_copy.join(y_df[y_df.columns.values[0]]).sort_values(by=y_df.columns.values[0], ascending=True)
    X_copy = joined_mat.iloc[:,:-1]
    y_copy = joined_mat.iloc[:,-1]


    variables =X_copy = x_df.copy()
    joined_mat = X_copy.join(y_df[y_df.columns.values[0]]).sort_values(by=y_df.columns.values[0], ascending=True)
    X_copy = joined_mat.iloc[:,:-1]
    y_copy = joined_mat.iloc[:,-1]


    variables = ['F03CF', 'nDB', 'AAC']
    ts, vs = one_to_x(x_df,3)

    Xt = X_copy.iloc[ts]
    Xv = X_copy.iloc[vs]
    yt = y_copy.iloc[ts]
    yv = y_copy.iloc[vs]

    X_train = Xt[variables]
    X_test = Xv[variables]
    y_train = yt.copy()
    y_test = yv.copy()
    ts, vs = one_to_x(x_df,3)

    Xt = X_copy.iloc[ts]
    Xv = X_copy.iloc[vs]
    yt = y_copy.iloc[ts]
    yv = y_copy.iloc[vs]

    X_train = Xt[variables]
    X_test = Xv[variables]
    y_train = yt.copy()
    y_test = yv.copy()

    from sklearn.preprocessing import StandardScaler
    scaling = StandardScaler()

    X_cols= X_train.columns
    X_train_idx = X_train.index
    X_test_idx = X_test.index

    X_train_z = scaling.fit_transform(X_train)
    X_train = pd.DataFrame(X_train_z, columns=X_cols, index=X_train_idx)

    X_test_z = scaling.transform(X_test)
    X_test = pd.DataFrame(X_test_z, columns=X_cols, index=X_test_idx)

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train,y_train)

    est = LinearRegression()
    est.fit(X_train,y_train)


    y_train = pd.DataFrame(y_train)

    Yt_pred_z = pd.DataFrame(model.predict(X_train), columns=[y_train.columns[0]], index=y_train.index.values)
    Yv_pred_z = pd.DataFrame(model.predict(X_test), columns=[y_train.columns[0]], index=y_test.index.values)
    Yt_pred = pd.DataFrame(Yt_pred_z, columns=[y_train.columns[0]], index=y_train.index.values)
    Yv_pred = pd.DataFrame(Yv_pred_z, columns=[y_train.columns[0]], index=y_test.index.values)

    train_obs = np.asarray(y_train).reshape(-1)
    train_pred = np.asarray(Yt_pred).reshape(-1)
    test_obs = np.asarray(y_test).reshape(-1)
    test_pred = np.asarray(Yv_pred).reshape(-1)


    def insubria_plot(df_x_train, df_x_test, y_train_pred, y_test_pred, df_user_x, user_pred, y_obs, cut_lev=None) -> None:

        h_k = (3 * (int(len(df_x_train.columns))+1))/len(df_x_train.index)
        h_train_idx = []
        h_val_idx = []
        h_us_idx= []


        max_y_pred = np.max([np.max(y_train_pred), np.max(y_test_pred)])
        min_y_pred = np.min([np.min(y_train_pred), np.min(y_test_pred)])

        mat_dot = matrix_power((np.dot(df_x_train.T,df_x_train)), -1)

        for idx in df_x_train.index.values:
            xi_t = np.asarray(df_x_train.loc[idx]).T
            xi = np.asarray(df_x_train.loc[idx])
            hi = np.dot(xi_t.dot(mat_dot), xi)
            h_train_idx.append(hi)


        for idx in df_x_test.index.values:
            xi_t = np.asarray(df_x_test.loc[idx]).T
            xi = np.asarray(df_x_test.loc[idx])
            hi_v = np.dot(xi_t.dot(mat_dot), xi)
            h_val_idx.append(hi_v)


        for idx in df_user_x.index.values:
            xi_t = np.asarray(df_user_x.loc[idx]).T
            xi = np.asarray(df_user_x.loc[idx])
            hi_v = np.dot(xi_t.dot(mat_dot), xi)
            h_us_idx.append(hi_v)

        if cut_lev:
            # h_train_idx = h_train_idx[h_train_idx <= cut_lev]
            # y_train_pred = y_train_pred[h_train_idx <= cut_lev]
        
            # h_val_idx = h_val_idx[h_val_idx <= cut_lev]
            # y_test_pred = y_test_pred[h_val_idx <= cut_lev]
            # print(h_us_idx)
            # print(user_pred)
            h_us_idx = np.asarray(h_us_idx)
            user_pred = user_pred[h_us_idx <= cut_lev]
            df_user_x = df_user_x[h_us_idx <= cut_lev]
            h_us_idx = h_us_idx[h_us_idx <= cut_lev]
        
    # Define a function to scale user inputs
    def scale_user_inputs(f03cf, ndb, aac):
        user_inputs = pd.DataFrame({'F03CF': [f03cf], 'nDB': [ndb], 'AAC': [aac]})
        scaled_inputs = scaling.transform(user_inputs)
        return scaled_inputs

    # Define a function to predict solubility pressure
    def predict_solubility_pressure(scaled_inputs):
        prediction = model.predict(scaled_inputs)
        return prediction[0]


    # Calculate mat_dot based on your training data
    mat_dot = matrix_power((np.dot(X_train.T, X_train)), -1)

    h_k = (3 * (int(len(X_train.columns)) + 1)) / len(X_train.index)




    F03CF = st.number_input("Enter F03[C-F] value", value=0)  # Ustawić domyślną wartość na przykład 0.0
    nDB = st.number_input("Enter nDB value", value=0)
    AAC = st.number_input("Enter AAC value", value=0)


    # Calculate h_us_idx
    scaled_inputs = scale_user_inputs(F03CF, nDB, AAC)
    xi = np.asarray(scaled_inputs)
    xi_t = xi.T  # Transpose xi to make it (1, 3)
    h_us_idx = np.dot(xi, np.dot(mat_dot, xi_t))[0, 0]

    # Check if the user has provided input values
    if st.button("Predict"):
        prediction = predict_solubility_pressure(scaled_inputs)
    
     # Check if h_us_idx is greater than h_k
        if h_us_idx > h_k:
            st.markdown(f'<p style="background-color: red; padding: 10px; color: white;">Predicted Value: {prediction:.2f} is outside the applicability domain.</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="background-color: green; padding: 10px; color: white;">Predicted Value: {prediction:.2f} is inside applicability domain</p>', unsafe_allow_html=True)