model_selection = 0

import streamlit as st
import os
import streamlit.components.v1 as components
import pandas as pd
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
from pycaret.classification import setup, compare_models, pull, save_model
from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
from sklearn.neighbors import KNeighborsClassifier
from io import StringIO
import numpy as np
import sweetviz
import base64
import time
import plotly.express as px
import streamlit as st
from pycaret.classification import *
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



# Using object notation
add_selectbox = st.sidebar.selectbox(
    "CHOOSE OPERATION",
    ("None","Regression", "Classification", "Clustering")
)

if os.path.exists("source.csv"):
    df = pd.read_csv("source.csv", index_col=None)

# Using "with" notation
with st.sidebar:
    st.image("images.jpeg")
    add_radio = st.radio(
        "SELECT OPERATION",
        ("Dataset Selection","Report","Visualisation","Pipeline","Preprocess","Data Split","Validate and Predict")
    )
    df = pd.read_csv("Cancer_Data.csv")

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')


    csv = convert_df(df)

    st.download_button(
        "Download",
        csv,
        "Cancer_Data.csv",
        "text/csv",
        key='download-csv'
    )




if(add_radio == "Dataset Selection"):
    #ste
    file = st.file_uploader("Choose a file")
    if file:
        df = pd.read_csv(file, index_col = None)
        df.to_csv("source.csv",index=None)
        st.success('Successfully Imported', icon="✅")
    #if os.path.exists("source.csv"):


if(add_radio == "Report"):
    st.write("# REPORT OF THE DATASET")
    if st.button('DATA FRAME'):
        st.write(df)
        st.write("The Column info")
        st.write(df.info(memory_usage='deep'))
        st.write("The Null values:",df.isnull().sum())


if(add_radio == "Data Split"):
    df = df.drop(['id', 'Unnamed: 32'], axis=1)
    target = st.selectbox("Select a column for Test:",df.columns)
    setup(df,target=target, session_id = 123)
    setup_df = pull()
    st.info("ML Info")
    st.dataframe(setup_df)
    #best = compare_models(verbose = False)
    #best1 = pull()
    #st.dataframe(best1)


if(add_radio == 'Visualisation'):
    st.title("Data Visualisation")
    df = pd.read_csv("source.csv")
    pyg_html = pyg.walk(df, return_html=True)
    components.html(pyg_html, height=800)

if(add_radio == "Preprocess"):
    st.title("Dataset proprocessing")
    target = st.selectbox("Select a column for Preprocess:",df.columns)
    df = pd.get_dummies(df, columns = [target])
    #tar = st.selectbox("Select Datatype:",['int','float','string','bool'])
    #if tar:
       # df = df.astype({
           # target : tar
       # })
    st.write(df)

def incr(n):
    global model_selection
    model_selection = n

if(add_radio == "Validate and Predict"):
    #global model_selection
    #model_selection = 0
    st.title(":::Validation of a model:::")
    tar = st.selectbox("Select a Model:",["None","SVC","Decision Tree", "KNN", "Logistic Regression","Naive Bayes"])
    values = st.slider(
    'Select the test split:',
    20.0, 40.0, (20.0))
    st.write('Values:', values)
    dset = pd.read_csv("source.csv")
    dset['diagnosis'] = dset['diagnosis'].map({
        'M':1,
        'B':0
    })
    X = dset.drop(["diagnosis",'id',"Unnamed: 32"],axis=1)
    Y = dset["diagnosis"]
    xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=(values/100),random_state=2)
    if(tar == 'SVC'):
        incr(1)
        model = svm.SVC()
    elif(tar == 'Decision Tree'):
        incr(2)
        model = DecisionTreeClassifier(random_state=0)
    elif(tar == 'KNN'):
        incr(3)
        model = KNeighborsClassifier()
    elif(tar == 'None'):
        st.write("Please select a model to get validate")

    model.fit(xtrain,ytrain)
    d = model.predict(xtest)
    acc = accuracy_score(d,ytest)
    st.write("Accuracy:",(acc*100),"%")
    if(acc == 1.0):
        file_ = open("1.gif", "rb")
    elif(acc<1.0 and acc>=0.9):
        file_ = open("2.gif", "rb")
    elif(acc<0.9 and acc>=0.6):
        file_ = open("3.gif", "rb")
    else:
        file_ = open("4.gif", "rb")

    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )
    st.title("ENTER THE PARAMETERS::👇🏻::")
    st.write("If you don't have a parameter than you can put the average value of each variable giver in the below")
    # user input
    a = st.text_input("Enter the radius_mean of Cancer cell",dset.radius_mean.mean())
    a1 = st.text_input("Enter the texture_mean of Cancer cell",dset.texture_mean.mean())
    a2 = st.text_input("Enter the perimeter_mean of Cancer cell",dset.perimeter_mean.mean())
    a3 = st.text_input("Enter the area_mean of Cancer cell",dset.area_mean.mean())
    a4 = st.text_input("Enter the smoothness_mean of Cancer cell",dset.smoothness_mean.mean())
    a5 = st.text_input("Enter the compactness_mean of Cancer cell",dset.compactness_mean.mean())
    a6 = st.text_input("Enter the concavity_mean of Cancer cell",dset.concavity_mean.mean())
    a7 = st.text_input("Enter the concave points_mean of Cancer cell",0.048919146)
    a8 = st.text_input("Enter the symmetry_mean of Cancer cell",dset.symmetry_mean.mean())
    a9 = st.text_input("Enter the fractal_dimension_mean of Cancer cell",dset.fractal_dimension_mean.mean())
    a10 = st.text_input("Enter the radius_se of Cancer cell",dset.radius_se.mean())
    a11 = st.text_input("Enter the texture_se of Cancer cell",dset.texture_se.mean())
    a12 = st.text_input("Enter the perimeter_se of Cancer cell",dset.perimeter_se.mean())
    a13 = st.text_input("Enter the area_se of Cancer cell",dset.area_se.mean())
    a14 = st.text_input("Enter the smoothness_se of Cancer cell",dset.smoothness_se.mean())
    a15 = st.text_input("Enter the compactness_se of Cancer cell",dset.compactness_se.mean())
    a16 = st.text_input("Enter the concavity_se of Cancer cell",dset.concavity_se.mean())
    a17 = st.text_input("Enter the concave points_se of Cancer cell",0.011796137)
    a18 = st.text_input("Enter the symmetry_se of Cancer cell",dset.symmetry_se.mean())
    a19 = st.text_input("Enter the fractal_dimension_se of Cancer cell",dset.fractal_dimension_se.mean())
    a20 = st.text_input("Enter the radius_worst of Cancer cell",dset.radius_worst.mean())
    a21 = st.text_input("Enter the texture_worst of Cancer cell",dset.texture_worst.mean())
    a22 = st.text_input("Enter the perimeter_worst of Cancer cell",dset.perimeter_worst.mean())
    a23 = st.text_input("Enter the area_worst of Cancer cell",dset.area_worst.mean())
    a24 = st.text_input("Enter the smoothness_worst of Cancer cell",dset. smoothness_worst.mean())
    a25 = st.text_input("Enter the compactness_worst of Cancer cell",dset.compactness_worst.mean())
    a26 = st.text_input("Enter the concavity_worst of Cancer cell",dset.concavity_worst.mean())
    a27 = st.text_input("Enter the concave points_worst of Cancer cell",0.114606223)
    a28 = st.text_input("Enter the symmetry_worst of Cancer cell",dset.symmetry_worst.mean())
    a29 = st.text_input("Enter the fractal_dimension_worst of Cancer cell",dset.fractal_dimension_worst.mean())


    rslt=np.array([[a,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29]])
    ans = model.predict(rslt)
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(dset)
    if st.button('TEST'):
        if(ans == 1):
            st.error('I\'m sorry you cancer showing MELIGNANT cancer properties: ', icon="🚨")

        else:
            st.success('Your cancer cell showing BENIGN cancer properties: ', icon="✅")

        st.download_button(
                label="Download the dataset",
                data=csv,
                file_name='Cancer_Data.csv',
                mime='text/csv',
            )
