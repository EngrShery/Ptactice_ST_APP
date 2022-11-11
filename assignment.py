import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

# make containers
header=st.container()
data_sets=st.container()
features=st.container()
model_training=st.container()


with header:
    st.title("Pakistanio ki favourite Khorak")


with data_sets:
    st.text("We will work on data obtained by students of Python_Chilla ")
    #import data
    st.subheader("Datset")
    df=pd.read_csv('mldata_biryani.csv')
    st.write(df.head(10))
    #bar chart
    st.subheader("Bar chart of likeness")
    st.bar_chart(df["likeness"].value_counts())
    
    # another bar chart
    st.subheader("Bar chart of weight")
    st.bar_chart(df["weight"].sample(10))



with features:
    st.subheader("The following are the main features of app:")
    st.markdown("- **Feature :** This app will tell us about likeliness of people towards traditional food based on their age and weight")

with model_training:
    st.subheader("Finding accuracy scores of model")  
    
    #making columns
    input, display=st.columns(2)
    #selection points in 1st column
    max_depth=input.slider("How many people do you know?", min_value=10,max_value=50,value=10,step=5)
    
    # n_estimators
    n_estimators=input.selectbox("How many trees should be there?",options=[50,100,200,200,'No limit'])

    #adding list of features
    input.write(df.columns)



    #ML model
    model=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)

    if n_estimators=="No limit":
        model=RandomForestClassifier(max_depth=max_depth)
    else:
        model=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)


#define X,y
    X=df[['age','weight','height']]
    y=df[["likeness"]]    
    model.fit(X,y)
    prediction=model.predict(X)

        #Display metrices
    display.subheader("Accuracy score of the model is:")
    display.write(accuracy_score(y,prediction))
    display.subheader("Precision score of the model is:")
    display.write(precision_score(y,prediction,pos_label='positive',
                                           average='micro'))
    display.subheader("Recall score of the model is:")
    display.write(recall_score(y,prediction,pos_label='positive',
                                           average='micro'))
    display.subheader("F1 score of the model is:")
    display.write(f1_score(y,prediction,pos_label='positive',
                                           average='micro'))    

    
   




