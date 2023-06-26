import streamlit as st
from web_functions import load_data,upload

from Tabs import home, predict, visualise

Tabs = {
    "Home": home,
    "Prediction" : predict,
    "Visualisation" : visualise,
}

#membuat sidebar
st.sidebar.title("Navigasi")

#membuat radio option
page = st.sidebar.radio("Pages",list(Tabs.keys()))

#upload file dataset
dataset =  st.file_uploader("Choose your database", accept_multiple_files=False,label_visibility="hidden")
if dataset is not None:
    file_name = dataset
else:
    file_name = "iris.csv"

#load dataset
df,x,y = load_data(file_name)
#kondisi call app function
if page in ["Prediction","Visualisation"]:
    Tabs[page].app(df,x,y)
else:
    Tabs[page].app() # type: ignore
