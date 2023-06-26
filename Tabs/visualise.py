import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import streamlit as st

from web_functions import train_model_DT,train_model_KNN,train_model_NBC

def app(df, x, y):
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Visualisasi Prediksi Tanaman Iris")


    if st.checkbox("Plot Decision Tree"):
        model,score = train_model_DT(x,y)
        dot_data = tree.export_graphviz(
            decision_tree=model, out_file=None, filled=True, rounded=True,
            feature_names=x.columns, class_names=['Iris-setosa','Iris-versicolor','Iris-virginica']
        )
        st.graphviz_chart(dot_data)
    elif st.checkbox("Confusion Matrix"):
        cm = train_model_NBC(x,y)
        
        st.write('\nConfusion matrix for Naive Bayes\n',cm)
        
        
