import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import streamlit as st
import numpy as np
import pandas as pd

from web_functions import train_model_DT, train_model_KNN, train_model_NBC, load_data


def app(df, x, y):
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Visualisasi Dataset Tanaman Iris")
    st.title("Pairplot")
    # s = sns.pairplot(df, hue="Species")
    st.pyplot(sns.pairplot(df, hue="Species"))  # type: ignore
    st.subheader("Penjelasan Pair Plot")
    st.markdown("""
        Gambar di atas adalah *pair plot*, yang merupakan visualisasi dari hubungan antar variabel pada dataset *Iris*. Berikut adalah penjelasan lebih lanjut:

        1. **Dataset Iris**: Dataset ini terdiri dari tiga spesies bunga Iris—*Iris-setosa* (biru), *Iris-versicolor* (oranye), dan *Iris-virginica* (hijau)—yang diklasifikasikan berdasarkan empat fitur:
        - **SepalLengthCm**: Panjang kelopak bunga (sepal) dalam cm.
        - **SepalWidthCm**: Lebar kelopak bunga dalam cm.
        - **PetalLengthCm**: Panjang mahkota bunga (petal) dalam cm.
        - **PetalWidthCm**: Lebar mahkota bunga dalam cm.
        - **Id**: Indeks untuk identifikasi.

        2. **Distribusi diagonal**: Di sepanjang diagonal, terlihat distribusi (plot densitas atau histogram) dari masing-masing fitur secara individu. Warna-warna pada distribusi ini mewakili spesies yang berbeda.

        3. **Scatter plot non-diagonal**: Di bagian lain dari pair plot ini, terlihat plot pencar (scatter plot) yang menunjukkan hubungan antara dua fitur. Setiap titik pada scatter plot mewakili satu contoh dari dataset, dengan warnanya mencerminkan spesiesnya. 

        4. **Klasifikasi visual**: Gambar ini memberikan indikasi seberapa mudah atau sulit spesies bunga dapat diklasifikasikan berdasarkan fitur tertentu. Beberapa pasangan fitur memperlihatkan pemisahan yang lebih jelas antar spesies, sedangkan beberapa pasangan lainnya menunjukkan tumpang tindih yang lebih besar.
        """)
    st.markdown("""

        Berikut ini adalah visualisasi scatter plot yang menggambarkan perbandingan panjang dan lebar **kelopak** serta **sepal** dari setiap spesies bunga Iris.
        """)

    # if st.checkbox("Plot Decision Tree"):
    #     model,score = train_model_DT(x,y)
    #     dot_data = tree.export_graphviz(
    #         decision_tree=model, out_file=None, filled=True, rounded=True,
    #         feature_names=x.columns, class_names=['Iris-setosa','Iris-versicolor','Iris-virginica']
    #     )
    #     st.graphviz_chart(dot_data)
    # elif st.checkbox("Pairplot"):
    Xv = df.loc[:, df.columns != 'Species']

    yv = df['Species']

    # Membuat plot
    plt.figure(figsize=(6, 5))
    subspecies = df.Species.unique()
    color = ['red', 'green', 'blue']

    for i in range(len(subspecies)):
        plt.scatter(Xv[df.Species == subspecies[i]]['PetalLengthCm'],
                    Xv[df.Species == subspecies[i]]['PetalWidthCm'],
                    c=color[i], label=subspecies[i])

    plt.legend()
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.title('Scatter plot of IRIS flowers')

    # Tampilkan plot di Streamlit
    st.subheader('Scatter Plot: Panjang dan Lebar Petal')
    st.pyplot(plt)  # type: ignore
    st.markdown("""

    Pada plot di atas, kita dapat melihat perbandingan antara panjang dan lebar kelopak (petal) dari tiga spesies bunga Iris. Warna merah mewakili **Iris-setosa**, warna hijau mewakili **Iris-versicolor**, dan warna biru mewakili **Iris-virginica**.

    Dari visualisasi ini, tampak bahwa spesies **Iris-setosa** memiliki panjang dan lebar kelopak yang relatif lebih kecil dibandingkan dengan dua spesies lainnya. Sedangkan **Iris-virginica** memiliki kelopak yang lebih panjang dan lebar dibandingkan **Iris-versicolor**.
    """)
    plt.figure(figsize=(6, 5))

    subspecies = df.Species.unique()
    color = ['red', 'green', 'blue']
    for i in range(len(subspecies)):
        plt.scatter(Xv[df.Species == subspecies[i]]['SepalLengthCm'], Xv[df.Species ==
                    subspecies[i]]['SepalWidthCm'], c=color[i], label=subspecies[i])
    plt.legend()
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('Scatter plot of IRIS flowers')
    st.subheader('Scatter Plot: Panjang dan Lebar Sepal')
    st.pyplot(plt)  # type: ignore
    st.markdown("""

    Pada plot ini, kita memvisualisasikan perbandingan panjang dan lebar **sepal** untuk ketiga spesies bunga Iris. Warna tetap sama, yaitu merah untuk **Iris-setosa**, hijau untuk **Iris-versicolor**, dan biru untuk **Iris-virginica**.

    Dari plot ini, kita bisa melihat bahwa **Iris-setosa** cenderung memiliki ukuran sepal yang lebih kecil (baik panjang maupun lebar) dibandingkan dengan kedua spesies lainnya. Di sisi lain, **Iris-virginica** dan **Iris-versicolor** memiliki sepal yang lebih mirip dalam hal panjang, namun lebar sepalnya sedikit berbeda.

    ## Kesimpulan

    Visualisasi ini membantu kita memahami bagaimana fitur panjang dan lebar kelopak serta sepal dapat digunakan untuk membedakan ketiga spesies bunga Iris. Berdasarkan pola yang muncul, kita dapat dengan mudah membedakan spesies **Iris-setosa** dari spesies lainnya, terutama karena ukuran kelopak dan sepalnya yang lebih kecil. Namun, untuk membedakan antara **Iris-versicolor** dan **Iris-virginica**, kita perlu memperhatikan detail lebih lanjut karena ukuran sepal dan kelopaknya lebih mirip.
    """)
    
