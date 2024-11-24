import pandas as pd;
import streamlit as st;
from sklearn import linear_model;

#Carregar os dados: (Pandas)

df = pd.read_csv("pizzas.csv") #lendo o arquivo csv

#Treinar meu modelo: (SciKit-Learn)

modelo = linear_model.LinearRegression() #Criando Modelo / definindo o tipo de resolucao e definindo os parametros (x,y)
x = df[["diametro"]]
y = df[["preco"]]

modelo.fit(x,y) #treinando o modelo com base no arquivo csv

#Criar AppWeb (Streamlit)

st.title("Prevendo valor:")
st.divider()

diametro = st.number_input("Tamanho:")

if diametro:
    preco_previsto = modelo.predict([[diametro]])[0][0]
    st.write(f"O preço de {diametro: .0f}cm é de R${preco_previsto: .2f}")