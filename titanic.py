# by geanclm on 14/01/2024
# with ChatGPT support for advanced subjects
# status: em desenvolvimento paralelo ao Machine Learning do Jupyter Notebook

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Predição de sobreviventes do Titanic")
with st.container():
    st.subheader("Projeto de Machine Learning: Previsão de Sobrevivência no Titanic")
    st.title("Machine Learning com Python")    
    # st.write("Para saber mais sobre o evento do naufrágio [Clique aqui] (https://pt.wikipedia.org/wiki/RMS_Titanic)")
    st.write("Nesse momento a boa notícia é a de que todos os passageiros com menos de 1 ano de idade sobreviveram")

# @st.cache_data
# def carregar_dados():
#     tabela = pd.read_csv("resultados.csv")
#     return tabela

# with st.container():
#     st.write("---")
#     qtde_dias = st.selectbox("Selecione o período", ["7D", "15D", "21D", "30D"])
#     num_dias = int(qtde_dias.replace("D", ""))
#     dados = carregar_dados()
#     dados = dados[-num_dias:]
#     st.area_chart(dados, x="Data", y="Contratos")

# Carregue o dataframe
df_train = pd.read_csv('train.csv')

# comando df_train.tail() para mostrar as últimas linhas do dataframe no streamlit
st.write(df_train[df_train['Age']<1])