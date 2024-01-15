# by geanclm on 14/01/2024
# with ChatGPT support for advanced subjects
# status: em desenvolvimento paralelo ao Machine Learning do Jupyter Notebook

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# st.set_page_config(page_title="Predição de sobreviventes do Titanic")
# with st.container():
    # st.subheader("Previsão de Sobreviventes no Titanic")

st.title("Titanic - Machine Learning")

image_url = "rms-titanic-bill-cannon.jpg"  # ou caminho local se a imagem estiver no mesmo diretório
st.image(image_url, caption='RMS Titanic', use_column_width=True)

st.write("Os passageiros, com menos de 1 ano de idade, SOBREVIVERAM!")
st.write("Essa prática, navegar na tabel de dados, faz parte da análise exploratório dos dados")
st.write("Apresentar esse data framne, em fornmato gráfico, faz parte da Análise Explanatória dos Dados (AED)")

# dataframe treino da base Titanic - Kaggle
df_train = pd.read_csv('train.csv')

# st.write(df_train[df_train['Age']<1])
st.dataframe(df_train[df_train['Age'] < 1])

st.text(
    "Quando é necessário saber algo que poderia acontecer, com base no padrão dos dados\n
    "conhecidos, utilizamos o Machine Learning, que é exatamente prever quem sobreviveu,\n
    "e melhor ainda quem sobreviveria considerando um novo perfil de passageiro"
    )

# Criar histograma
fig, ax = plt.subplots(figsize=(15, 8))
df_train.hist(ax=ax)
plt.tight_layout()

# Exibir no Streamlit
st.pyplot(fig)

st.text(
    "No histograma (hist), por exemplo, podemos verificar a contagem de cada situação presente na base de dados.\n"
    "Nem todas as colunas são utilizadas no Machine Learning, e aqui estão publicadas apenas para efeito de entendimento dos dados presentes na base.\n"
    "No primeiro hist temos a informação de que a maioria da identificação do passageiro está abaixo do número 800;\n"
    "No segundo hist 'Survived' é visível a informação de que todas as crianças sobreviveram;\n"
    "Terceiro hist, as crianças estavam mais presentes na classe 2 e 3 do Titanic;\n"
    "Histograma 'Age', percebe-se mais crianças entre 7 e 9 meses de idade;\n"
    "O hist 'SibSp' informa o número de irmãos ou conjûges, e a maioria das crianças tinha pelo menos um irmão a bordo;\n"
    "O hist 'Parch' informa se o passageiro tinha pais ou filhos, e nesse caso a maioria estava ou com o pai ou com mãe, a bordo;\n"
    "Na última coluna da base, 'Fare', temos a informação da tarifa paga pelo passageiro. A maioria das crianças pagou menos de $25 na passagem."
    )

st.text(
    "Uma pessoa com outro perfil sobreviveria ao naufrágio do Titanic?\n"
    "Neste modelo de Machine Learning, com acurácia de 87,6% de acerto\n
    "é possível simular!"
    )