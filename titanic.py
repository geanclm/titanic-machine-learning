# by geanclm on 14/01/2024
# with ChatGPT support for advanced subjects
# status: em desenvolvimento paralelo ao Machine Learning do Jupyter Notebook

pip install joblib

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# st.set_page_config(page_title="Predição de sobreviventes do Titanic")
# with st.container():
    # st.subheader("Previsão de Sobreviventes no Titanic")

st.title("Titanic - Machine Learning")

image_url = "rms-titanic-bill-cannon.jpg"
st.image(image_url, caption='RMS Titanic (abril de 1912)', use_column_width=True)

st.write("Os passageiros, com menos de 1 ano de idade, SOBREVIVERAM!")
st.text(
    "Essa prática, navegar na tabel de dados, faz parte da análise exploratória\n"
    "dos dados, e apresentar esse conjunto de dados, em fornmato gráfico,\n"
    "faz parte da Análise Explanatória dos Dados (AED)"
    )
st.write("Na sequência é possível conferir a tabela dos passageiros")

# dataframe treino da base Titanic - Kaggle
df_train = pd.read_csv('train.csv')

# st.write(df_train[df_train['Age']<1])
st.dataframe(df_train[df_train['Age'] < 1])

st.text(
    "Quando é necessário saber algo que poderia acontecer, com base no padrão dos\n"
    "dados conhecidos, utilizamos o Machine Learning, que é exatamente, neste caso,\n"
    "prever quem sobreviveu ao naufrágio, desafio proposto pelo site Kaggle."
    )

# # Criar histograma
# fig, ax = plt.subplots(figsize=(15, 8))
# df_train.hist(ax=ax)
# plt.tight_layout()

# # Exibir no Streamlit
# st.pyplot(fig)

# st.text(
#     "No histograma (hist), por exemplo, podemos verificar a contagem de cada situação presente na base de dados.\n"
#     "Nem todas as colunas são utilizadas no Machine Learning, e aqui estão publicadas apenas para efeito de entendimento dos dados presentes na base.\n"
#     "No primeiro hist temos a informação de que a maioria da identificação do passageiro está abaixo do número 800;\n"
#     "No segundo hist 'Survived' é visível a informação de que todas as crianças sobreviveram;\n"
#     "Terceiro hist, as crianças estavam mais presentes na classe 2 e 3 do Titanic;\n"
#     "Histograma 'Age', percebe-se mais crianças entre 7 e 9 meses de idade;\n"
#     "O hist 'SibSp' informa o número de irmãos ou conjûges, e a maioria das crianças tinha pelo menos um irmão a bordo;\n"
#     "O hist 'Parch' informa se o passageiro tinha pais ou filhos, e nesse caso a maioria estava ou com o pai ou com mãe, a bordo;\n"
#     "Na última coluna da base, 'Fare', temos a informação da tarifa paga pelo passageiro. A maioria das crianças pagou menos de $25 na passagem."
#     )

st.text(    
    "Uma pessoa de nossa época sobreviveria ao Titanic?\n"
    "Neste modelo de Machine Learning, com acurácia de 87,68% é possível simular!"
    )

image_url = "Designer_Microsoft.jpeg"
st.image(image_url, caption='Imagem ilustrativa (IA Designer Microsoft)', use_column_width=True)

st.text( 
    "Um homem que embarcou em Southampton (Inglaterra)/n"
    "na classe 2, com 35 anos de idade, sem parentes e com uma tarifa paga de $35,/n"
    "sobreviveria ao naufrágio do Titanic?"
    )




st.write("Digite os dados para que a Inteligência Artificial (IA) apresente a previsão")




# texto


# Definir as colunas
colunas = ['Age', 'Fare', 'Sex_bin', 'Cabin_bin', 'Embarked_Q', 'Embarked_S',
           'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Person', 'Pclass_2',
           'Pclass_3', 'familySize_c_Pequena', 'familySize_c_Sozinho']

# Criar interface do Streamlit
st.title("Previsão de Sobrevivência no Titanic")

# Criar sliders para cada coluna
valores_colunas = {}
for coluna in colunas:
    valores_colunas[coluna] = st.slider(f'Selecione o valor para {coluna}', min_value=0, max_value=1, step=1)

# Criar DataFrame com os valores selecionados
df_passageiro = pd.DataFrame([valores_colunas])

# CARREGAR modelo salvo para previsão
lerOutroModelo = 'CLASSIFICAÇÃO_Titanic_Kaggle_csv_2-2024-01-11_rfc_08768656716417911.csv.joblib'
persistentModel = joblib.load(lerOutroModelo)

# Realizar previsão
pred_passageiro = persistentModel.predict(df_passageiro)

# Exibir resultado no Streamlit
if pred_passageiro == 0:
    st.write("Com esse perfil, e segundo o modelo de acurácia 87,68%, o passageiro SOBREVIVERIA ao naufrágio do Titanic")
else:
    st.write("Passageiro NÃO sobreviveria")
