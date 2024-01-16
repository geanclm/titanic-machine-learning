# by geanclm on 14/01/2024
# with ChatGPT support for advanced subjects
# status: em desenvolvimento paralelo ao Machine Learning do Jupyter Notebook
# - - -
# pip install -r requirements.txt

# - - -
# AMBIENTE VIRTUAL
# Para evitar conflitos entre dependências de projetos diferentes
# é possível criar um ambiente virtual:
# 1 - python -m venv nomePastaProjeto
# 2 - definir a diretiva de execução para o computador local através do PowerShell:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
# 3 - .\Scripts\activate                                                                  # ativar o env
# 4 - .\Scripts\deactivate                                                                # desativar o env   
# 5 - pip list                                                                            # listar as dependências do env
# - - -

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# st.set_page_config(page_title="Predição de sobreviventes do Titanic")
# with st.container():
    # st.subheader("Previsão de Sobreviventes no Titanic")

st.title("Titanic - Machine Learning")

image_url = "310553_filme_Titanic_1997.JPG"
st.image(image_url, caption='Cena do filme Titanic (1997) dirigido por James Cameron', use_column_width=True)

st.write("Os passageiros, com menos de 1 ano de idade, SOBREVIVERAM!")
st.text(
    "Essa prática de navegar na tabela de dados faz parte da análise exploratória.\n"
    "Apresentar esse conjunto de dados, em fornmato gráfico,\n"
    "faz parte da Análise Explanatória dos Dados (AED).\n"
    "Definitivamente nenhuma análise caminha sem antes mesmo conhecer\n"
    "bem a base de dados e ter segurança no que pode ser feito em cada feature."
    )
st.write("Na sequência é possível conferir a tabela dos passageiros")

# leitura das bases de dados do Titanic - Kaggle
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
# df_train = pd.read_csv(r"C:\Users\geanc\OneDrive\Documentos\GitHub\titanic-machine-learning\train.csv")
# df_test = pd.read_csv(r"C:\Users\geanc\OneDrive\Documentos\GitHub\titanic-machine-learning\test.csv")

# st.write(df_train[df_train['Age']<1])
st.dataframe(df_train[df_train['Age'] < 1])

st.text(
    "Quando é necessário saber algo que poderia acontecer, com base no padrão dos\n"
    "dados conhecidos, utilizamos o Machine Learning. Neste caso,\n"
    "prever quem sobreviveu ao naufrágio é o desafio proposto pelo site Kaggle."
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
    "Neste modelo de Machine Learning, com acurácia dinâmica, é possível simular!"
    )

image_url = "Designer_Microsoft.jpeg"
st.image(image_url, caption='Imagem ilustrativa (IA Designer Microsoft)', use_column_width=True)


st.title("O Poder do Machine Learning!")
st.text( 
    "Previsão feita com um modelo ML gerado no Jupyter Notebook:\n"
    "Um homem que embarcou em Southampton (Inglaterra) na classe 2,\n"
    "com 35 anos de idade, sem parentes e com uma tarifa paga\n"
    "de 35 dólares, sobreviveria ao naufrágio do Titanic?"
    )
st.text(
    "Com esse perfil, segundo o modelo de Machine Learnming, de acurácia 87,68%,\n"
     "o passageiro SOBREVIVERIA ao naufrágio do Titanic."
     )

# Importar o arquivo com modelo salvo em joblib
# model_path = r"C:\Users\geanc\OneDrive\Documentos\GitHub\titanic-machine-learning\CLASSIFICAÇÃO_Titanic_Kaggle_csv_1-2024-01-16_dtc_0.8212290502793296.csv.joblib"
# model = joblib.load(model_path)

# Eliminar colunas sem efeito para o modelo
for df in [df_train, df_test]:    
    df = df.drop(['Name','Cabin','Ticket'], axis=1, inplace=True)

# Age - Ajustando os dados faltantes com a média de todas as idades
for df in [df_train, df_test]:    
    df.loc[df.Age.isnull(), 'Age'] = df.Age.mean()

# Embarked - Ajustando os dados faltantes com a moda
df_train.loc[df_train.Embarked.isnull(), 'Embarked'] = df_train.Embarked.mode()[0]

# Fare - Preencher os dados nulos com a média dos valores
for df in [df_test]:  
    df.loc[df.Fare.isnull(), 'Fare'] = df.Fare.mean()

# Embarked - Converter variável categórica em quantitativa ordinal
for df in [df_train, df_test]:
    def EmbarkedCat(x):
        if x == 'S':
            return 0
        elif x == 'C':
            return 1        
        else:
            return 2
    df['Embarked_c'] = df['Embarked'].apply(EmbarkedCat)    
    df.drop(['Embarked'], axis=1, inplace=True)

# Sex - Substituir textos por valores '0 = male' e '1 = female'
def sex(texto):
    if texto == 'female':
        return 1
    else:
        return 0
for df in [df_train, df_test]:  
    df['Sex_bin'] = df['Sex'].map(sex)

# SOMENTE COLUNAS NUMÉRICAS PARA O MACHINE LEARNING
col_df_train_nr = df_train.columns[df_train.dtypes != 'object']
col_df_test_nr = df_test.columns[df_test.dtypes != 'object']
# - - -
df_train_nr = df_train.loc[:,col_df_train_nr]
df_test_nr = df_test.loc[:,col_df_test_nr]

# st.dataframe(df_train_nr.tail())

# SEPARANDO VARIÁVEIS EM TREINO E TESTE
X = df_train_nr.drop(['PassengerId' ,'Survived'], axis=1)
y = df_train_nr.Survived
X_tr, X_ts, y_tr, y_ts = train_test_split(X , y, test_size = 0.2)

# # Treinar o modelo - Decision Tree Classifier
dtc = dtc2 = DecisionTreeClassifier(min_samples_split=4, criterion='gini', max_depth=4)
dtc.fit(X_tr, y_tr)
dtc_y_pred = dtc.predict(X_ts)

# Acurárcia do modelo
acc = accuracy_score(y_ts, dtc_y_pred)*100

st.title("Previsão de Sobrevivência")

st.text(    
    "Logo abaixo é possível simular um dierente perfil de passageiro,\n"
    "e em seguida o modelo de Machine Learning informará a previsão"      
    )

# Definindo valores padrão para os campos de início da tela
default_values = {
    "Pclass": 2,
    "Age": 35,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 35.50,
    "Embarked_c": 1,
    "Sex_bin": 0
}

# Recebendo dados do usuário para a previsão do modelo de Machine Learning
col1, col2 = st.columns(2)

Pclass = col1.number_input(
    label="Classe do passageiro [1, 2, 3]",
    step=1,
    format="%d",
    value=default_values["Pclass"],
    min_value=1,
    max_value=3
    )

Age = col2.number_input(label="Idade", step=1, format="%d", value=default_values["Age"])
SibSp = col1.number_input(label="Irmãos + Cônjuge:", step=1, format="%d", value=default_values["SibSp"])
Parch = col2.number_input(label="Pais + Filho(s):", step=1, format="%d", value=default_values["Parch"])
Fare = col1.number_input(label="Valor em dólar, da tarifa paga. Ex.: 30.50", value=default_values["Fare"])

Embarked_c = col2.number_input(
    label="Embarque: Inglaterra = 1, França = 2, Irlanda = 3",
    step=1,
    format="%d",
    min_value=1,
    max_value=3,
    value=default_values["Embarked_c"]
    )

Sex_bin = col1.number_input(
    label="Gênero Masculino = 0, Feminino = 1",
    step=1,
    format="%d",
    min_value=0,
    max_value=1,
    value=default_values["Sex_bin"]
    )

passageiro = np.array([[Pclass, Age, SibSp, Parch, Fare, Embarked_c, Sex_bin]])
df_passageiro = pd.DataFrame(passageiro, columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_c', 'Sex_bin'])
pred_passageiro = dtc.predict(df_passageiro)

st.write(f'A Inteligência Artificial (AI) está em constante treinamento')
st.write(f'Neste momento a ACURÁCIA do modelo é de {acc:.2f}%')

# if pred_passageiro == 0:
#     st.write(f'Com o perfil escolhido acima nos campos, o passageiro SOBREVIVERIA ao naufrágio do Titanic :-)')    
# else:
#     st.write(f'Passageiro NÃO sobreviveria :-()')
if pred_passageiro == 0:
    st.markdown('<p style="color: green; font-size: 18px;">Com o perfil escolhido acima nos campos, o passageiro SOBREVIVERIA ao naufrágio do Titanic :-)</p>', unsafe_allow_html=True)
else:
    st.markdown('<p style="color: red; font-size: 18px;">Passageiro NÃO sobreviveria :-(</p>', unsafe_allow_html=True)

st.write(
        "Este é o Mundo Novo onde a Inteligeência Artificial (IA)\n"
        "trabalha junto ao homem na busca por insights cada vez melhores,\n"
        "com o intuito de transformar a rotina de vida das pessoas\n"         
        )