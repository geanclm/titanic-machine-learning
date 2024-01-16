# Modelo interativo - Titanic - Machine Learning
# by geanclm on 16/01/2024
# versão simplificada para testar a execução do modelo

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Importar o arquivo com modelo salvo em joblib
# model_path = r"C:\Users\geanc\OneDrive\Documentos\GitHub\titanic-machine-learning\CLASSIFICAÇÃO_Titanic_Kaggle_csv_1-2024-01-16_dtc_0.8212290502793296.csv.joblib"
# model = joblib.load(model_path)

# leitura da base de dados
df_train = pd.read_csv(r"C:\Users\geanc\OneDrive\Documentos\GitHub\titanic-machine-learning\train.csv")
df_test = pd.read_csv(r"C:\Users\geanc\OneDrive\Documentos\GitHub\titanic-machine-learning\test.csv")

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

st.title('Aplicativo de Previsão')

st.text(
    "Selecione, logo abaixo, algumas características diferentes e o modelo\n"
    "de Machine Learning informará a previsão para esse passageiro"
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