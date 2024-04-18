#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, LabelBinarizer, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb


# In[2]:


#Import database
df = pd.read_csv('C:/Users/502694/Desktop/Dissertação/Análises Oficial/Base final.csv',sep=";")
print(df.info())


# In[3]:


#Replace , for . in columns
columns_to_replace = ['CAAF', 'CA', 'ID Matriz', 'Peso Alojamento', 'Pre-Inicial(kg/ave)', 'Inicial(Kg/ave)','Crescimento - I(Kg/ave)','Final-II(Kg/ave)','% MT','% Mort 07','% Mort 14',
                      '% Mort 21','Densidade Alojamento','Densidade Abate (Kg/M²)','Real_Calo_Pata_A','ID Abate','% Condenação Total','% Condenação Parcial','PM']
df[columns_to_replace] = df[columns_to_replace].replace(',', '.', regex=True)

# Convert columns to numeric format
numeric_columns = ['CAAF', 'CA', 'ID Matriz', 'Peso Alojamento', 'Pre-Inicial(kg/ave)', 'Inicial(Kg/ave)','Crescimento - I(Kg/ave)','Final-II(Kg/ave)','% MT','% Mort 07','% Mort 14',
                   '% Mort 21','Densidade Alojamento','Densidade Abate (Kg/M²)','Real_Calo_Pata_A','ID Abate','% Condenação Total','% Condenação Parcial','PM','Intervalo Diário','Dist. Abatedouro',
                   'Área Util',]                    
df[numeric_columns] = df[numeric_columns].astype(float)

# Convert columns to integer format
df['Nº Cama'] = df['Nº Cama'].astype(int)

# Convert columns to category format
df['Extensionista'] = df['Extensionista'].astype(str).str.replace('\.0', '', regex=True)
category_columns = ['Nº Lote','Linhagem','Timer Iluminação','Origem Incubatório','Sexo','Categoria','Tipo Ventilação','Tipo Telhado','Município','Origem Água','Extensionista','Nº Fornecedor']
df[category_columns] = df[category_columns].astype('category')

#Convert to datetime
df['Data Hora Aloj'] = pd.to_datetime(df['Data Hora Aloj'], dayfirst=True)
df['Data Abate'] = pd.to_datetime(df['Data Abate'], dayfirst=True)

#print(df.head(30))
#print(df.info())
#print(df.groupby('Município')['Nº Lote'].count())


# In[4]:


#Drop rows data have CAAF Above 1.9 and Extensionista NaN
df = df[df['CAAF'] <= 1.9]
df.dropna(subset=['Extensionista'], inplace=True)
#print((df['CAAF'] > 1.9).any())
#df['Extensionista'].isnull().sum()


# In[5]:


#Replace empty columns in categorical for NaN
columns_to_replace_empty_NaN=['Tipo Ventilação','Tipo Telhado','Origem Água','Timer Iluminação','Extensionista']
df[columns_to_replace_empty_NaN] = df[columns_to_replace_empty_NaN].replace('nan', pd.NA)
#print(df.isnull().sum())
#print(df.head(60))


# In[6]:


#Create timelapse columns
#Convert columns to datetime format
df['Data Hora Aloj'] = pd.to_datetime(df['Data Hora Aloj'], format='%d/%m/%Y %H:%M:%S')
df['Data Abate'] = pd.to_datetime(df['Data Abate'], format='%d/%m/%Y %H:%M:%S')

# Create new columns for day, month and year
df['Dia Aloj'] = df['Data Hora Aloj'].dt.day
df['Mês Aloj'] = df['Data Hora Aloj'].dt.month
df['Ano Aloj'] = df['Data Hora Aloj'].dt.year

df['Dia Abate'] = df['Data Abate'].dt.day
df['Mês Abate'] = df['Data Abate'].dt.month
df['Ano Abate'] = df['Data Abate'].dt.year

# Create new coolumn for year season
def estacao_do_ano(mes):
    if 3 <= mes <= 5:
        return 'Outono'
    elif 6 <= mes <= 8:
        return 'Inverno'
    elif 9 <= mes <= 11:
        return 'Primavera'
    else:
        return 'Verão'

df['Estação Aloj'] = df['Mês Aloj'].apply(estacao_do_ano)
df['Estação Abate'] = df['Mês Abate'].apply(estacao_do_ano)

# Create new column for day period
def periodo_do_dia(hora):
    if 6 <= hora < 12:
        return 'Manhã'
    elif 12 <= hora < 18:
        return 'Tarde'
    else:
        return 'Noite'

df['Período Aloj'] = df['Data Hora Aloj'].dt.hour.apply(periodo_do_dia)
df['Período Abate'] = df['Data Abate'].dt.hour.apply(periodo_do_dia)

#Convert columns type
columns_category = ['Dia Aloj','Mês Aloj','Ano Aloj','Dia Abate','Mês Abate','Ano Abate','Estação Aloj','Estação Abate','Período Aloj','Período Abate']
df[columns_category] = df[columns_category].astype('category')
#print(df.info())


# In[7]:


# Replace wrong information in categorical columns
# Correct 'Origem Água'
df['Origem Água'] = df['Origem Água'].replace(['Fonte - Clorada, Poço - Clorada', 
                                               'Fonte - Clorada, Poço Artesiano - Clorada'], 
                                              pd.NA)

df['Origem Água'] = df['Origem Água'].replace({'Fonte - Clorada': 'Fonte',
                                                'Fonte - Não Clorada': 'Fonte',
                                                'Poço - Clorada': 'Poço',
                                               'Poço - Clorada, Poço Artesiano - Clorada': 'Poço',
                                                'Poço - não Clorada': 'Poço',
                                               'Poço - não Clorada, Poço Artesiano - Clorada': 'Poço',
                                                'Poço Artesiano - Clorada': 'Poço',
                                                'Poço Artesiano - não Clorada': 'Poço'})
# Correct 'Tipo Ventilação'
df['Tipo Ventilação'] = df['Tipo Ventilação'].replace(['Dark Produtor, Semi Dark'], 
                                              pd.NA)
# Correct 'Tipo Telhado' 
df['Tipo Telhado'] = df['Tipo Telhado'].replace(['Amianto, Misto',
                                                  'Amianto, Telha Barro',
                                                   'Outro',
                                                   'Misto',
                                                   'Telha Barro, Zinco Galvanizado'
                                                  ], 
                                              pd.NA)

# Correct 'Timer Iluminação' 
df['Timer Iluminação'] = df['Timer Iluminação'].replace(['Não, Sim'
                                                  ], 
                                              pd.NA)

#Replace NaN by new category 'Desconhecido'
df['Timer Iluminação'] = df['Timer Iluminação'].cat.add_categories('desconhecido')
df['Timer Iluminação'].fillna('desconhecido', inplace=True)

# Replace NaN by mode
for col in ['Tipo Ventilação', 'Origem Água', 'Tipo Telhado']:
    mode_value = df[col].mode()[0] 
    df[col].fillna(mode_value, inplace=True) 

#print(df[['Origem Água','Tipo Ventilação','Tipo Telhado']].value_counts())


# In[8]:


#Replace wrong information and cleaning missing values in numerical columns
#ID Matriz - Replace wrong information by mode
df.loc[(df['ID Matriz'] < 27) | (df['ID Matriz'] > 65), 'ID Matriz'] = pd.NA
for col in ['ID Matriz']:
    mode_value = df[col].mode()[0] 
    df[col].fillna(mode_value, inplace=True)

#Pre-Inicial(kg/ave) - Replace wrong information by inferior and superior limits & Replace NaN by mode
limit_pre_2023 = 0.300
limit_pos_2023 = 0.447

condicao_1 = ((df['Data Hora Aloj'] < '2023-10-10') & ((df['Pre-Inicial(kg/ave)'] < 0.8 * limit_pre_2023) | 
                                                       (df['Pre-Inicial(kg/ave)'] > 1.2 * limit_pre_2023)))
condicao_2 = ((df['Data Hora Aloj'] >= '2023-10-10') & ((df['Pre-Inicial(kg/ave)'] < 0.8 * limit_pos_2023) | 
                                                        (df['Pre-Inicial(kg/ave)'] > 1.2 * limit_pos_2023)))

df.loc[condicao_1, 'Pre-Inicial(kg/ave)'] = df.loc[condicao_1, 'Pre-Inicial(kg/ave)'].apply(lambda x: 0.8 * limit_pre_2023 
                                                                                            if x < 0.8 * limit_pre_2023 
                                                                                            else (1.2 * limit_pre_2023 
                                                                                                  if x > 1.2 * limit_pre_2023 else x))
df.loc[condicao_2, 'Pre-Inicial(kg/ave)'] = df.loc[condicao_2, 'Pre-Inicial(kg/ave)'].apply(lambda x: 0.8 * limit_pos_2023 
                                                                                            if x < 0.8 * limit_pos_2023 
                                                                                            else (1.2 * limit_pos_2023 
                                                                                                  if x > 1.2 * limit_pos_2023 else x))
df['Pre-Inicial(kg/ave)'].fillna(df['Pre-Inicial(kg/ave)'].mode()[0], inplace=True)
df['Pre-Inicial(kg/ave)'] = df['Pre-Inicial(kg/ave)'].round(3)

#Inicial(Kg/ave) - Replace wrong information by inferior and superior limits
limit_pre_2023 = 0.387
limit_pos_2023 = 0.385

condicao_1 = ((df['Data Hora Aloj'] < '2023-10-10') & ((df['Inicial(Kg/ave)'] < 0.8 * limit_pre_2023) | 
                                                       (df['Inicial(Kg/ave)'] > 1.2 * limit_pre_2023)))
condicao_2 = ((df['Data Hora Aloj'] >= '2023-10-10') & ((df['Inicial(Kg/ave)'] < 0.8 * limit_pos_2023) | 
                                                        (df['Inicial(Kg/ave)'] > 1.2 * limit_pos_2023)))

df.loc[condicao_1, 'Inicial(Kg/ave)'] = df.loc[condicao_1, 'Inicial(Kg/ave)'].apply(lambda x: 0.8 * limit_pre_2023 
                                                                                            if x < 0.8 * limit_pre_2023 
                                                                                            else (1.2 * limit_pre_2023 
                                                                                                  if x > 1.2 * limit_pre_2023 else x))
df.loc[condicao_2, 'Inicial(Kg/ave)'] = df.loc[condicao_2, 'Inicial(Kg/ave)'].apply(lambda x: 0.8 * limit_pos_2023 
                                                                                            if x < 0.8 * limit_pos_2023 
                                                                                            else (1.2 * limit_pos_2023 
                                                                                                  if x > 1.2 * limit_pos_2023 else x))
df['Inicial(Kg/ave)'].fillna(df['Inicial(Kg/ave)'].mode()[0], inplace=True)
df['Inicial(Kg/ave)'] = df['Inicial(Kg/ave)'].round(3)

#Crescimento - I(Kg/ave) - Replace wrong information by inferior and superior limits
limit_pre_2023 = 0.899
limit_pos_2023 = 0.638

condicao_1 = ((df['Data Hora Aloj'] < '2023-10-10') & ((df['Crescimento - I(Kg/ave)'] < 0.8 * limit_pre_2023) | 
                                                       (df['Crescimento - I(Kg/ave)'] > 1.2 * limit_pre_2023)))
condicao_2 = ((df['Data Hora Aloj'] >= '2023-10-10') & ((df['Crescimento - I(Kg/ave)'] < 0.8 * limit_pos_2023) | 
                                                        (df['Crescimento - I(Kg/ave)'] > 1.2 * limit_pos_2023)))

df.loc[condicao_1, 'Crescimento - I(Kg/ave)'] = df.loc[condicao_1, 'Crescimento - I(Kg/ave)'].apply(lambda x: 0.8 * limit_pre_2023 
                                                                                            if x < 0.8 * limit_pre_2023 
                                                                                            else (1.2 * limit_pre_2023 
                                                                                                  if x > 1.2 * limit_pre_2023 else x))
df.loc[condicao_2, 'Crescimento - I(Kg/ave)'] = df.loc[condicao_2, 'Crescimento - I(Kg/ave)'].apply(lambda x: 0.8 * limit_pos_2023 
                                                                                            if x < 0.8 * limit_pos_2023 
                                                                                            else (1.2 * limit_pos_2023 
                                                                                                  if x > 1.2 * limit_pos_2023 else x))
df['Crescimento - I(Kg/ave)'].fillna(df['Crescimento - I(Kg/ave)'].mode()[0], inplace=True)
df['Crescimento - I(Kg/ave)'] = df['Crescimento - I(Kg/ave)'].round(3)

#Final-II(Kg/ave) - Replace wrong information by inferior and superior limits
limit_pre_2023 = 0.668
limit_pos_2023 = 0.668

condicao_1 = ((df['Data Hora Aloj'] < '2023-10-10') & ((df['Final-II(Kg/ave)'] < 0.8 * limit_pre_2023) | 
                                                       (df['Final-II(Kg/ave)'] > 1.2 * limit_pre_2023)))
condicao_2 = ((df['Data Hora Aloj'] >= '2023-10-10') & ((df['Final-II(Kg/ave)'] < 0.8 * limit_pos_2023) | 
                                                        (df['Final-II(Kg/ave)'] > 1.2 * limit_pos_2023)))

df.loc[condicao_1, 'Final-II(Kg/ave)'] = df.loc[condicao_1, 'Final-II(Kg/ave)'].apply(lambda x: 0.8 * limit_pre_2023 
                                                                                            if x < 0.8 * limit_pre_2023 
                                                                                            else (1.2 * limit_pre_2023 
                                                                                                  if x > 1.2 * limit_pre_2023 else x))
df.loc[condicao_2, 'Final-II(Kg/ave)'] = df.loc[condicao_2, 'Final-II(Kg/ave)'].apply(lambda x: 0.8 * limit_pos_2023 
                                                                                            if x < 0.8 * limit_pos_2023 
                                                                                            else (1.2 * limit_pos_2023 
                                                                                                  if x > 1.2 * limit_pos_2023 else x))
df['Final-II(Kg/ave)'].fillna(df['Final-II(Kg/ave)'].mode()[0], inplace=True)
df['Final-II(Kg/ave)'] = df['Final-II(Kg/ave)'].round(3)

#% MT
df.loc[df['% MT'] < 1, '% MT'] = 1
df.loc[df['% MT'] > 15, '% MT'] = 15
#print((df['% MT'] <1).any())

# % Mort 07 , % Mort 14 , % Mort 21
colunas = ['% Mort 07', '% Mort 14', '% Mort 21']
for coluna in colunas:
    df.loc[df[coluna] < 0.3, coluna] = 0.3 
#print((df['% Mort 07'] <0.3).any())   

# Densidade Alojamento
df.loc[df['Densidade Alojamento'] < 13, 'Densidade Alojamento'] = 13
df.loc[df['Densidade Alojamento'] > 25, 'Densidade Alojamento'] = 25

# Densidade Abate (Kg/M²)
df.loc[df['Densidade Abate (Kg/M²)'] < 18, 'Densidade Abate (Kg/M²)'] = 18
df.loc[df['Densidade Abate (Kg/M²)'] > 30, 'Densidade Abate (Kg/M²)'] = 30

# Real_Calo_Pata_A
df.loc[df['Real_Calo_Pata_A'] > 100, 'Real_Calo_Pata_A'] = 100

# ID Abate
df.loc[df['ID Abate'] < 24, 'ID Abate'] = 24
df.loc[df['ID Abate'] > 35, 'ID Abate'] = 35

# % Condenação Total
df.loc[df['% Condenação Total'] < 0.1, '% Condenação Total'] = 0.1
df.loc[df['% Condenação Total'] > 20, '% Condenação Total'] = 20

# % Condenação Parcial
df.loc[df['% Condenação Parcial'] < 0.1, '% Condenação Parcial'] = 100

# Dist. Abatedouro - 0 values by mode of the city
city_modes = df.groupby('Município')['Dist. Abatedouro'].apply(lambda x: x.mode().iloc[0])
for city, mode_distance in city_modes.items():
    df.loc[(df['Dist. Abatedouro'] == 0) & (df['Município'] == city), 'Dist. Abatedouro'] = mode_distance
#df.loc[(df['Dist. Abatedouro'] == 0), 'Dist. Abatedouro'] = pd.NA
#print((df['Dist. Abatedouro'] ==0).any()) 


# In[9]:


# Group informations in the columns
# ID Matriz 5 per 5
limites = list(range(27, 71, 5))
rotulos = [f"{inicio} a {inicio + 4} semanas" for inicio in limites[:-1]]
df['ID Matriz'] = pd.cut(df['ID Matriz'], bins=limites, labels=rotulos, right=False)
#print((df['ID Matriz'].head(30)))
#print(df.info())

#Peso Alojamento 3 per 3
limites_peso = list(np.arange(0.030, 0.061, 0.003))
rotulos_peso = [f"{round(inicio,3)} a {round(inicio + 0.002, 3)} Kg" for inicio in limites_peso[:-1]]
df['Peso Alojamento'] = pd.cut(df['Peso Alojamento'], bins=limites_peso, labels=rotulos_peso, right=False)
#print(df['Peso Alojamento'].head(30))
#print(df['Peso Alojamento'].info())

# Área Útil 
limites_area_util = [0, 1200, 1800, 2400, 3000, 6000]
rotulos_area_util = ['<=1200 m²', '1201 a 1800 m²', '1801 a 2400 m²', '2401 a 3000 m²', '3001 a 6000 m²']
df['Área Util'] = pd.cut(df['Área Util'], bins=limites_area_util, labels=rotulos_area_util, right=False)
#print(df['Área Util'].head(30))
#print(df['Área Util'].info())

#Nº camas
limites_camas = list(range(1, df['Nº Cama'].max() + 4, 3))
rotulos_camas = [f"{inicio} a {inicio + 2} camas" for inicio in range(1, df['Nº Cama'].max() + 1, 3)]
df['Nº Cama'] = pd.cut(df['Nº Cama'], bins=limites_camas, labels=rotulos_camas, right=False)
#print(df['Área Util'].head(30))
#print(df['Área Util'].info())

# Intervalo Diário 
limites_intervalo = [0, 6, 11, 16, df['Intervalo Diário'].max() + 1]
rotulos_intervalo = ['Até 6 dias', '7 a 11 dias', '12 a 16 dias', 'Acima de 17 dias']
df['Intervalo Diário'] = pd.cut(df['Intervalo Diário'], bins=limites_intervalo, labels=rotulos_intervalo, right=False)
#print(df['Intervalo Diário'].head(30))
#print(df['Intervalo Diário'].info())

# Real_Calo_Pata_A
limites_real_calo = list(range(0, 101, 5))
limites_real_calo.append(np.inf)  
rotulos_real_calo = [f"{inicio} a {inicio + 4} %" for inicio in limites_real_calo[:-1]]
rotulos_real_calo[-1] = '100 %' 
df['Real_Calo_Pata_A'] = pd.cut(df['Real_Calo_Pata_A'], bins=limites_real_calo, labels=rotulos_real_calo, right=False)


# In[10]:


#Understand numerical columns to define scaling method
numerical_columns = ['Dist. Abatedouro', 'Densidade Abate (Kg/M²)', 'PM', 
                     '% Mort 07', '% Mort 14', '% Mort 21', '% MT', 
                     '% Condenação Parcial', '% Condenação Total', 
                     'Densidade Alojamento', 'Pre-Inicial(kg/ave)', 
                     'Inicial(Kg/ave)', 'Crescimento - I(Kg/ave)', 'Final-II(Kg/ave)']

min_max_values = {}
for column_name in numerical_columns:
    min_value = df[column_name].min()
    max_value = df[column_name].max()
    min_max_values[column_name] = {'min': min_value, 'max': max_value}

for column_name, values in min_max_values.items():
    print(f"Column '{column_name}': Min = {values['min']}, Max = {values['max']}")
    
# Check for missing values in each column
missing_values = df[numerical_columns].isnull().sum()
print("\nMissing values in numerical columns:")
print(missing_values)


# In[11]:


# Scaling numerical variables
scaling_methods = {
    'Dist. Abatedouro': MinMaxScaler(),  # MinMaxScaler chosen to scale between 0 and 1 due to wide range
    'Densidade Abate (Kg/M²)': MinMaxScaler(),  # MinMaxScaler chosen to scale between 0 and 1 due to bounded values
    'PM': StandardScaler(),  # StandardScaler chosen for standardization to center around 0 with a standard deviation of 1
    '% Mort 07': MinMaxScaler(),  # MinMaxScaler chosen to scale between 0 and 1 due to bounded values
    '% Mort 14': MinMaxScaler(),  # MinMaxScaler chosen to scale between 0 and 1 due to bounded values
    '% Mort 21': MinMaxScaler(),  # MinMaxScaler chosen to scale between 0 and 1 due to bounded values
    '% MT': MinMaxScaler(),  # MinMaxScaler chosen to scale between 0 and 1 due to bounded values
    '% Condenação Parcial': MinMaxScaler(),  # MinMaxScaler chosen to scale between 0 and 1 due to bounded values
    '% Condenação Total': MinMaxScaler(),  # MinMaxScaler chosen to scale between 0 and 1 due to bounded values
    'Densidade Alojamento': MinMaxScaler(),  # MinMaxScaler chosen to scale between 0 and 1 due to bounded values
    'Pre-Inicial(kg/ave)': StandardScaler(),  # StandardScaler chosen for standardization to center around 0 with a standard deviation of 1
    'Inicial(Kg/ave)': StandardScaler(),  # StandardScaler chosen for standardization to center around 0 with a standard deviation of 1
    'Crescimento - I(Kg/ave)': StandardScaler(),  # StandardScaler chosen for standardization to center around 0 with a standard deviation of 1
    'Final-II(Kg/ave)': StandardScaler()  # StandardScaler chosen for standardization to center around 0 with a standard deviation of 1
}

scaled_cols = [col + '_scaled' for col in numerical_columns]

for column_name, scaler in scaling_methods.items():
   df[column_name + '_scaled'] = scaler.fit_transform(df[[column_name]])

#print(df[scaled_cols].head())


# In[12]:


#Understand categorical columns to define encoding method
categorical_columns = ['Sexo', 'Linhagem', 'Categoria', 'Tipo Ventilação', 'Área Util', 'Nº Cama', 'Origem Incubatório', 
                       'ID Matriz', 'Peso Alojamento', 'Real_Calo_Pata_A', 'Origem Água', 'Timer Iluminação', 'Tipo Telhado', 
                       'Extensionista', 'Nº Fornecedor', 'Município', 'Intervalo Diário', 'Mês Aloj', 
                       'Mês Abate', 'Estação Aloj', 'Estação Abate', 'Período Aloj', 
                       'Período Abate']

unique_values_counts = {}
for column_name in categorical_columns:
    unique_values_counts[column_name] = df[column_name].nunique()

for column_name, count in unique_values_counts.items():
    print(f"Column '{column_name}' has {count} unique values.")
    
# Check for missing values in each column
missing_values = df[categorical_columns].isnull().sum()
print("\nMissing values in categorical columns:")
print(missing_values)


# In[13]:


# Encoding categorical variables
# Label Encoding
label_encoder = LabelEncoder()
df['Extensionista_encoded'] = label_encoder.fit_transform(df['Extensionista'])
df['Município_encoded'] = label_encoder.fit_transform(df['Município'])
df['Nº Fornecedor_encoded'] = label_encoder.fit_transform(df['Nº Fornecedor'])

#print(df.head())


# In[14]:


# One-hot Encoding
categorical_columns_one_hot = ['Sexo', 'Linhagem', 'Categoria', 'Tipo Ventilação', 'Origem Incubatório', 'Origem Água', 
                       'Tipo Telhado', 'Timer Iluminação']
label_encoders = {}
for col in categorical_columns_one_hot:
    label_encoders[col] = LabelEncoder()
    df[col + '_encoded'] = label_encoders[col].fit_transform(df[col])

#print(df.head())
#print(df[['Sexo', 'Linhagem', 'Categoria', 'Tipo Ventilação', 'Origem Incubatório', 'Origem Água', 'Tipo Telhado','Timer Iluminação']].head(60))
#print(df.groupby('Categoria')['Nº Lote'].count())                                                                                                                            


# In[15]:


# Ordinal Encoding - Replace missing values with 100 and mode
ordinal_cols = ['Área Util', 'Nº Cama', 'ID Matriz', 'Peso Alojamento', 'Real_Calo_Pata_A', 'Intervalo Diário', 'Mês Aloj', 
               'Mês Abate', 'Estação Aloj', 'Estação Abate', 'Período Aloj', 'Período Abate']
ordinal_mapping = [
    ['<=1200 m²', '1201 a 1800 m²', '1801 a 2400 m²', '2401 a 3000 m²', '3001 a 6000 m²'],
    ['1 a 3 camas', '4 a 6 camas', '7 a 9 camas', '10 a 12 camas', '13 a 15 camas', '16 a 18 camas', '19 a 21 camas', 
     '22 a 24 camas', '25 a 27 camas', '28 a 30 camas', '31 a 33 camas', '34 a 36 camas', '37 a 39 camas', '40 a 42 camas', 
     '43 a 45 camas', '46 a 48 camas', '49 a 51 camas', '52 a 54 camas', '55 a 57 camas', '58 a 60 camas', '61 a 63 camas', 
     '64 a 66 camas'],
    ['27 a 31 semanas', '32 a 36 semanas', '37 a 41 semanas', '42 a 46 semanas', '47 a 51 semanas', '52 a 56 semanas', '57 a 61 semanas',
    '62 a 66 semanas'],
    ['0.03 a 0.032 Kg','0.033 a 0.035 Kg', '0.036 a 0.038 Kg', '0.039 a 0.041 Kg', '0.042 a 0.044 Kg', '0.045 a 0.047 Kg',
     '0.048 a 0.05 Kg', '0.051 a 0.053 Kg', '0.054 a 0.056 Kg', '0.057 a 0.059 Kg'],
    ['0 a 4 %', '5 a 9 %', '10 a 14 %', '15 a 19 %', '20 a 24 %', '25 a 29 %', '30 a 34 %', '35 a 39 %', '40 a 44 %', '45 a 49 %', 
     '50 a 54 %', '55 a 59 %', '60 a 64 %', '65 a 69 %', '70 a 74 %', '75 a 79 %', '80 a 84 %', '85 a 89 %', '90 a 94 %', '95 a 99 %',
     '100 %'],
    ['Até 6 dias', '7 a 11 dias', '12 a 16 dias', 'Acima de 17 dias'],
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
    ['Verão', 'Outono', 'Inverno', 'Primavera'],
    ['Verão', 'Outono', 'Inverno', 'Primavera'],
    ['Manhã', 'Tarde', 'Noite'],
    ['Manhã', 'Tarde', 'Noite']
]

# Criando um codificador ordinal
ordinal_encoder = OrdinalEncoder(categories=ordinal_mapping)

# Criando novas colunas com sufixo "_encoded"
encoded_cols = [col + '_encoded' for col in ordinal_cols]
df[encoded_cols] = df[ordinal_cols]  # Copiando as colunas originais para as novas colunas

# Substituindo as informações das colunas originais pelos valores codificados
df[encoded_cols] = ordinal_encoder.fit_transform(df[encoded_cols])

#print(df.head())


# In[16]:


numerical_columns_scaled = ['Dist. Abatedouro_scaled', 'Densidade Abate (Kg/M²)_scaled', 'PM_scaled', 
                     '% Mort 07_scaled', '% Mort 14_scaled', '% Mort 21_scaled', '% MT_scaled', 
                     '% Condenação Parcial_scaled', '% Condenação Total_scaled', 
                     'Densidade Alojamento_scaled', 'Pre-Inicial(kg/ave)_scaled', 
                     'Inicial(Kg/ave)_scaled', 'Crescimento - I(Kg/ave)_scaled', 'Final-II(Kg/ave)_scaled']
print(df[numerical_columns_scaled].head())


# In[17]:


#Run XGBoost Model
#Set numerical and categorical columns
numerical_columns_scaled = ['Dist. Abatedouro_scaled', 'Densidade Abate (Kg/M²)_scaled', 'PM_scaled', 
                     '% Mort 07_scaled', '% Mort 14_scaled', '% Mort 21_scaled', '% MT_scaled', 
                     '% Condenação Parcial_scaled', '% Condenação Total_scaled', 
                     'Densidade Alojamento_scaled', 'Pre-Inicial(kg/ave)_scaled', 
                     'Inicial(Kg/ave)_scaled', 'Crescimento - I(Kg/ave)_scaled', 'Final-II(Kg/ave)_scaled']

categorical_columns_encoded = ['Sexo_encoded', 'Linhagem_encoded', 'Categoria_encoded', 'Tipo Ventilação_encoded', 'Área Util_encoded', 'Nº Cama_encoded', 
                       'Origem Incubatório_encoded', 'ID Matriz_encoded', 'Peso Alojamento_encoded', 'Real_Calo_Pata_A_encoded', 
                       'Origem Água_encoded', 'Timer Iluminação_encoded', 'Tipo Telhado_encoded', 'Extensionista_encoded', 'Nº Fornecedor_encoded', 
                       'Município_encoded', 'Intervalo Diário_encoded', 'Mês Aloj_encoded', 'Mês Abate_encoded', 'Estação Aloj_encoded', 
                       'Estação Abate_encoded', 'Período Aloj_encoded', 'Período Abate_encoded']
y = df['CAAF']

# Concatenating the numeric and categorical columns
numerical_columns_array = df[numerical_columns_scaled].values
categorical_columns_array = df[categorical_columns_encoded].values
X = np.concatenate((numerical_columns_array, categorical_columns_array), axis=1)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating an XGBoost regression model
model = xgb.XGBRegressor()

# Training the model with the training data
model.fit(X_train, y_train)

# Making predictions with the test data
y_pred = model.predict(X_test)

# Calculating and printing the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plotting the results (if needed)
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values (CAAF)')
plt.ylabel('Predicted Values (CAAF)')
plt.title('XGBoost: Actual Values vs. Predicted Values')
plt.show()


# In[20]:


# Calcular o MSE
mse = mean_squared_error(y_test, y_pred)

# Calcular o RMSE
rmse = np.sqrt(mse)

# Calcular o R²
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²):", r2)


# In[22]:


importances = model.feature_importances_

# Criar um DataFrame para visualizar a importância das variáveis
importance_df = pd.DataFrame({'Feature': numerical_columns_scaled + categorical_columns_encoded,
                              'Importance': importances})

# Ordenar o DataFrame pela importância das variáveis
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plotar a importância das variáveis
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# In[23]:


# Calcular a correlação entre cada variável e o indicador de CAAF
correlation_df = pd.DataFrame({'Feature': numerical_columns_scaled + categorical_columns_encoded,
                               'Correlation': [np.corrcoef(df[feature], df['CAAF'])[0, 1] for feature in numerical_columns_scaled + categorical_columns_encoded]})

# Adicionar uma coluna indicando se a relação é diretamente ou inversamente proporcional
correlation_df['Relation'] = ['Directly Proportional' if corr > 0 else 'Inversely Proportional' for corr in correlation_df['Correlation'].values]

print(correlation_df)


# In[25]:


# Calcular a correlação entre cada variável e o indicador de CAAF
correlation_df = pd.DataFrame({'Feature': numerical_columns_scaled + categorical_columns_encoded,
                               'Correlation': [np.corrcoef(df[feature], df['CAAF'])[0, 1] for feature in numerical_columns_scaled + categorical_columns_encoded]})

# Adicionar uma coluna indicando se a relação é diretamente ou inversamente proporcional
correlation_df['Relation'] = ['Directly Proportional' if corr > 0 else 'Inversely Proportional' for corr in correlation_df['Correlation'].values]

# Calcular a importância das variáveis
importances = model.feature_importances_

# Criar um DataFrame para visualizar a importância das variáveis
importance_df = pd.DataFrame({'Feature': numerical_columns_scaled + categorical_columns_encoded,
                              'Importance': importances})

# Ordenar o DataFrame pela importância das variáveis
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plotar a importância das variáveis
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)

# Adicionar informações de relação diretamente ou inversamente proporcional
for i, feature in enumerate(correlation_df['Feature']):
    relation = correlation_df.loc[correlation_df['Feature'] == feature, 'Relation'].values[0]
    plt.text(0.02, i, relation, color='black', va='center')

plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# In[26]:


import seaborn as sns

# Criar o box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Categoria', y='CAAF', data=df)
plt.title('Box Plot of CAAF by Categoria')
plt.xlabel('Categoria')
plt.ylabel('CAAF')
plt.xticks(rotation=45)
plt.show()


# In[27]:


# Convertendo X_test em um DataFrame pandas
X_test_df = pd.DataFrame(X_test, columns=numerical_columns_scaled + categorical_columns_encoded)

# Resetando os índices de df
df_reset_index = df.reset_index(drop=True)

# Adicionando a coluna 'Nº Fornecedor' de df_reset_index a X_test_df
X_test_df['Nº Fornecedor'] = df_reset_index.loc[X_test_df.index, 'Nº Fornecedor'].values

# Criando um DataFrame com as previsões do modelo e os fornecedores correspondentes
predictions_df_linear = pd.DataFrame({'Nº Fornecedor': X_test_df['Nº Fornecedor'],
                                      'Predicted CAAF': y_pred})

# Calculando a média da CAAF prevista para cada fornecedor
mean_caaf_by_supplier_linear = predictions_df_linear.groupby('Nº Fornecedor')['Predicted CAAF'].mean().reset_index()

# Ordenando os fornecedores pela média da CAAF prevista
mean_caaf_by_supplier_linear = mean_caaf_by_supplier_linear.sort_values(by='Predicted CAAF', ascending=False)

# Selecionando os top 10 fornecedores com maior CAAF previsto
top_10_suppliers = mean_caaf_by_supplier_linear.head(10)

# Exibindo a lista dos top 10 fornecedores
print("Top 10 fornecedores com maior CAAF previsto:")
print(top_10_suppliers)

# Convertendo os números de fornecedor de volta para o tipo de dados str
mean_caaf_by_supplier_linear['Nº Fornecedor'] = mean_caaf_by_supplier_linear['Nº Fornecedor'].astype(str)

# Selecionando apenas os top 10 fornecedores com as maiores médias de CAAF prevista
top_10_suppliers = mean_caaf_by_supplier_linear.nlargest(10, 'Predicted CAAF')

# Plotando os top 10 fornecedores com maiores médias de CAAF prevista
plt.figure(figsize=(10, 6))
plt.bar(top_10_suppliers['Nº Fornecedor'], top_10_suppliers['Predicted CAAF'])
plt.xlabel('Nº Fornecedor')
plt.ylabel('Média da CAAF Prevista (Regressão Linear)')
plt.title('Top 10 Fornecedores com Maiores Médias de CAAF Prevista (Regressão Linear)')
plt.xticks(rotation=90)
plt.show()


# In[29]:


from sklearn.model_selection import GridSearchCV

# Definir os hiperparâmetros para ajustar
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Criar o modelo base
base_model = xgb.XGBRegressor()

# Inicializar a grade de busca cruzada
grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Realizar a busca cruzada na grade
grid_search.fit(X_train, y_train)

# Obter os melhores hiperparâmetros encontrados
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Usar o modelo com os melhores hiperparâmetros
best_model = grid_search.best_estimator_

# Fazer previsões com o modelo otimizado
y_pred_tuned = best_model.predict(X_test)

# Calcular e imprimir o MSE do modelo otimizado
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
print("Mean Squared Error (Tuned Model):", mse_tuned)

# Fazendo previsões com os dados de teste usando o modelo ajustado
y_pred_tuned = model.predict(X_test)

# Calculando o R²
r2_tuned = r2_score(y_test, y_pred_tuned)

print("R-squared (Tuned Model):", r2_tuned)


# In[ ]:




