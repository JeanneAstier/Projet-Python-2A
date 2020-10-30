#!/usr/bin/env python
# coding: utf-8

# # Analyse de le base de données training

# Ce Notebook a pour but d'analyser notre base de données initiale afin d'expliquer nos choix d'échantillonnage.

# ## 0 - Introduction et analyse grossière de la base 

# In[233]:


# Nous aurons besoin des modules :
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Pour cela nous avons besoin de la base de données "Training_DataFrame" créée dans le fichier "" a partir des images au format DICOM récupérées sur le site : https://challenge2020.isic-archive.com/ .

# In[49]:


path_Training_DataFrame='C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/Training_DataFrame.csv'
df_train=pd.read_csv(path_Training_DataFrame)
df_train.head()


# In[50]:


df_train.shape


# On remarque que cette base est volumineuse ( 33 126 lignes représentant les 33 126 images de mélanomes (le nombre de patient peut être moindre car un patient peut avoir plusieurs ménalomes). Par manque de puissance de calculs, nous ne sommes pas en mesure de traiter une telle base de données. C'est pourquoi nous souhaitons l'échantillonner avant de créer un algorithme de machin learning permettant d'identifier les mélanomes bénins et malins.
# Comment selectionner notre échantillon ? 

# In[51]:


df_train.isnull().sum()


# Cette commande permet d'identifier le nombre de valeurs manques. Il n'y en a ici que très peu : seulement 65 patients sur 33 126 ont omis de renseigner leur genre (valeur 'nan') et 49 patients ont renseigné 'X'. 
# Les valeurs non-exploitables de la variable 'Patient_sex' représentent moins de 0,3% de notre échantillon, nous ne ferons donc pas d'étude sur les non réponses, et retirons ces patients de notre base de données. 

# In[55]:


indexNames = df_train[ df_train['patient_sex']=='X'].index
df_train.drop(indexNames , inplace=True)
df_train.dropna(inplace=True)


# ## 1 - Analyses univariées

# ### a) L'age

# In[62]:


df_train['patient_age'].hist( facecolor='b', alpha=0.5)
plt.title('Age des patients')
plt.show()
print('Moyenne :', df_train['patient_age'].mean(), '\n', 'Ecart type:',  df_train['patient_age'].std())


# ### b) Le sexe

# In[87]:


df_train.groupby('patient_sex')['patient_id'].nunique()


# In[144]:


df_train["patient_sex"].value_counts().plot(kind='pie' , autopct='%1.1f%%')
plt.xlabel(' Parité Hommes/Femmes', fontsize=15)


# La parité Homme/Femme est respectée. 

# ### c) La partie du corps

# In[216]:


df_train['body_part'].value_counts()


# In[217]:


plt.figure(figsize=(10,6))
sns.barplot(x=df_train['body_part'].unique(), y=df_train['body_part'].value_counts(), palette="Reds_r")
plt.xlabel('\nParties du corps', fontsize=15, color='#c0392b')
plt.ylabel("Nombre de grains de beauté\n", fontsize=15, color='#c0392b')
plt.title("Localisation des grains de beauté\n", fontsize=18, color='#e74c3c')
plt.xticks(rotation= 45)
plt.tight_layout()


# ### c) Melanomes bénins et malins

# In[64]:


plt.pie([sum(df_train['target']==1),sum(df_train['target']==0)], labels = ['Malins','Benins'],colors = ['lightcoral','lightskyblue'],autopct='%1.1f%%')
plt.title("Taux de melanomes benins et malins dans l'échantillon")
plt.show()


# ## 2 - Analyses bivariées

# ### a) Les mélanomes et l'age

# In[158]:


plt.figure(figsize=(15,8))
ax = sns.kdeplot(df_train["patient_age"][df_train.target == 1], color="darkturquoise", shade=True)
sns.kdeplot(df_train["patient_age"][df_train.target == 0], color="lightcoral", shade=True)
plt.legend(['Melanome malin', 'Melanome bénin'])
plt.title("Fonctions de densité : répartition de l'age pour les mélanomes malins et bénins")
ax.set(xlabel='Age')
plt.xlim(-10,90)
plt.show()


# La distribution des mélanomes malins est décalée sur la droite par rapport à celle des bénins. Cela signifie que les personnes agées sont plus touchées par les mélanomes malins que les personnes jeunes. Les articles scientifiques confirment cette hypothèse. 

# ### b) Melanomes et sexe

# In[190]:


(df_train.groupby('patient_sex')['target'].sum() / df_train.groupby('patient_sex')['patient_id'].count() *100)


# In[193]:


sns.barplot('patient_sex', 'target', data=df_train, color="aquamarine")
plt.show()


# D'après cet échantillon les hommes sont plus touchés par les mélanomes malins que les femmes. 
# 1,37% des femmes de l'échantillon sont porteuses d'un mélanome bénins contre 2,13% des hommes.
# Les articles scientifiques ne confirment pas cette hypothèse. 
# 
# 
# FAIRE UN TEST ?? 

# ### c) Melanomes et partie du corps

# In[204]:


(df_train.groupby('body_part')['target'].sum() / df_train.groupby('body_part')['patient_id'].count() *100)


# In[236]:


sns.barplot(df_train['body_part'],df_train['target'], palette='Blues_d')


# Le taux de mélanomes malins est plus élevé sur les parties du corps : 
#      - Tête et cou
#      - orales et génitales 
#      - les membres supérieurs
#      
# Selon les articles scientifiques, les zones les plus exposées aux mélanomes malins sont les parties les plus exposées au soleil.
# Il est donc raisonnable d'identifier la tête, le cou et les membres supérieurs dans les parties du corps les plus touchées par les mélanomes malins. 
# Dans ce sens, cela est étonnant d'identifier les zones orales et génitales comme zones à risque. Cependant, dans notre échantillon, la catégorie "Oral/Genital" est la moins représentée des parties du corps; il n'y a que 124 images. Ce qui est très faible par rapport à notre base totale. L'échantillon est probablement peu représentatif de la population total pour cette partie du corps.
# 
# 
# J'AIME PAS TROP CE GRAPHIQUE CAR ON VOIT RIEN SUR L'AXE DES ABSCISSES. Ce serait mieux en horizontal mais j'ai pas réussi 
# 
# FAIRE UN TEST ? 

# ## 3 - Regression linéaire logistique

# Nous voulons ici expliquer la variable 'target' en fonction des variables 'patient_age', 'patient_sex', 'body_part'

# In[ ]:




