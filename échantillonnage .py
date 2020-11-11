#!/usr/bin/env python
# coding: utf-8

# téléchargement préalable des modules requis

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import shutil, os



# # 1. Construction d'un échantillon de la base train

# On importe le dataframe complet de la base train, composé des plus de 33 000 images

df_train = pd.read_csv(r"C:\Users\jeann\OneDrive\Documents\scolaire\ENSAE\2A\S1\python\projet\base_complete.csv")


# Calcul du taux de mélanomes malins au sein de cette base :

t = sum(df_train['target']==1) / len(df_train.index)
print("taux de malignité des mélanomes :", t*100, "%")


# Comme nous l'avions déjà vu, le taux de malignité des mélanomes est très faible (1,76%). 
# 
# Aussi, dans un premier temps, pour faciliter la construction de premiers modèles, nous construisons une base réduite avec un taux de malignité bien supérieur (50%). Nous réduisons ainsi le nombre total d'images, ce qui nous permettra aussi de construire des modèles plus économes en mémoire vive pour l'instant.

# Pour ce faire, on commence par définir une fonction d'échantillonnage simple :

np.random.seed

def simple_sampling(df, n) : 
    '''
    Cette fonction cree un échantillon simple de taille n à partir d'un dataframe. 
    Les lignes sélectionnées dans l'échantillon sont tirées aléatoirement, de manière équiprobable. 
        
        Cette fonction :
            - prend en parametres :
                - df : le dataframe à partir duquel on souhaite construire l'échantillon
                -  n : la taille de l'échantillon         
            - renvoie : le dataframe échantilloné, de n lignes  
            
        Exemple : 
            df = df_train
            n = 100
            => renvoie un dataframe de 100 lignes sélectionnées aléatoirement dans df_train
    '''
    rows = np.random.choice(df.index.values, n, replace = False)
    df_sample = df[df.index.isin(rows)]
    return df_sample   


# On sépare ensuite la base initiale en une partie contenant tous les mélanomes malins, une autre contenant tous les mélanomes bénins

df_train_malin = df_train[df_train["target"] == 1]
df_train_benin = df_train[df_train["target"] == 0]


# Puis on échantillonne chacune de ces sous-bases, avant de les réunir dans l'échantillon final

sample_malin = simple_sampling(df_train_malin, 200)
sample_benin = simple_sampling(df_train_benin, 200)


sample_train = sample_malin.append(sample_benin)

sample_train

#Enregistrement de notre dataframe au format csv :
sample_train.to_csv('C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/Images echantillon training/Echantillon_DataFrame.csv', index=False)


# # 2. Analyse de l'échantillon
#
# On analyse maintenant rapidement la base échantillon, pour comparer sa structure à celle de la base originale. On s'attend à une structure déformée, dans la mesure où on a considérablement sur-représenté les mélanomes malins. 

# Vérifions les informations que l'on connaît déjà par construction :

sample_train.shape


# ## 1- Analyses univariées

# ### a) âge

df_train['patient_age'].hist( facecolor='b', alpha=0.5)
plt.title('Age des patients - base complète')
plt.show()
print('Moyenne :', df_train['patient_age'].mean(), '\n', 'Ecart type:',  df_train['patient_age'].std())

sample_train['patient_age'].hist( facecolor='b', alpha=0.5)
plt.title('Age des patients - échantillon')
plt.show()
print('Moyenne :', sample_train['patient_age'].mean(), '\n', 'Ecart type:',  sample_train['patient_age'].std())


# Contrairement à nos attentes, la distribution des âges ne semble pas trop affectée par notre échantillonnage. 
# 
# L'âge moyen est toute fois supérieur de presque 4 ans, ce qui était prévisible : nous avons sur-représenté les mélanomes malins dans notre échantillon, et les mélanomes malins sont portées dans les individus plus âgés de la moyenne. 

# ### b) sexe 

sample_train.groupby('patient_sex')['patient_id'].nunique()

df_train["patient_sex"].value_counts().plot(kind='pie' , autopct='%1.1f%%')
plt.xlabel(' Parité Hommes/Femmes - base complète', fontsize=15)
plt.show()

sample_train["patient_sex"].value_counts().plot(kind='pie' , autopct='%1.1f%%')
plt.xlabel(' Parité Hommes/Femmes - échantillon', fontsize=15)
plt.show()


# Notre échantillon sur-représente donc les hommes par rapport à la base originale : là-aussi prévisible, dans la mesure où les hommes sont sur-représentés parmis les porteurs de mélanomes malins. 

# ### c) parties du corps

sample_train['body_part'].value_counts()

plt.figure(figsize=(10,6))
sns.barplot(x=df_train['body_part'].unique(), y=df_train['body_part'].value_counts(), palette="Reds_r")
plt.xlabel('\nParties du corps', fontsize=15, color='#c0392b')
plt.ylabel("Nombre de grains de beauté\n", fontsize=15, color='#c0392b')
plt.title("Localisation des grains de beauté\n base complète\n", fontsize=18, color='#e74c3c')
plt.xticks(rotation= 45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x=sample_train['body_part'].unique(), y=sample_train['body_part'].value_counts(), palette="Reds_r")
plt.xlabel('\nParties du corps', fontsize=15, color='#c0392b')
plt.ylabel("Nombre de grains de beauté\n", fontsize=15, color='#c0392b')
plt.title("Localisation des grains de beauté\n échantillon\n", fontsize=18, color='#e74c3c')
plt.xticks(rotation= 45)
plt.tight_layout()


# Pas de différence notable dans la répartition des localisations des mélanomes. 

# ### d) mélanomes bénins et malins

# Comme nous le savons par construction, les mélanomes malins sont considérablement sur-représentés dans notre échantillon.

plt.pie([sum(df_train['target']==1),sum(df_train['target']==0)], labels = ['Malins','Benins'],colors = ['lightcoral','lightskyblue'],autopct='%1.1f%%')
plt.title("Taux de melanomes benins et malins dans la base complète")
plt.show()

plt.pie([sum(sample_train['target']==1),sum(sample_train['target']==0)], labels = ['Malins','Benins'],colors = ['lightcoral','lightskyblue'],autopct='%1.1f%%')
plt.title("Taux de melanomes benins et malins dans l'échantillon")
plt.show()


# ## 2- Analyses bivariées

# ### a) mélanomes et âge

plt.figure(figsize=(15,8))
ax = sns.kdeplot(df_train["patient_age"][df_train.target == 1], color="darkturquoise", shade=True)
sns.kdeplot(df_train["patient_age"][df_train.target == 0], color="lightcoral", shade=True)
plt.legend(['Melanome malin', 'Melanome bénin'])
plt.title("Fonctions de densité : répartition de l'age pour les mélanomes malins et bénins - base complète")
ax.set(xlabel='Age')
plt.xlim(-10,90)
plt.show()

plt.figure(figsize=(15,8))
ax = sns.kdeplot(sample_train["patient_age"][sample_train.target == 1], color="darkturquoise", shade=True)
sns.kdeplot(sample_train["patient_age"][sample_train.target == 0], color="lightcoral", shade=True)
plt.legend(['Melanome malin', 'Melanome bénin'])
plt.title("Fonctions de densité : répartition de l'age pour les mélanomes malins et bénins - échantillon")
ax.set(xlabel='Age')
plt.xlim(-10,90)
plt.show()


# La distribution des mélanomes bénins dans notre échantillon est lissée par rapport à la base complète. Sinon, les distributions semblent relativement similaires. 

# ### b) mélanomes et sexe

(sample_train.groupby('patient_sex')['target'].sum() / sample_train.groupby('patient_sex')['patient_id'].count() *100)

sns.barplot('patient_sex', 'target', data=df_train, color="aquamarine")
plt.title("répartition des mélanomes par genre - base complète")
plt.show()

sns.barplot('patient_sex', 'target', data=sample_train, color="aquamarine")
plt.title("répartition des mélanomes par genre - échantillon")
plt.show()


# On remarque que le taux de malignité augmente pour chaque sexe - cohérent, car le taux de malignité général a augmenté. 
# Les hommes restent plus touchés par les mélanomes malins que les femmes, même si la différence entre hommes et femmes se réduit légèrement.

# ### c) mélanomes et parties du coprs

(sample_train.groupby('body_part')['target'].sum() / sample_train.groupby('body_part')['patient_id'].count() *100)

sns.barplot(df_train['target'], df_train['body_part'],palette='Blues_d', orient='h',  order=["HEAD/NECK", "ORAL/GENITAL","UPPER EXTREMITY","SKIN", "TORSO", "LOWER EXTREMITY", "PALMS/SOLES" ])
plt.title("répartition des mélanomes selon les parties du corps - base complète \n")
plt.show()

sns.barplot(sample_train['target'], sample_train['body_part'],palette='Blues_d', orient='h',  order=["HEAD/NECK", "ORAL/GENITAL","UPPER EXTREMITY","SKIN", "TORSO", "LOWER EXTREMITY", "PALMS/SOLES" ])
plt.title("répartition des mélanomes selon les parties du corps - échantillon \n")
plt.show()


# La distribution des mélanomes malins sur les parties du corps semble perturbée par l'échantillonnage : les mélanomes sur les parties orales / génitales sont par exemple beaucoup plus malins dans l'échantillon que dans la base complète. 
# 
# Nous ne voyons pas d'explication particulière à ce phénomène, si ce n'est pas sélection au hasard qui pertubre la distribution. 

# ## 3- Création d'un dossier regroupant les images de notre echantillon
sample_train.to_csv('C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/Images echantillon training/Dataframe_Echantillon.csv')

def dossier_images (path_dicom_complete , path_dicom_sample):
    """
    Cette fonction permet de transferer les images selectionnées dans l'échantillon vers un dossier 
    Sample dicom.
    Attention : Il faut préalablement créer les dossier "Sample dicom" aux emplacements souhaités
    
    Elle prend en parametre :
        - path_image_dicom: chemin vers la base de donnée compléte des images au format dicom
        - lien vers le dossier Sample dicom 
        
    Exemple : 
        path_dicom_complete='C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/train'
        path_dicom_sample = 'C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/Images echantillon training/Sample dicom'
        dossier_images (path_dicom_complete , path_dicom_sample)
    """
    for file in sample_train['image_id']:
        shutil.copy(path_dicom_complete + '/' + file +'.dcm', path_dicom_sample + '/' + file +'.dcm')

#path_dicom_complete='C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/train'
#path_dicom_sample = 'C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/Images echantillon training/Sample dicom'
# dossier_images (path_dicom_complete , path_dicom_sample)

