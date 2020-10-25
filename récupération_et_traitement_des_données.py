#!/usr/bin/env python
# coding: utf-8


import pydicom
import pydicom.data
import matplotlib.pyplot as plt
import os
import pandas as pd


## ****RECUPERATION DE LA LISTE DES FICHIERS DICOM****
## (path_dicom = chemin du dossier en local où sont stockés les fichiers DICOM, à adapter)

path_dicom = r"C:\Users\jeann\OneDrive\Documents\scolaire\ENSAE\2A\S1\python\projet\DICOM" 

files = os.listdir(path_dicom)
files


## ****EXPLORATION  DU FORMAT DICOM****
## (je ne sais pas s'il faut mettre des trucs comme ça dans le rendu final du projet ou pas)
## (oui je raconte un peu ma vie dans les commentaires)

file = files[0]

filename = pydicom.data.data_manager.get_files(path_dicom, file)[0]
ds = pydicom.dcmread(filename)

print(ds) #affiche toutes les métadonnées associés à l'image sous forme de tags

print(ds.PatientName) #affiche la valeur du tag

print(ds[0x10, 0x20]) #affiche le tag complet qui a ces coordonnées

plt.imshow(ds.pixel_array) #affiche l'image en tant que telle




## ****CREATION ET REMPLISSAGE D'UN DATAFRAME A PARTIR DES DONNEES DCIM****

def from_DICOM_to_DF(dicom_list) : 
    '''
    Cette fonction crée et remplit un dataframe à partir de quelques-unes des informations présentes
    dans les métadonnées DICOM. 
    Le format DICOM peut être traité par Python, mais nous sommes beaucoup plus à l'aise avec un dataframe 
    et nous n'avons pas besoin de toutes les informations contenues dans les métadonnées DICOM. 
    
    Cette fonction :
        - prend en paramètres : dicom_list, la liste des fichiers DICOM dont on veut extraire les informations         
        - renvoie : le dataframe comprenant les informations extraites  
    
    Exemple : 
        dicom_list = os.listdir("C:\\Users\\jeann\\OneDrive\\Documents\\scolaire\\ENSAE\\2A\\S1\\python\\projet\\DICOM")
    
    '''
    
    # création du dataframe, à adapter selon les informations qui nous intéressent parmi les métadonnées DICOM :
    df = pd.DataFrame(columns = ["patient_id", "patient_name", "patient_age", "patient_sex", "body_part", "image"]) 
    
    # remplissage du dataframe avec les données des images choisies, sans les images pour l'instant
    # (à mettre à jour si on ajoute ou retire des informations du dataframe)
    for file in dicom_list : 
        filename = pydicom.data.data_manager.get_files(path_dicom, file)[0]
        ds = pydicom.dcmread(filename)
        values = [ds.PatientID, ds.PatientName, ds.PatientAge, ds.PatientSex, ds.BodyPartExamined, 0]
        df_new_row = pd.DataFrame(data = [values], columns = df.columns )
        df = pd.concat([df, df_new_row], ignore_index = True)
    return df



from_DICOM_to_DF(files)




## ****TRAITEMENT DES IMAGES****

from pydicom.pixel_data_handlers.util import convert_color_space 
import cv2
import pydicom as dicom

def convert_DICOM_to_JPG (Dossier_dicom, Dossier_jpg): 
    '''
    Cette fonction permet de convertir un dossier dont les images sont sous
    le format DICOM (Digital imaging and communications in medicine, c'est un 
    standard international pour la gestion informatique des données issues de
    l'imagerie médicale) en format JPG.
    Python est capable de traiter les données au format .dcm, cependant, pour
    ouvrir plus facilement les images individuellement sans devoir installer
    de nouveaux logiciels sur nos ordinateurs, il est plus pratique pour nous
    de convertir ces images sous un format plus commun. 
    
    Cette fonction :
        - prend en paramétres :
            - Dossier_dicom : le chemin du dossier à traiter
            - Dossier_jpg : le chemin du dossier vers lequel les images.jpg sont enregistrées
        - renvoie : Le dossier où les images.jpg sont enregistrées  
    
    Exemple : 
        Dossier_dicom = "C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/train"
        Dossier_jpg = "C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/train png"

    '''
    images_path = os.listdir(Dossier_dicom) # renvoie le nom des fichiers dans le dossier
    for n, image in enumerate(images_path):
        ds = dicom.dcmread(os.path.join(Dossier_dicom, image)) # lire un fichier dicom a partir d'un chemin de dossier et un nom de fichier
        pixel_array_numpy = ds.pixel_array # données sur les pixels
        image = image.replace('.dcm', '.jpg')
        cv2.imwrite(os.path.join(Dossier_jpg, image), pixel_array_numpy) # enregistrer l'image
    print ('Le dossier est pret ! ')

def convert_to_JPG_RGB (Dossier_DICOM, Dossier_JPG_RGB) : 
             
    ''''
    Cette fonction permet de convertir les images d'un dossier au format DICOM  
    (Digital imaging and communications in medicine, c'est un 
    standard international pour la gestion informatique des données issues de
    l'imagerie médicale) et au format de couleurs "YBR_FULL_422", en images au 
    format JPG et au format de couleurs "RGB" (ce qui permet d'avoir un rendu 
    plus "naturel" de l'image).
       
    Cette fonction :
        - prend en paramètres :
            - Dossier_dicom : le chemin du dossier à traiter
            - Dossier_jpg_rgb : le chemin du dossier vers lequel les images.jpg et rgbsont enregistrées
        - renvoie : Le dossier où les images.jpg et rgb sont enregistrées  
    
    Exemple : 
        Dossier_dicom = "C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/train"
        Dossier_jpg = "C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/train png"

    '''    
    images_path = os.listdir(Dossier_DICOM) 
    for n, image in enumerate(images_path) : 
        ds = dicom.dcmread(os.path.join(Dossier_DICOM, image))
        convert = convert_color_space(ds.pixel_array, 'YBR_FULL_422', 'RGB')
        image = image.replace('.dcm', '.jpg') 
        cv2.imwrite(os.path.join(Dossier_JPG_RGB, image), cv2.cvtColor(convert, cv2.COLOR_RGB2BGR))
    print ('Le dossier est pret !')
    



path_jpg = r"C:\Users\jeann\OneDrive\Documents\scolaire\ENSAE\2A\S1\python\projet\JPG"
path_jpg_rgb = r"C:\Users\jeann\OneDrive\Documents\scolaire\ENSAE\2A\S1\python\projet\JPG_RGB"

## on convertit les images originales dicom en jpg
convert_DICOM_to_JPG(path_dicom, path_jpg)
    
# on convertit les images originales dicom en jpg rgb
convert_to_JPG_RGB(path_dicom, path_jpg) 




## ****AJOUT DES IMAGES AU DATAFRAME****

def add_image_names_to_DF(df, path_JPG) : 
    '''
    Cette fonction peremt de compléter la colonne "image" d'un dataframe avec le nom de l'image
    correspondant à chaque ligne. Nous nous contenterons d'ajouter le nom de cette image pour pouvoir 
    la retrouver par la suite dans son dossier, et pas d'ajouter l'image en tant que telle au dataframe. 
     
    Cette fonction :
        - prend en paramètres : 
            - df : le dataframe à compléter, qui doit avoir une colonne "image" 
            - path_JPG : le chemin du dossier comprenant les images que l'on veut mettre dans le dataframe ; 
            ces images doivent être dans le même ordre que les fichiers DICOM le sont dans leur fichier
        - renvoie : le dataframe complété avec le nom des images  
    
    Exemple : 
        df = from_DICOM_to_DF(dicom_list)
        path_JPG = "C:\\Users\\jeann\\OneDrive\\Documents\\scolaire\\ENSAE\\2A\\S1\\python\\projet\\JPG"
    
    '''

    pictures_list =  os.listdir(path_JPG)
    
    n = len(df.patient_id)
    for i in range(n) : 
        df.image = pictures_list[i]
    return df


# In[271]:


add_image_names_to_DF(df, path_jpg)


# In[ ]:




