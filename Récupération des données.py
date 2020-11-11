"""
Ce module permet de traiter les donnees DICOM : 
    - extraire les metadonnees
    - transformer les images .dcm en JPG
    - analyser la table
@author: louis
"""
#TEST 

# Pour cela nous aurons besoin des packages suivant : 
import pydicom as dicom # module pour utiliser les donnees au format DICOM
import pydicom.data
from pydicom.pixel_data_handlers.util import convert_color_space 
import matplotlib.pyplot as plt
import os # Permet d'interagir avec le systeme d'exploitation
import cv2 # Pour le traitement des image 
import PIL
import pandas as pd 
import csv
import numpy as np

class Dataframe :
    def __init__(self):
        self.path_dicom = "C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/train" # Chemin vers les images.dcm
        self.path_Training_GroundTruth ="C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/ISIC_2020_Training_GroundTruth.csv"
        self.path_jpg = "C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/test jpg" # Chemin vers les images.jpg
        self.path_jpg_RGB= "C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/test jpg" # Chemin vers les images.jpg aux couleurs RGB
        self.columns =["image_id", "patient_age", "patient_sex", "body_part"] # informations qu'on va récupérer dans les données DICOM
        
    def from_DICOM_to_DF(self):
        '''
        Cette fonction cree et remplit un dataframe a partir de quelques-unes des informations presentes
        dans les metadonnees DICOM. 
        Le format DICOM peut etre traite par Python, mais nous sommes beaucoup plus a l'aise avec un dataframe 
        et nous n'avons pas besoin de toutes les informations contenues dans les metadonnees DICOM. 
        
        Cette fonction :
            - prend en parametre "self" permettant d'aller chercher les informations dans la partie init         
            - renvoie : le dataframe comprenant les informations extraites  
            
        Exemple : 
            w = Dataframe()
            w.from_DICOM_to_DF()
        '''
        global df
        df = pd.DataFrame(columns=self.columns)
        for file in os.listdir(self.path_dicom) : 
            filename = pydicom.data.data_manager.get_files(self.path_dicom, file)[0]
            ds = pydicom.dcmread(filename)
            values = [ds.PatientID[1:-1], int(ds.PatientAge[:3]), ds.PatientSex, ds.BodyPartExamined] 
            df_new_row = pd.DataFrame(data = [values], columns = self.columns)
            df = pd.concat([df, df_new_row], ignore_index = True)
        df2 = pd.read_csv(self.path_Training_GroundTruth) 
        # on va chercher l'information target (bénin ou malin) dans un document à part (elle n'est pas comprise dans les metadonnées DICOM)
        # ainsi que le patient_id
        df = pd.merge(df,df2[['image_name','target', "patient_id"]], left_on='image_id', right_on='image_name')
        return df

    def convert_DICOM_to_JPG (self) :
        '''
        Cette fonction permet de convertir un dossier dont les images sont sous
        le format DICOM (Digital imaging and communications in medicine, c'est un 
        standard international pour la gestion informatique des donnees issues de
        l'imagerie medicale) en format JPG
        Python est capable de traiter les donnees au format .dcm, cependant, pour
        ouvrir plus facilement les images individuellement sans devoir installer
        de nouveaux logiciels sur nos ordinateurs, il est plus pratique pour nous
        de convertir ces images sous un format plus commun.  
    
        Cette fonction :
            - prend en parametres "self" permettant d'aller chercher les informations 
            dans la partie init
            - renvoie : Le dossier ou les images.jpg sont enregistrees  
            
        Exemple : 
            w = Dataframe()
            w.convert_DICOM_to_JPG()
        '''
        images_path = os.listdir(self.path_dicom) # renvoie le nom des fichiers dans le dossier
        for n, image in enumerate(images_path):
            ds = pydicom.dcmread(os.path.join(self.path_dicom, image)) # lire un fichier dicom a partir d'un chemin de dossier et un nom de fichier
            pixel_array_numpy = ds.pixel_array # donnees sur les pixels
            image = image.replace('.dcm', '.jpg')
            cv2.imwrite(os.path.join(self.path_jpg, image), pixel_array_numpy) # enregistrer l'image
        print ('Le dossier est pret ! ')
    
 
    def convert_to_JPG_RGB (self) : 
        ''''
        Cette fonction permet de convertir les images d'un dossier au format DICOM  
        (Digital imaging and communications in medicine, c'est un 
        standard international pour la gestion informatique des donnees issues de
        l'imagerie medicale) et au format de couleurs "YBR_FULL_422", en images au 
        format JPG et au format de couleurs "RGB" (ce qui permet d'avoir un rendu 
        plus "naturel" de l'image).
       
        Cette fonction :
            - prend en parametres "self" permettant d'aller chercher les informations 
            dans la partie init
            - renvoie : Le dossier ou les images.jpg et rgb sont enregistrees  
    
        Exemple : 
            w = Dataframe()
            w.convert_to_JPG_RGB()
        '''    
        images_path = os.listdir(self.path_dicom) 
        for n, image in enumerate(images_path) : 
            ds = pydicom.dcmread(os.path.join(self.path_dicom, image))
            convert = convert_color_space(ds.pixel_array, 'YBR_FULL_422', 'RGB')
            image = image.replace('.dcm', '.jpg') 
            cv2.imwrite(os.path.join(self.path_jpg_RGB, image), cv2.cvtColor(convert, cv2.COLOR_RGB2BGR))
        print ('Le dossier est pret !')


w = Dataframe()
w.from_DICOM_to_DF()
df.to_csv('C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/Training_DataFrame.csv', index=False)

