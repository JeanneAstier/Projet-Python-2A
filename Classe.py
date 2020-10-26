"""
Ce module permet de traiter les données DICOM : 
    - extraire les métadonnées
    - transformer les images .dcm en JPG
    - analyser la table
@author: louis
"""

# Pour cela nous aurons besoin des packages suivant : 
import pydicom # module pour utiliser les données au format DICOM
import pydicom.data
import matplotlib.pyplot as plt
import os # Permet d'interagir avec le systeme d'exploitation
import cv2 # Pour le traitement des image 
import PIL
import pandas as pd 
import csv
from pydicom.pixel_data_handlers.util import convert_color_space 
import pydicom as dicom

class Dataframe :
    def __init__(self):
        self.path_dicom = "C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/test" # Chemin vers les images.dcm
        self.path_jpg = "C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/test jpg" # Chemin vers les images.jpg
        self.path_jpg_RGB= "C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/test jpg" # Chemin vers les images.jpg au couleur RGB
        self.columns =["patient_id", "patient_name", "patient_age", "patient_sex", "body_part", "image"] 
        self.ds = pydicom.dcmread(filename)
        self.values =[ds.PatientID, ds.PatientName, ds.PatientAge, ds.PatientSex, ds.BodyPartExamined, 0]
        
        
    def from_DICOM_to_DF(self):
        '''
        Cette fonction crée et remplit un dataframe à partir de quelques-unes des informations présentes
        dans les métadonnées DICOM. 
        Le format DICOM peut être traité par Python, mais nous sommes beaucoup plus à l'aise avec un dataframe 
        et nous n'avons pas besoin de toutes les informations contenues dans les métadonnées DICOM. 
        
        Cette fonction :
            - prend en paramètres "self" permettant d'aller chercher les informations dans la partie init         
            - renvoie : le dataframe comprenant les informations extraites  
            
        Exemple : 
            dicom_list = os.listdir("C:\\Users\\jeann\\OneDrive\\Documents\\scolaire\\ENSAE\\2A\\S1\\python\\projet\\DICOM")
    
        '''
        global df
        df=pd.DataFrame(columns=self.columns)
        for file in os.listdir(self.path_dicom) : 
            filename = pydicom.data.data_manager.get_files(self.path_dicom, file)[0]
            df_new_row = pd.DataFrame(data = [self.values], columns = self.columns )
            df= pd.concat([df, df_new_row], ignore_index = True)
        return df
    
    def convert_DICOM_to_JPG (self) :
        '''
        Cette fonction permet de convertir un dossier dont les images sont sous
        le format DICOM (Digital imaging and communications in medicine, c'est un 
        standard international pour la gestion informatique des données issues de
        l'imagerie médicale) en format JPG
        Python est capable de traiter les données au format .dcm, cependant, pour
        ouvrir plus facilement les images individuellement sans devoir installer
        de nouveaux logiciels sur nos ordinateurs, il est plus pratique pour nous
        de convertir ces images sous un format plus commun.  
    
        Cette fonction :
            - prend en paramétres "self" permettant d'aller chercher les informations 
            dans la partie init
            - renvoie : Le dossier où les images.jpg sont enregistrées  
            
        Exemple : 
            w= Dataframe()
            w.convert_DICOM_to_JPG()
        '''
        images_path = os.listdir(self.path_dicom) # renvoie le nom des fichiers dans le dossier
        for n, image in enumerate(images_path):
            ds = pydicom.dcmread(os.path.join(self.path_dicom, image)) # lire un fichier dicom a partir d'un chemin de dossier et un nom de fichier
            pixel_array_numpy = ds.pixel_array # données sur les pixels
            image = image.replace('.dcm', '.jpg')
            cv2.imwrite(os.path.join(self.path_jpg, image), pixel_array_numpy) # enregistrer l'image
        print ('Le dossier est pret ! ')
    
 
    def convert_to_JPG_RGB (self) : 
        ''''
        Cette fonction permet de convertir les images d'un dossier au format DICOM  
        (Digital imaging and communications in medicine, c'est un 
        standard international pour la gestion informatique des données issues de
        l'imagerie médicale) et au format de couleurs "YBR_FULL_422", en images au 
        format JPG et au format de couleurs "RGB" (ce qui permet d'avoir un rendu 
        plus "naturel" de l'image).
       
        Cette fonction :
            - prend en paramètres "self" permettant d'aller chercher les informations 
            dans la partie init
            - renvoie : Le dossier où les images.jpg et rgb sont enregistrées  
    
        Exemple : 
            w= Dataframe()
            w.convert_to_JPG_RGB()
        '''    
        images_path = os.listdir(self.path_dicom) 
        for n, image in enumerate(images_path) : 
            ds = pydicom.dcmread(os.path.join(self.path_dicom, image))
            convert = convert_color_space(ds.pixel_array, 'YBR_FULL_422', 'RGB')
            image = image.replace('.dcm', '.jpg') 
            cv2.imwrite(os.path.join(self.path_jpg_RGB, image), cv2.cvtColor(convert, cv2.COLOR_RGB2BGR))
        print ('Le dossier est pret !')