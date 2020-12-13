# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:59:28 2020

@author: louis
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from keras.models import load_model
from keras.preprocessing.image import (load_img ,img_to_array)

class_names = ['benin', 'malin']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}


def complexite_couleurs (image): 
    '''
    Cette fonction prend en paramétre une image. Elle affiche un 
    graphique représentant les différents canaux de couleurs en fonction
    que cette image soit en couleur ou en nuances de gris. 
    Elle montre ainsi la différence de complexité entre les deux formats.
    
    Elle prend en parametre : 
        - image : le chemin vers une image 
        
    Elle retourne : 
        - Un graphe représentant les canaux de couleurs 
        - Un graphe représentant le canal de nuance de gris
    '''
    
    img1 = cv2.imread(image,cv2.IMREAD_COLOR)
    color = ('b','g','r')
    plt.subplot(2,1,2)
    for i,col in enumerate(color):
        histr = cv2.calcHist([img1],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
    print ( "Trois canaux représentant les espaces de couleurs Rouge, Vert et Bleu")
    plt.show()
    img2= cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    roihist = cv2.calcHist([img2],[0], None, [256], [ 0, 256] )
    plt.subplot(2,1,1)
    xs=np.linspace(0,255,256)
    plt.plot(xs,roihist,color='k')
    print( ' Un seul canal pour les images en nuances de gris')
    
def display_examples(class_names, images, labels):
    '''
    Cette fonction permet de visualiser un échantillon d'images de la base "images" 
    accompagnées de leur label ('bénin' ou 'malin'). 
    
    Elle prend en paramétre : 
        - class_names : classe selon laquelle nous répartissons les images
        - images : base de données 
        - labels : les labels associés à la base "images"
    
    Exemple : 
        display_examples(class_names, train_images, train_labels)
    '''
   
    fig = plt.figure(figsize=(15,10))
    fig.suptitle("Exemples d'images de la base train", fontsize=20)
    for i in range(15):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i])
        plt.xlabel(class_names[labels[i].astype(np.integer)])
    plt.show()
    
    

def Predire_une_image(img_path, img_name, modele , path_Diagnostic = None):
    
    '''
    Cette fonction permet de prédire si l'image placée en pararemetre appartient à la classe des mélanomes bénins ou malins.
    
    Elle prend en parametre :
        - img_path : le chemin vers l'image  (ex : 'C:/Users/Bureau/ISIC_4960802.jpg')
        - img_name : le nom de l'image (ex : 'ISIC_4960802' )
        - modele : le modele choisi pour prédire (ex: 'model.h5')
        - path_Diagnostic : chemin vers fichier comprenant le diagnostic associé à l'image. Ce fichier peut ne pas exister, c'est pourquoi il prend en valeur par défaut 'None'
        
    Elle renvoie : 
        - A quelle classe appartient l'image sélectionnée ainsi que la probabilité d'appartenir à cette classe 
        
    Exemple :
        --> Predire_une_image('../input/test1000/ISIC_4960802.jpg',ISIC_4960802,'model.h5')
        --> Cette image appartient à la classe malin avec probabilité de 99.96951818466187 %
        
    '''
    df_diag= pd.read_csv(path_Diagnostic) 
    
    model = load_model(modele)
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

    img = load_img(img_path, target_size=(150, 150))
    img_tensor = img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    plt.imshow(img_tensor[0])
    plt.show()
    
    classe = model.predict_classes(img_tensor) # Prédit la classe 0 ou 1 
    proba = model.predict_proba(img_tensor)
    
    print('Cette image appartient à la classe ', list(class_names_label.keys())[list(class_names_label.values()).index(classe)], 
          'avec une probabilité de' ,round((100*proba[0][classe[0]]),2),'%' )
    
    if True in df_diag['image_name'].str.contains(img_name):
        
        print('Selon le réel pronostic médical, ce grain de beauté est de type ', list(class_names_label.keys())[list(class_names_label.values()).index(classe)],'.')