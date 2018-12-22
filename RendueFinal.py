
#on commence par importe tt les bibliothéque dont on a besoin
import os
from glob import glob
import re
import ast
import numpy as np 
import pandas as pd
from PIL import Image, ImageDraw 
from tqdm import tqdm
from dask import bag
import pandas as pd
import numpy as np
import os
import ast
import math
import csv
import cv2

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# On commence par creer un fichier pour mettre tout dedans
dm = pd.read_csv('C:/Users/chaoui/Desktop/LifProjet/RC4/circle.csv')
drr= dm.loc[0:2]

#une fonction qui parcours notre dossier RC4 et renvoie un fichier csv avec 5000 premieres lignes de chaque fichiers
#draw
def ParcourDirectory():
    global drr
    i=0
    for element in os.listdir('C:/Users/chaoui/Desktop/LifProjet/RC4/'):
        dd= pd.read_csv('C:/Users/chaoui/Desktop/LifProjet/RC4/'+element)
        drr=drr.append(dd.loc[0:5000])

ParcourDirectory()

#on transforme le resultat qu'on a eu en un fichier csv
drr.to_csv('C:/Users/chaoui/Desktop/LifProjet/test.csv')

#on stock le fichier resultat dans une variable
dmm=pd.read_csv('C:/Users/chaoui/Desktop/LifProjet/test.csv')
dmm

#on applique la fct ast eval pour traiter les drawing en tant que liste et non une chaine de caractére
dmm['drawing'] = dmm['drawing'].apply(ast.literal_eval)

#fonction pr transformer la liste de vecteur en en couple (x,y)
def zipper(liste):
        return list(map(list, list(zip(*liste))))

#une fonction resres qui appelle cette fonction sur tout un tableau et renvoie en resultat le tableau modifié
def resres(dataFrameDraw):
    res=[]
    length=len(dataFrameDraw)
    for i in range(length):
        ss=dataFrameDraw.iloc[i]
        dd = [zipper(liste) for liste in ss]
        res.append(dd)
    return res

#une fonction qui apelle zipper mais uniquement pour une seul ligne du tableau
def resres1(dataFrame):
    ss=dataFrame
    dd = [zipper(liste) for liste in ss]
    return dd

#une fonction pour cree notre image , prend en parametre la ligne du tableau, puis la parcours et dessine sur une image blache
#des traits entre chaque couple et renvoie le resultat en tant que tableau contenant tout les pixels de l'image graçe à la 
#fonction predefinie open_cv  
def createimage(cordes):
    def normaliseCoords(x,y):
        tmpx = int(round(x*taillex/255))
        tmpy = int(round(y*tailley/255))
        return tmpx, tmpy

    img = np.zeros((taillex, tailley,), np.uint8)
    for corde in cordes : 
        for i in range(len(corde)-1):
            cv2.line(img, normaliseCoords(corde[i][0], corde[i][1]), normaliseCoords(corde[i+1][0], corde[i+1][1]), (255), 1)
    
    return img

#juste un petit affichage d'image
listeV=[]
img = createimage(a[6])
plt.imshow(img, cmap='binary')
plt.title('my picture')
plt.show()

def draw_img(to_draw,a,b,listepoints):
    i=0
    cd=[]
    tt=[]
    a=resres(to_draw['drawing'].iloc[a:b])
    for i in range(len(a)):
        cd.append(tt)
    for i in range(len(a)):
        img = createimage(a[i],cd[i])
        listepoints.append(cd[i])
        plt.imshow(img, cmap='binary')
        plt.title(to_draw['word'][i])
        plt.show() 

#fonction qui prend une ligne la transforme, la convertie et recupere les pixels apres avoir dessiner l'image
def Pixel_Flatten(dataFrame):
        a=resres1(dataFrame)
        listeF=[]
        listeF=createimage(a)
        lsls=np.array(listeF).flatten()
        return lsls

#Fait le meme travail que Pixel Flatten mais sur tout le tableau
def Parcour_Tab_pixel(dataFrame):
    for i in range(len(dataFrame)):
        dataFrame[i]=Pixel_Flatten(dataFrame[i])

#on appelle cette fonction sur notre tableau dmm qui contient toutes les lignes 
Parcour_Tab_pixel(dmm['drawing'])
dmm['drawing']

#dans un premier temps on n'utilise pas de reseau neuronal, on utilise plutot au depart un model
#predefinit avec sklearn (scikitlearn)

#on commence a instancier les models
images = dmm['drawing']
labels = dmm['word']
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

#on commence à mettre nos listes au bon format 
lss=list(train_images)
res=[list(x) for x in lss ]
lss1=list(test_images)
res1=[list(x) for x in lss1 ]

#on passe les resultats au bon format
ress=list(train_labels)
ress1=list(test_labels)

#on appelle la fonction qui trouve le resultat 
clf = svm.SVC()
clf.fit(res,ress)
clf.score(res1,ress1)

#Cela va nous renvoyer la valeur 0.096 ce qui est à peine 10% et donc represente un resultat tres faible
#on a ce meme resultat si on devine les noms des draw au hasard
#et donc on essaye une autre methode avec notre propre cnn



///////////////////////////////////////////cnn propre a nous//////////////////////////////////////////////////



#ici on va dans le train_simplified et on commence par remplacer tout les espaces vides dans les noms par des tirets 
classfiles = os.listdir('../input/train_simplified/')
numstonames = {i: v[:-4].replace(" ", "_") for i, v in enumerate(classfiles)} #ajoute tt les underscore

#declarer quelques constantes, on utilise l'image avec une taille de 32*32 px, avec cette methode
#nous avons un meilleur resulat tout en ayant un temps moins important de calculs
num_classes = 340    #340 valeur max le nombre de fichier dans RC4 
imheight, imwidth = 32, 32  
ims_per_class = 2000  #max?

#ici apres avoir fait nos tests avec la premiere fonction pour dessiner les images 
#la fonction n'etait pas tres memory-friendly et donc on a du redefinire une fonction pour dessiner l'image 
#qui est un peu plus rapide mais aussi ne devore pas la memoire 

def draw_it(strokes):
    image = Image.new("P", (256,256), color=255)
    image_draw = ImageDraw.Draw(image)
    for stroke in ast.literal_eval(strokes):
        for i in range(len(stroke[0])-1):
            image_draw.line([stroke[0][i], 
                             stroke[1][i],
                             stroke[0][i+1], 
                             stroke[1][i+1]],
                            fill=0, width=5)
    #on redimensionne l'image au parametre de taille qu'on a         
    image = image.resize((imheight, imwidth))
    return np.array(image)/255.


#ici c'est une fonction pr definir les train_images et les train_labels quand je l'utilise j'ai un meilleur 
#resultats , c'est une methode de monsieur fracois chollet que j'ai reprise a patire du tuto qu'il a mis en ligne 
#pour le digit reconizer , c'est la personne qui a cree la bibliotheque keras 
#en gros ici il remplace ce que on a fait pr remplire les dataset train_images and train_labels 
train_grand = []
#recupere tt les fichier avec .csv a la fin, et cree les train arrays
class_paths = glob('../input/train_simplified/*.csv')
for i,c in enumerate(tqdm(class_paths[0: num_classes])):
    train = pd.read_csv(c, usecols=['drawing', 'recognized'], nrows=ims_per_class*5//4)
    train = train[train.recognized == True].head(ims_per_class)
    imagebag = bag.from_sequence(train.drawing.values).map(draw_it) 
    trainarray = np.array(imagebag.compute())  
    trainarray = np.reshape(trainarray, (ims_per_class, -1))    
    labelarray = np.full((train.shape[0], 1), i)
    trainarray = np.concatenate((labelarray, trainarray), axis=1)
    train_grand.append(trainarray)
    
train_grand = np.array([train_grand.pop() for i in np.arange(num_classes)]) #on economise de la memoire plus que np.concatenate
train_grand = train_grand.reshape((-1, (imheight*imwidth+1)))

del trainarray
del train
 
#ici on essaye de remplacer la methode du  train_test_split avec une methode un peu plus memory-friendly
#mais aussi pour renvoyer un meilleur resultat 
valfrac = 0.1
cutpt = int(valfrac * train_grand.shape[0])

np.random.shuffle(train_grand)
y_train, X_train = train_grand[cutpt: , 0], train_grand[cutpt: , 1:]
y_val, X_val = train_grand[0:cutpt, 0], train_grand[0:cutpt, 1:] #on recherche tt les case ou regognized est a true

del train_grand

y_train = keras.utils.to_categorical(y_train, num_classes)
X_train = X_train.reshape(X_train.shape[0], imheight, imwidth, 1)
y_val = keras.utils.to_categorical(y_val, num_classes)
X_val = X_val.reshape(X_val.shape[0], imheight, imwidth, 1)

print(y_train.shape, "\n",
      X_train.shape, "\n",
      y_val.shape, "\n",
      X_val.shape)

#la ici on cree notre cnn : reseau neuronal 
model = Sequential()
#on prend des parties de l'image de taille 3par 3 pixel et on applique 32 filtre(matrice) de differente valeur aleatoire a l'interieur 
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(imheight, imwidth, 1)))
#on ne garde que les pixel avec une valeur positive ou bien la plus grande entre 0 et 1
model.add(MaxPooling2D(pool_size=(2, 2)))
#ici on reaplique 64 filtres 
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
#ici on reprends que les 2 plus grandes valeures 
model.add(MaxPooling2D(pool_size=(2, 2)))
#ici on enleve les reseaux neuroneaux qui ont moins de 20% de reconnaisance comme ca on aura des calcul plus rapide
model.add(Dropout(0.2))
#on alligne notre reseaux neuronal
model.add(Flatten())
#affiche juste 680 resultats de l'image avec l'option relu qui permet d'avoir les neurone les plus positive
model.add(Dense(680, activation='relu'))
#on enleve la ou les neurone sont a moins de 50% comme ca les calcul sont encore plus rapide
model.add(Dropout(0.5))
#ici on va encore diminuer le nombre de layer quand veut comme resultats a 340 mais avec cette fois ci softmax 
model.add(Dense(num_classes, activation='softmax'))
model.summary()

#la j'applique mon reseaux neuronal avec des fonction predefinis 
#ici je retire les trois meilleur ressemblance
def top_3_accuracy(x,y): 
    t3 = top_k_categorical_accuracy(x,y, 3)
    return t3
#cette fonction est faite pour avoir des information sur notre data qui sont entrain detre procede
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 
                                   verbose=1, mode='auto', min_delta=0.005, cooldown=5, min_lr=0.0001)
earlystop = EarlyStopping(monitor='val_top_3_accuracy', mode='max', patience=5) 
callbacks = [reduceLROnPlat, earlystop]

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', top_3_accuracy])

model.fit(x=X_train, y=y_train,
          batch_size = 32,
          epochs = 22,
          validation_data = (X_val, y_val),
          callbacks = callbacks,
          verbose = 1)
# ici j'obtient
#p_3_accuracy: 0.6057
#ce qui est largement beaucoup mieux que le test avec le scikitlearn 