"""expand_mnist.py
~~~~~~~~~~~~~~~~~~

Take the 50,000 MNIST training images, and create an expanded set of
250,000 images, by displacing each training image up, down, left and
right, by one pixel.  Save the resulting file to
../data/mnist_expanded.pkl.gz.

Note that this program is memory intensive, and may not run on small
systems.

"""

from __future__ import print_function

#### Libraries

# Standard library
import _pickle as cPickle
import gzip
import os.path
import random
from tqdm import tqdm

# Third-party libraries
import numpy as np

print("Expanding the MNIST training set")

if os.path.exists("../data/mnist_expanded.pkl.gz"):
    
    # Si le fichier existe déjà, on n'a pas besoin de refaire le traitement 
    print("The expanded training set already exists.  Exiting.")
else:

    # On charge le dataset original en mode lecture
    f = gzip.open("../data/mnist.pkl.gz", 'rb')

    # On charge les données
    training_data, validation_data, test_data = cPickle.load(f)

    #  On ferme le fichier
    f.close()


    # On stocke dans la liste le nouveau dataset dir'apprentissage généré à partir du dataset dir'apprentissage initial
    expanded_training_pairs = []

    # Pour chaque image et nombre à prédire du dataset dir'apprentissage initial
    for x, y in tqdm(zip(training_data[0], training_data[1])):

        # On conserve le couple (x, y) initial dans le nouveau dataset d'apprentissage
        expanded_training_pairs.append((x, y))

        # On transforme la mode de stockage de l'image : d'un tableau unidirectionnel de taille (28 * 28) en une matrice de taille (28, 28)
        # On représente le tableau en 2D, car cela va nous permettre de faire plus facilement les calculs de déplacements pixels par pixels
        image = np.reshape(x, (-1, 28))
        
        # Pour chaque série de déplacements
        for dir, axis, index in [ 
            (1, 0, 0),   # Déplacement de 1 pixel vers la haut
            (-1, 0, 27), # Déplacement de 1 pixel vers la bas
            (1,  1, 0),  # Déplacement de 1 pixel vers le droite
            (-1, 1, 27)  # Déplacement de 1 pixel vers la gauche
        ]:

            # On déplace tous les pixels d'un nombre "dir" de pixels dans la direction "axis"
            # On notera que les pixels sortant de la fin de la matrice (dernière ligne ou dernière
            # colonne si dir > 0) reviennent au début de la matrice (première ligne ou première
            # colonne si dir > 0)
            new_img = np.roll(image, dir, axis)

            # On remplace les pixels étant "sortis de la fin de la matrice et dpnc remis au début" par du noir (0)
            if axis == 0: 
                # Cas d'un déplacement vertical
                new_img[index, :] = np.zeros(28)
            else: 
                # Cas d'un déplacement horizontal
                new_img[:, index] = np.zeros(28)

            # On retransforme la matrice (28, 28) en tableau unidirectionnel de taille (28 * 28)
            new_img = np.reshape(new_img, 784)

            # On ajoute le nouveau couple (x, y) initial dans le nouveau dataset d'apprentissage
            expanded_training_pairs.append((new_img, y))

    # On mélange de manière aléatoire le tableau
    random.shuffle(expanded_training_pairs)

    # On restore le dataset d'apprentissage dans le format initial chargé en mémoire
    expanded_training_data = [list(dir) for dir in zip(*expanded_training_pairs)]


    # On sauvegarde les données dans un nouveau fichier gz, pour éviter de refaire le traitement à chaque lancement du programme
    print("Saving expanded data. This may take a few minutes.")

    # On ouvre le fichier en mode écriture
    f = gzip.open("../data/mnist_expanded.pkl.gz", "w")

    # On sauvegarde les datasets dans le fichier
    cPickle.dump((expanded_training_data, validation_data, test_data), f)
    
    # On ferme le fichier
    f.close()
