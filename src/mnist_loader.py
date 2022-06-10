"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import _pickle as cPickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    
    print("Loading datasets...")

    # On ouvre le fichier contenant le dataset d'images MNIST en mode lecture
    f = gzip.open('../data/mnist.pkl.gz', 'rb')

    # On utilise cPickle afin de segmenter le dataset en 3 partie : 
    # training_data   : couple (x, y) 
    #   - Où x est une liste de taille 50 000, comprenant pour chaque entrée un numpy ndarray de taille 28 * 28 = 784, oû chaque entrée corresponds à un pixel ;
    #   - Où y est une liste de taille 50 000, comprenant pour chaque entrée le nombre correspondant à l'image (donc la prédiction à faire) ;
    # validation_data : pareil, avec une taille de 10 000. 
    # test_data       : pareil, avec une taille de 10 000.
    training_data, validation_data, test_data = cPickle.load(f, encoding="latin1")

    # On ferme le fichier
    f.close()

    print("Datasets loaded...")

    # On retourne les valeurs
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""

    # On charge le jeu de données d'entrainement, de validation et de test
    tr_d, va_d, te_d = load_data()

    print("Formatting datasets...")

    # > Dataset d'entrainement

    # On transforme la manière dont les images sont stockées en mémoire : on passe d'une dimention de (784) à (784, 1), cad une liste de taille 1 pour chaque pixel
    # Ex : la liste [ 1, 2, 3, ..., 784 ] devient [ [1], [2], [3], ..., [784] ]
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]

    # On transforme la manière dont les nombres à prédire sont stockées en mémoire : on passe d'une valeur d'entier à une liste de booléens ou un seul indice vaut 1
    # Ex: le nombre 1 devient [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ] et le 9 devient [ 0, 0, 0, 0, 0, 0, 0, 0, 9, 0 ]
    training_results = [vectorized_result(y) for y in tr_d[1]]

    # On transforme la manière dont le jeu d'entraînement est stocké : on passe à une liste de 50 000 entrées de la forme [ [ti1, tr1], [ti2, tr2], ..., [trn, trn] ] 
    # tel que tin corresponds à training_inputs[n] et trn à training_results[n] et n = 50 000. En clair, on obtient une liste de taille 50 000, contenant à chaque 
    # indice un couple (x, y) où x est l'image en entrée de dimention (784, 1), et où y est la représentation du nombre à prédire sous la forme d'un tableau de booléens.
    training_data = list(zip(training_inputs, training_results))

    # > Dataset de validation

    # On transforme la manière dont les images sont stockées en mémoire : on passe d'une dimention de (784) à (784, 1), cad une liste de taille 1 pour chaque pixel
    # Ex : la liste [ 1, 2, 3, ..., 784 ] devient [ [1], [2], [3], ..., [784] ]
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]

    # On transforme la manière dont le jeu d'entraînement est stocké : on passe à une liste de 10 000 entrées de la forme [ [ti1, tr1], [ti2, tr2], ..., [trn, trn] ] 
    # tel que tin corresponds à validation_inputs[n] et trn à va_d[1][n] et n = 10 000. En clair, on obtient une liste de taille 10 000, contenant à chaque indice un 
    # couple (x, y) où x est l'image en entrée de dimention (784, 1), et où y est la valeur du nombre à prédire (sous la forme d'un entier).
    validation_data = list(zip(validation_inputs, va_d[1]))

    # > Dataset de test

    # On transforme la manière dont les images sont stockées en mémoire : on passe d'une dimention de (784) à (784, 1), cad une liste de taille 1 pour chaque pixel
    # Ex : la liste [ 1, 2, 3, ..., 784 ] devient [ [1], [2], [3], ..., [784] ]
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]

    # On transforme la manière dont le jeu d'entraînement est stocké : on passe à une liste de 10 000 entrées de la forme [ [ti1, tr1], [ti2, tr2], ..., [trn, trn] ] 
    # tel que tin corresponds à test_inputs[n] et trn à te_d[1][n] et n = 10 000. En clair, on obtient une liste de taille 10 000, contenant à chaque indice un 
    # couple (x, y) où x est l'image en entrée de dimention (784, 1), et où y est la valeur du nombre à prédire (sous la forme d'un entier).
    test_data = list(zip(test_inputs, te_d[1]))

    # On affiche le nombre de données chargées
    n_training_data, n_validation_data, n_test_data = len(training_data), len(validation_data), len(test_data)
    print("Datasets formatted : {0} images for training / {1} images for validation / {2} images for test.".format(n_training_data, n_validation_data, n_test_data))

    # On retourne les valeurs
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""

    # On créée la liste [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    e = np.zeros((10, 1))

    # On mets un 1 sur la valeur associée
    e[j] = 1.0

    return e