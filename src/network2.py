"""network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np
from plot import SmartPlot


#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)

#### Main Network class
class Network(object):
    
    ID = 1

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        # Nombre de layers
        self.num_layers = len(sizes)
        
        # Taille de chaque layer
        self.sizes = sizes

        # Liste des biais et du poids
        self.default_weight_initializer()

        # Fonctions de cout (crosss entropy)
        self.cost=cost

        # Id de l'IA
        self.id = Network.ID
        Network.ID += 1

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        # Liste des biais
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]

        # Liste des poids
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    # Ancienne méthode (cf network.py)
    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        # Liste des biais
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]

        # Liste des poids
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""

        # Pour chaque pour de biais et de poids
        for b, w in zip(self.biases, self.weights):

            # On calcule la sortie de chaque neurone
            a = sigmoid(np.dot(w, a)+b)
        
        # On renvoie la sortie de la derniere couche
        return a

    def SGD(self, training_data : list, evaluation_data : list, epochs : int, batch_size : int, eta : float, lmbda  : float, accPlot : SmartPlot, costPlot : SmartPlot, color : str,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        # On récupère le nombre total de données à charger pour l'entrainement et le test / l'évaluation
        n_training_data = len(training_data)
        n_evaluation_data = len(evaluation_data)

        # On initialise les listes qui contiendront les données de cout et de précision pour le dataset 
        # d'entrainement et de test / d'évaluation (pour chaque epoch)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        # Pour chaque epoch
        for j in range(1, epochs + 1):

            # On mélange le jeu de données d'entrainement
            random.shuffle(training_data)

            # On segmente le jeu de données d'entrainement en sous listes de taille batch_size
            batches = [ training_data[k:k+batch_size] for k in range(0, n_training_data, batch_size) ]

            # Pour chaque sous liste de données d'entrainement
            for mini_batch in batches:

                # On entraine l'IA (mise à jour des poids et des biais) vis à vis de la sous liste de données
                self.update_batch(mini_batch, eta, lmbda, len(training_data))

            # On calcule le cout sur le jeu de données d'entrainement
            if monitor_training_cost:
                tr_cost = self.total_cost(training_data, lmbda, convert=False)
                training_cost.append(tr_cost)
                costPlot.addPoint("Network {}".format(self.id), color, tr_cost)

            # On calcule la précision sur le jeu de données d'entrainement
            if monitor_training_accuracy:
                tr_accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(tr_accuracy)
                accPlot.addPoint("Network {}".format(self.id), color, tr_accuracy * 100 / n_training_data)

            # On calcule le cout sur le jeu de données de test / évaluation
            if monitor_evaluation_cost:
                ev_cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(ev_cost)
                costPlot.addPoint("Network {}".format(self.id), color, ev_cost)

            # On calcule la précision sur le jeu de test / évaluation
            if monitor_evaluation_accuracy:
                ev_accuracy = self.accuracy(evaluation_data, convert=False)
                evaluation_accuracy.append(ev_accuracy)
                accPlot.addPoint("Network {}".format(self.id), color, ev_accuracy * 100 / n_evaluation_data)

            if j % 5 == 0 or j == epochs:
                print("Epoch %s training complete" % j)

                if monitor_training_cost:
                    print("Cost on training data: {:.4f}".format(tr_cost))

                # On calcule la précision sur le jeu de données d'entrainement
                if monitor_training_accuracy:
                    print("Accuracy on training data: {} / {} ({:.2f}%)".format(tr_accuracy, n_training_data, tr_accuracy * 100 / n_training_data))

                # On calcule le cout sur le jeu de données de test / évaluation
                if monitor_evaluation_cost:
                    print("Cost on evaluation data: {:.4f}".format(ev_cost))

                # On calcule la précision sur le jeu de données de test / évaluation
                if monitor_evaluation_accuracy:
                    print("Accuracy on evaluation data: {} / {} ({:.2f}%)".format(ev_accuracy, n_evaluation_data, ev_accuracy * 100 / n_evaluation_data))
            
                print()

        # On renvoie les listes de cout et de précision pour le dataset d'entrainement et de test / d'évaluation
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_batch(self, mini_batch, eta, lmbda, n_training_data):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n_training_data`` is the total size of the training data set.

        """
        # On initialise la liste des nouveaux poids et biais à 0 avec des dimentions identiques
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Pour chaque batch (x = données en entrée, y = données attendues en sortie)
        for x, y in mini_batch:

            # On calcule le gradient de la fonction de cout pour chaque neurone de chaque couche
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            # On mets à jour les valeurs des poids et des biais en fonction du gradient
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # On mets à jour les valeurs des poids et des biais en fonction du gradient, de lmbda et des valeurs précédantes
        self.weights = [(1-eta*(lmbda/n_training_data))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        # On initialise la liste des nouveaux poids et biais à 0 avec des dimentions identiques
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # On effectue la phase de feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):

            z = np.dot(w, activation)+b
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        # On effectue la phase de backward
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        # On retourne le couple représentatif du gradient
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        # Pour chaque couple (x, y) du dataset, où x est l'image en entrée et y la donnée attendue 
        # Attention: y est une liste de booleans pour le dataset d'apprentissage où 1 corresponds 
        # à l'index du nombre à prédire. Dans les autres cas (validation ou test), y est directement
        # le nombre entier à prédire

        # On compte le nombre de bonnes prédictions
        if convert:
            # Dans le cas de données d'apprentissage, on a besoin de calculer l'indice corresondant au score maximal pour récupérer la prédiction
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]

        # On calcule le nombre de bonnes prédictions
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        # Le cout total est la somme des couts pour chaque image
        cost = 0.0

        # Pour chaque couple (x, y) du dataset, où x est l'image en entrée et y la donnée attendue 
        # Attention: y est une liste de booleans pour le dataset d'apprentissage où 1 corresponds 
        # à l'index du nombre à prédire. Dans les autres cas (validation ou test), y est directement
        # le nombre entier à prédire
        for x, y in data:

            # On effectue la phase de feedforward
            a = self.feedforward(x)

            # Si on doit convertir les données (cas du dataset de test / validation), on transforme
            # le nombre en liste de booléens (via la fonction vectorized_result())
            if convert: y = vectorized_result(y)

            # On ajoute le cout de la fonction de cout pour la donnée courante
            cost += self.cost.fn(a, y) / len(data)

        # On ajoute le coût de la regularisation
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)

        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""

        # On enregistre les layers, les poids, les biais et la fonction de cout dans un dictionnaire
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": str(self.cost.__name__)
        }

        # On ouvre le fichier en mode écriture
        f = open(filename, "w")

        # On écrit le dictionnaire dans le fichier
        json.dump(data, f)

        # On ferme le fichier
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """

    # On ouvre le fichier en mode lecture
    f = open(filename, "r")

    # On lit le contenu du fichier (et donc le dictionnaire sauvegardé via la fonction .save())
    data = json.load(f)

    # On ferme le fichier
    f.close()

    # On récupère la fonction de cout
    cost = getattr(sys.modules[__name__], data["cost"])

    # On crée un nouveau réseau avec les données récupérées
    net = Network(data["sizes"], cost=cost)

    # On récupère les poids
    net.weights = [np.array(w) for w in data["weights"]]

    # On récupère les biais
    net.biases = [np.array(b) for b in data["biases"]]

    # On retourne le réseau
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """

    # On transforme la manière dont les nombres à prédire sont stockées en mémoire : on passe d'une valeur d'entier à une liste de booléens ou un seul indice vaut 1
    # Ex: le nombre 1 devient [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ] et le 9 devient [ 0, 0, 0, 0, 0, 0, 0, 0, 9, 0 ]

    # On créée la liste [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    e = np.zeros((10, 1))

    # On mets un 1 sur la valeur associée
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""

    # On renvoie la fonction sigmoid de z
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""

    # On renvoie la dérivée de la fonction sigmoid de z
    return sigmoid(z)*(1-sigmoid(z))
