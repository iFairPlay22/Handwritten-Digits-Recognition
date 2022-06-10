"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
from plot import SmartPlot



class Network(object):

    ID = 1

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""

        # Nombre de layers
        self.num_layers = len(sizes)

        # Taille de chaque layer
        self.sizes = sizes

        # Liste des biais
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # Liste des poids
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        # Id de l'IA
        self.id = Network.ID
        Network.ID += 1

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""

        # Pour chaque pour de biais et de poids
        for b, w in zip(self.biases, self.weights):

            # On calcule la sortie de chaque neurone
            a = sigmoid(np.dot(w, a)+b)

        # On renvoie la sortie de la derniere couche
        return a

    def SGD(self, training_data : list, testing_data : list, epochs : int, batch_size : int, eta : float , plot : SmartPlot, color : str):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        # On récupère le nombre total de données à charger pour l'entrainement et le test
        n_learn = len(training_data)
        n_test = len(testing_data)
        percentage = 0

        # Pour chaque epoch (chaque itération de l'ensemble du jeu de données)
        for j in range(1, epochs + 1):

            # On mélange le jeu de données d'entrainement
            random.shuffle(training_data)

            # On segmente le jeu de données d'entrainement en sous listes de taille batch_size
            batches = [ training_data[k:k+batch_size] for k in range(0, n_learn, batch_size) ]

            # Pour chaque sous liste de données d'entrainement
            for batch in batches:

                # On entraine l'IA (mise à jour des poids et des biais) vis à vis de la sous liste de données
                self.update_batch(batch, eta)

            # On calcule le nombre et le pourcentage de bonnes prédictions sur l'ensemble du jeu de données de tests
            goodPredictions = self.evaluate(testing_data)
            percentage = 100.0 * goodPredictions / n_test

            # On ajoute le pourcentage de prédictions sur le plot
            plot.addPoint("N{0} > layers={1}, eta={2}".format(self.id, self.sizes, eta), color, percentage)

            if j % 5 == 0 or j == epochs:
                # On affiche le score de prédiction à partir du jeu de données de test (sans mettre à jour les poids et les biais car on teste uniquement)
                print("Epoch {0} / {1}: {2} / {3} ({4}%)".format(j, epochs, goodPredictions, n_test, percentage))

        return percentage

    def update_batch(self, batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        
        # On initialise la liste des nouveaux poids et biais à 0 avec des dimentions identiques
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Pour chaque batch (x = données en entrée, y = données attendues en sortie)
        for x, y in batch:

            # On calcule le gradient de la fonction de cout pour chaque neurone de chaque couche
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            
            # On mets à jour les valeurs des poids et des biais en fonction du gradient
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # On mets à jour les valeurs des poids et des biais en fonction du gradient et des valeurs précédantes
        self.weights = [w-(eta/len(batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(batch))*nb for b, nb in zip(self.biases, nabla_b)]

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
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
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

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""

        # On fait une dérivée partielle à partir des données de sorties et du résultat attendu
        return (output_activations-y)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        # On construit une liste de la forme [ (pred1, expe1), (pred1, expe2), ... ] pour chaque données du jeu de test
        # pred correspond à la catégorie prédite (index correspondant au score maximal du feed forward)
        # expe correspond à la catégorie attendue (défini au préalable dans le jeu de tests)
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        # On compte le nombre de fois où on a prédit la bonne catégorie
        return sum(int(x == y) for (x, y) in test_results)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""

    # On renvoie la fonction sigmoide de z
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""

    # On renvoie la dérivée de la fonction sigmoide de z
    return sigmoid(z)*(1-sigmoid(z))
