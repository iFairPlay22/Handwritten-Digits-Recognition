"""
mnist_svm
~~~~~~~~~

A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier."""

#### Libraries
# My libraries
import mnist_loader 

# Third-party libraries
from sklearn import svm

def svm_baseline():

    # On charge les datasets
    print("\n>> 1. DATASETS <<\n")
    training_data, validation_data, test_data = mnist_loader.load_data()

    # On entraine une IA
    print("\n>> 2. TRAINING <<\n")
    print("Training SVM...")

    # On utilise un classifieur SVC (Support vector machine = Séparateurs à vaste marge), afin de faire de la classification
    clf = svm.SVC()
    
    # On entraine le classifieur à partir du dataset d'apprentissage
    clf.fit(training_data[0], training_data[1])
    print("SVM trained...")

    # On teste l'efficacité de l'apprentissage en confrontant le classifieur au dataset de test
    print("\n>> 3. TESTING <<\n")
    print("Testing SVM...")

    # On récupère le nombre prédit par le classifieur pour chaque image du dataset d'apprentissage
    predictions = [int(a) for a in clf.predict(test_data[0])]

    # On commpte le nombre de prédictions justes, en comparant la prédiction avec le résultat attendu
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))

    # On affiche les performances du classifieur
    n_test_data = len(test_data[1])
    print("Baseline classifier using an SVM.")
    print("{0} of {1} ({2}%) values are correct.".format(num_correct, n_test_data, num_correct * 100.0 / n_test_data))
    print("SVM tested...")

if __name__ == "__main__":
    svm_baseline()
    
