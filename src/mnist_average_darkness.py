"""
mnist_average_darkness
~~~~~~~~~~~~~~~~~~~~~~

A naive classifier for recognizing handwritten digits from the MNIST
data set.  The program classifies digits based on how dark they are
--- the idea is that digits like "1" tend to be less dark than digits
like "8", simply because the latter has a more complex shape.  When
shown an image the classifier returns whichever digit in the training
data had the closest average darkness.

The program works in two steps: first it trains the classifier, and
then it applies the classifier to the MNIST test data to see how many
digits are correctly classified.

Needless to say, this isn't a very good way of recognizing handwritten
digits!  Still, it's useful to show what sort of performance we get
from naive ideas."""

#### Libraries
# Standard library
from collections import defaultdict

# My libraries
import mnist_loader

def main():

    # On charge les datasets
    print("\n>> 1. DATASETS <<\n")
    training_data, validation_data, test_data = mnist_loader.load_data()

    # On compte les moyennes d'obscurité des images pour chaque nombre a prédire pour le dataset d'apprentissage
    print("\n>> 2. \"TRAINING\" <<\n")
    print("\"Training\"...")
    avgs = avg_darknesses(training_data)
    print("\"Trained\"...")

    # On compte le nombre de chiffres correctement prédits pour le dataset de test
    print("\n>> 3. \"TESTING\" <<\n")
    print("Testing...")
    num_correct = sum(int(guess_digit(image, avgs) == digit) for image, digit in zip(test_data[0], test_data[1]))
    print("Tested...")

    n_test_data = len(test_data[1])
    print("Baseline classifier using average darkness of image.")
    print("{0} of {1} ({2}%) values are correct.".format(num_correct, n_test_data, num_correct * 100.0 / n_test_data))
    print("Tested...")

def avg_darknesses(training_data):
    """ Return a defaultdict whose keys are the digits 0 through 9.
    For each digit we compute a value which is the average darkness of
    training images containing that digit.  The darkness for any
    particular image is just the sum of the darknesses for each pixel."""

    digit_counts = defaultdict(int)
    darknesses = defaultdict(float)

    # Pour chaque couple, image / nombre à prédire
    for image, digit in zip(training_data[0], training_data[1]):

        # On compte le nombre d'images pour chaque chiffre
        digit_counts[digit] += 1

        # On compte l'obscurité de l'image pour chaque chiffre
        darknesses[digit] += sum(image)
    
    avgs = defaultdict(float)

    # Pour chaque chiffre
    for digit, n in digit_counts.items():

        # On associe chaque chiffre à une moyenne d'obscurité d'image
        avgs[digit] = darknesses[digit] / n

    return avgs

def guess_digit(image, avgs):
    """Return the digit whose average darkness in the training data is
    closest to the darkness of ``image``.  Note that ``avgs`` is
    assumed to be a defaultdict whose keys are 0...9, and whose values
    are the corresponding average darknesses across the training data."""

    # On calcule l'obscurité de l'image
    darkness = sum(image)

    # On calcule les différences entre l'obscurité de l'image et la moyenne d'obscurité de chaque chiffre
    distances = {k: abs(v-darkness) for k, v in avgs.items()}

    # On retourne le chiffre correspondant à une obscurité la plus proche de celle calculée
    return min(distances, key=distances.get)

if __name__ == "__main__":
    main()
