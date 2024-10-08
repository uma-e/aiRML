import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

def load_data(file_path):
    data = pd.read_csv(file_path, delimiter=',', header=None)
    try:
        data.iloc[:, -1] = data.iloc[:, -1].astype(int)
    except ValueError as e:
        print(f"Error converting labels to integers: {e}")
    return data


def normalize(data):
    normDF = data.copy()
    for column in normDF.columns[:-1]:  
        min = normDF[column].min()
        max = normDF[column].max()
        normDF[column] = (normDF[column] - min) / (max - min)
    return normDF

#split data into training and testing sets
def splitData(X, y):

    #shuffling for randomness
    data = np.hstack((X, y.reshape(-1, 1)))
    np.random.shuffle(data)

    shuffledX = data[:, :-1] #seperating shuffled data again
    shuffledY = data[:, -1]

    splitIndex = int(len(X) * 0.75)
    xTrain = shuffledX[:splitIndex]
    xTest = shuffledX[splitIndex:]
    yTrain = shuffledY[:splitIndex]
    yTest = shuffledY[splitIndex:]

    return xTrain, xTest, yTrain, yTest

def hardUnipolarActivation(x):
    return np.where(x >= 0, 1, 0)

def softUnipolarActivation(x, gain=1):
    return 1 / (1 + np.exp(-gain * x))

#perceptron training
def train(X, y, activationFunc, alpha, errorThreshold, maxIter=5000):
    weights = np.random.uniform(-0.5, 0.5, X.shape[1])
    bias = np.random.uniform(-0.5, 0.5)
    totalError = float('inf')
    iteration = 0

    while totalError > errorThreshold and iteration < maxIter:
        totalError = 0
        for i in range(len(X)):
            netInput = np.dot(X[i], weights) + bias
            prediction = activationFunc(netInput)
            error = y[i] - prediction
            totalError += error**2

            weights += alpha * error * X[i]
            bias += alpha * error

        iteration += 1
    print("Total error: ", totalError)
    return weights, bias

def predict(X, weights, bias, activationFunc):
    netInput = np.dot(X, weights) + bias
    return np.array([activationFunc(x) for x in netInput])

# Plot decision boundary and data points
def plotting(X, y, weights, bias, activationFunc, title):
    import matplotlib.pyplot as plt
import numpy as np

# Function to plot data and decision boundary
def plot_decision_boundary(X, y, weights, bias, activationFunc, title):
    plt.figure(figsize=(8, 6))

    # Plot data points: small car (0) as red and big car (1) as blue
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Small Car (0)', alpha=0.7)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Big Car (1)', alpha=0.7)

    # Create decision boundary
    x_values = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 200)
    y_values = -(weights[0] * x_values + bias) / weights[1]  # Derived from w1*x1 + w2*x2 + bias = 0

    # Plot the decision boundary
    plt.plot(x_values, y_values, color='green', label='Decision Boundary', linewidth=2)

    # Labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)

    # Show plot
    plt.show()


# Evaluate the perceptron model
from sklearn.metrics import confusion_matrix, accuracy_score

# Evaluation function to calculate confusion matrices and accuracy
def evalutation(xTrain, yTrain, xTest, yTest, weights, bias, activationFunc):
    # Train set predictions
    yPredTrain = activationFunc(np.dot(xTrain, weights) + bias)

    # Test set predictions
    yPredTest = activationFunc(np.dot(xTest, weights) + bias)

    # Apply threshold to convert continuous values to binary predictions (0 or 1)
    yPredTrainBin = np.where(yPredTrain >= 0.5, 1, 0)
    yPredTestBin = np.where(yPredTest >= 0.5, 1, 0)

    # Confusion matrices
    conf_matrixTrain = confusion_matrix(yTrain, yPredTrainBin)
    conf_matrixTest = confusion_matrix(yTest, yPredTestBin)

    # Accuracy scores
    accuracyTrain = accuracy_score(yTrain, yPredTrainBin)
    accuracyTest = accuracy_score(yTest, yPredTestBin)

    return conf_matrixTrain, conf_matrixTest, accuracyTrain, accuracyTest


def perceptron(file_path, activation, errorThreshold, alpha, gain=1):
    data = load_data(file_path)
    data = normalize(data)
    
    # Split features and labels
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values.astype(int)   # Ensure labels are integers

    # 75% training, 25% testing
    xTrain, xTest, yTrain, yTest = splitData(X, y)

    # Train perceptron
    weights, bias = train(xTrain, yTrain, activation, errorThreshold=errorThreshold, alpha=alpha)

    # Plot decision boundary
    plotting(xTrain, yTrain, weights, bias, activation, 'Training Data Decision Boundary')
    plotting(xTest, yTest, weights, bias, activation, 'Testing Data Decision Boundary')

    # Evaluate model
    conf_matrixTrain, conf_matrixTest, accuracyTrain, accuracyTest = evalutation(xTrain, yTrain, xTest, yTest, weights, bias, activation)

    # Print results
    print("Training Confusion Matrix:\n", conf_matrixTrain)
    print("Testing Confusion Matrix:\n", conf_matrixTest)
    print("Training Accuracy:", accuracyTrain)
    print("Testing Accuracy:", accuracyTest)


# Run for both hard and soft unipolar activation functions
file_path = 'groupB.txt'

print("Hard Unipolar Activation Function")
perceptron(file_path, hardUnipolarActivation, 40, alpha=0.05)

print("Soft Unipolar Activation Function")
perceptron(file_path, softUnipolarActivation, 40, alpha=0.05, gain=1)
