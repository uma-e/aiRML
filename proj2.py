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


def plotting(X, y, weights, bias, activationFunc, title):

    plt.figure(figsize=(8, 6))

    #plot data points: small car (0) as red and big car (1) as blue
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Small Car (0)', alpha=0.7)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Big Car (1)', alpha=0.7)

    #create decision boundary
    x_values = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 200)
    y_values = -(weights[0] * x_values + bias) / weights[1]  #derived from w1*x1 + w2*x2 + bias = 0

    #decision boundary
    plt.plot(x_values, y_values, color='green', label='Decision Boundary', linewidth=2)

    plt.xlabel('cost in USD')
    plt.ylabel('Weight in pounds (lbs)')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)

    plt.show()


# Evaluate the perceptron model
from sklearn.metrics import confusion_matrix, accuracy_score

# Evaluation function to calculate confusion matrices and accuracy
def evalutation(xTrain, yTrain, xTest, yTest, weights, bias, activationFunc):
    
    yPredTrain = activationFunc(np.dot(xTrain, weights) + bias)
    yPredTest = activationFunc(np.dot(xTest, weights) + bias)

    
    yPredTrainBin = np.where(yPredTrain >= 0.5, 1, 0)
    yPredTestBin = np.where(yPredTest >= 0.5, 1, 0)

    #confusion matrices
    matrixTrain = confusion_matrix(yTrain, yPredTrainBin)
    matrixTest = confusion_matrix(yTest, yPredTestBin)

    #accuracy scores
    accTrain = accuracy_score(yTrain, yPredTrainBin)
    accTest = accuracy_score(yTest, yPredTestBin)

    return matrixTrain, matrixTest, accTrain, accTest


def perceptron(file_path, activationFunc,title, errorThreshold, alpha, gain=1):
    data = load_data(file_path)
    data = normalize(data)
    
    #split features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.astype(int) 

    xTrain, xTest, yTrain, yTest = splitData(X, y)

    #train perceptron
    weights, bias = train(xTrain, yTrain, activationFunc, errorThreshold=errorThreshold, alpha=alpha)

    plotting(xTrain, yTrain, weights, bias, activationFunc, title + ' (Train)')
    plotting(xTest, yTest, weights, bias, activationFunc, title + ' (Test)')

    matrixTrain, matrixTest, accTrain, accTest = evalutation(xTrain, yTrain, xTest, yTest, weights, bias, activationFunc)

    print("Training Confusion Matrix:\n", matrixTrain)
    print("Testing Confusion Matrix:\n", matrixTest)
    print("Training Accuracy:", accTrain)
    print("Testing Accuracy:", accTest)


#run for both hard and soft unipolar activation functions
file_path = 'groupB.txt'

print("Hard Unipolar Activation Function - Group B")
perceptron(file_path, hardUnipolarActivation, 'Hard Unipolar Activation Function - Group B', 40, alpha=0.02)

print("Soft Unipolar Activation Function - Group B")
perceptron(file_path, softUnipolarActivation, 'Soft Unipolar Activation Function - Group B', 40, alpha=0.02, gain=1)
