import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Use by running python proj2.py

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
        min_val = normDF[column].min()
        max_val = normDF[column].max()
        
        normDF[column] = (normDF[column] - min_val) / (max_val - min_val)  
        normDF[column] = normDF[column] * (0.5 - (-0.5)) + (-0.5)
    
    return normDF

#split data into training and testing sets
def splitData(X, y, trainSize):

    #shuffling for randomness
    data = np.hstack((X, y.reshape(-1, 1)))
    np.random.shuffle(data)

    shuffledX = data[:, :-1] #seperating shuffled data again
    shuffledY = data[:, -1]

    splitIndex = int(len(X) * trainSize)
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

def plotting(X, y, weights, bias, title):

    plt.figure(figsize=(8, 6))

    #plot data points: small car (0) as red and big car (1) as blue
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Small Car (0)', alpha=0.7)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Big Car (1)', alpha=0.7)

    #create decision boundary
    xVals = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 200)
    yVals = -(weights[0] * xVals + bias) / weights[1]  

    #decision boundary
    plt.plot(xVals, yVals, color='green', label='Decision Boundary', linewidth=2)

    plt.xlim(-0.55, 0.55)
    plt.ylim(-0.55, 0.55)

    plt.xlabel('cost in USD')
    plt.ylabel('Weight in pounds (lbs)')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)

    plt.show()


def confMatrix(yTrue, yPred):
    TP = TN = FP = FN = 0

    for tLabel, pLabel in zip(yTrue, yPred):
        if tLabel == 1 and pLabel == 1:
            TP += 1
        elif tLabel == 0 and pLabel == 0:
            TN += 1
        elif tLabel == 0 and pLabel == 1:
            FP += 1
        elif tLabel == 1 and pLabel == 0:
            FN += 1

    mtrx = np.array([[TN, FP],
                     [FN, TP]])

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    totalP = TP + FN  
    totalN = TN + FP 

    tpRate = TP / totalP if totalP > 0 else 0
    tnRate = TN / totalN if totalN > 0 else 0
    fpRate = FP / totalN if totalN > 0 else 0
    fnRate = FN / totalP if totalP > 0 else 0

    print('\n')
    print(f"True Positive Rate (TP): {tpRate:.4f}")
    print(f"True Negative Rate (TN): {tnRate:.4f}")
    print(f"False Positive Rate (FP): {fpRate:.4f}")
    print(f"False Negative Rate (FN): {fnRate:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    return mtrx, accuracy

def evalutation(xTest, yTest, weights, bias, activationFunc):

    yPredTest = activationFunc(np.dot(xTest, weights) + bias)
    yPredTestBin = np.where(yPredTest >= 0.5, 1, 0) 

    print('\n')
    print("\nTesting Metrics:")
    matrixTest, accuracyTest = confMatrix(yTest, yPredTestBin)

    return matrixTest, accuracyTest



def perceptron(file_path, activationFunc, title, trainSize, errorThreshold, alpha, gain=1):
    data = load_data(file_path)
    data = normalize(data)
    
    #split features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.astype(int) 

    xTrain, xTest, yTrain, yTest = splitData(X, y, trainSize)

    #train perceptron
    weights, bias = train(xTrain, yTrain, activationFunc, errorThreshold=errorThreshold, alpha=alpha)

    plotting(xTrain, yTrain, weights, bias, title + ' (Train)')
    plotting(xTest, yTest, weights, bias, title + ' (Test)')

    matrixTest, accTest = evalutation(xTest, yTest, weights, bias, activationFunc)

    print('\n')
    print("Testing Confusion Matrix:\n", matrixTest)
    print("Testing Accuracy:", accTest)

print('\n train size is 75%')

file_path = 'groupA.txt'

print("\nHard Unipolar Activation Function - Group A")
perceptron(file_path, hardUnipolarActivation, 'Hard Unipolar Activation Function - Group A', 0.75, 1e-5, alpha=0.05)

print("\nSoft Unipolar Activation Function - Group A")
perceptron(file_path, softUnipolarActivation, 'Soft Unipolar Activation Function - Group A', 0.75, 1e-5, alpha=0.05, gain=0.08)


file_path = 'groupB.txt'

print("\nHard Unipolar Activation Function - Group B")
perceptron(file_path, hardUnipolarActivation, 'Hard Unipolar Activation Function - Group B', 0.75,  40, alpha=0.05)

print("\nSoft Unipolar Activation Function - Group B")
perceptron(file_path, softUnipolarActivation, 'Soft Unipolar Activation Function - Group B', 0.75, 40, alpha=0.05, gain=0.08)

file_path = 'groupC.txt'

print("\nHard Unipolar Activation Function - Group C")
perceptron(file_path, hardUnipolarActivation, 'Hard Unipolar Activation Function - Group C', 0.75, 700, alpha=0.01)

print("\nSoft Unipolar Activation Function - Group C")
perceptron(file_path, softUnipolarActivation, 'Soft Unipolar Activation Function - Group C', 0.75, 700, alpha=0.01, gain=0.1)

print('\n train size is 25%')

file_path = 'groupA.txt'

print("\nHard Unipolar Activation Function - Group A")
perceptron(file_path, hardUnipolarActivation, 'Hard Unipolar Activation Function - Group A', 0.25, 1e-5, alpha=0.05)

print("\nSoft Unipolar Activation Function - Group A")
perceptron(file_path, softUnipolarActivation, 'Soft Unipolar Activation Function - Group A', 0.25, 1e-5, alpha=0.05, gain=0.08)


file_path = 'groupB.txt'

print("\nHard Unipolar Activation Function - Group B")
perceptron(file_path, hardUnipolarActivation, 'Hard Unipolar Activation Function - Group B', 0.25,  40, alpha=0.05)

print("\nSoft Unipolar Activation Function - Group B")
perceptron(file_path, softUnipolarActivation, 'Soft Unipolar Activation Function - Group B', 0.25, 40, alpha=0.05, gain=0.08)

file_path = 'groupC.txt'

print("\nHard Unipolar Activation Function - Group C")
perceptron(file_path, hardUnipolarActivation, 'Hard Unipolar Activation Function - Group C', 0.25, 700, alpha=0.01)

print("\nSoft Unipolar Activation Function - Group C")
perceptron(file_path, softUnipolarActivation, 'Soft Unipolar Activation Function - Group C', 0.25, 700, alpha=0.01, gain=0.1)
