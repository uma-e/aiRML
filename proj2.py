import pandas as pd
import numpy as np
import seaborn as sns

def hardUnipolar(x):
    return 1 if x >= 0 else 0

def softUnipolar(x, gain=1):
    return 1 / (1 + np.exp(-gain * x))

def normalize(df):
    #zi = (xi - min(x)) / (max(x) - min(x))
    normDF = df.copy()
    for column in normDF.columns[:-1]:  
        min = normDF[column].min()
        max = normDF[column].max()
        normDF[column] = (normDF[column] - min) / (max - min)
    
    return normDF

def load_data(filePath, trainSize=0.75):
    df = pd.read_csv(filePath)
    df = normalize(df)  
    splitIDX = int(len(df) * trainSize)
    trainData = df[:splitIDX]
    testData = df[splitIDX:]
    return trainData, testData

def trainNeuron(trainData, alpha, maxIter, epsilon, activationFunc, gain=1):
    features = trainData.iloc[:, :-1].values 
    labels = trainData.iloc[:, -1].values 
    nSamples, nFeatures = features.shape
    
    weights = np.random.uniform(-0.5, 0.5, size=nFeatures)
    bias = np.random.uniform(-0.5, 0.5)
    
    totalError = float('inf')
    iteration = 0
    totalError = 0

    while totalError > epsilon and iteration < maxIter:
        
        for i in range(nSamples):
            weightedSum = np.dot(features[i], weights) + bias
            if activationFunc == "hard":
                prediction = hardUnipolar(weightedSum)
            elif activationFunc == "soft":
                prediction = softUnipolar(weightedSum, gain)
            
            error = labels[i] - prediction
            weights += alpha * error * features[i]
            bias += alpha * error
            
            totalError += error**2
            
        iteration += 1
    
    return weights, bias, totalError

def testNeuron(testData, weights, bias, activationFunc, gain=1):
    features = testData.iloc[:, :-1].values
    labels = testData.iloc[:, -1].values
    nSamples = len(testData)
    
    predictions = []
    
    for i in range(nSamples):
        weightedSum = np.dot(features[i], weights) + bias
        if activationFunc == "hard":
            predictions.append(hardUnipolar(weightedSum))
        elif activationFunc == "soft":
            predictions.append(softUnipolar(weightedSum, gain))
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    print("Confusion Matrix:")
    conf_matrix = confMatrix(predictions, labels)
    print(conf_matrix)
    
    accuracy = np.sum(predictions == labels) / nSamples
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    return predictions



import matplotlib.pyplot as plt

def plotBoundary(trainData, weights, bias, title):
    features = trainData.iloc[:, :-1].values
    labels = trainData.iloc[:, -1].values
    
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm', edgecolors='k')
    
    x1 = np.linspace(min(features[:, 0]), max(features[:, 0]), 100)
    x2 = -(weights[0] / weights[1]) * x1 - bias / weights[1]
    
    plt.plot(x1, x2, 'k-', label='Decision Boundary')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def confMatrix(predictions, labels):
    TP = np.sum((predictions == 1) & (labels == 1))
    TN = np.sum((predictions == 0) & (labels == 0))
    FP = np.sum((predictions == 1) & (labels == 0))
    FN = np.sum((predictions == 0) & (labels == 1))

    return np.array([[TP, FP], [FN, TN]])

def main():
    trainData, testData = load_data('groupB.txt')
    
    weights, bias, totalError = trainNeuron(trainData, alpha=0.01, maxIter=5000, epsilon=40, activationFunc="hard")
    plotBoundary(trainData, weights, bias, title="Decision Boundary (Hard Activation)")
    
    testNeuron(testData, weights, bias, activationFunc="hard")

    
if __name__ == '__main__':
    main()
