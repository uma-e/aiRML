import pandas as pd
import numpy as np
import seaborn as sns

def hardUnipolar(x):
    return 1

def softUnipolar(x):
    return 1

def normalize(df):
    #zi = (xi - min(x)) / (max(x) - min(x))
    normDF = df.copy()
    for column in normDF.columns[:-1]:  # Skip the last column (target/label)
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

def trainNeuron():
    return 1

def testNeuron():
    return 0

def plotBoundary():
    return 0

def main():
    print("yo mama is so wonderful")

if __name__ == '__main__':
    main()
