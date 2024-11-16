import re
from collections import Counter
import pandas as pd
from IPython.display import display
from Porter_Stemmer_Python import PorterStemmer
import numpy as np



from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def load_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

#;oad paragraphs and stop words from given files
textParagraphs = load_file('Project4_paragraphs.txt')
stopWordstxt = set(load_file('Project4_stop_words.txt').split())

#initialize the stemmer
stemmer = PorterStemmer()

#tokenizing by paragraphs
paragraphs = textParagraphs.split('\n\n')  #assuming paragraphs are separated by double newlines i think


def cleanText(text):
    text = re.sub(r'<.*?>', '', text)  #remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  #remove punctuation
    text = re.sub(r'\d+', '', text)  #remove numbers
    return text.lower()  #convert to lowercase

def removeStopWords(tokens):
    return [word for word in tokens if word not in stopWordstxt]

def wordStemmer(tokens):
    return [stemmer.stem(word, 0, len(word) - 1) for word in tokens]


def kohonen(tdm, nClusters, epochs, iLearningRate):
    nFeatures = tdm.shape[1]
    weights = np.random.rand(nClusters, nFeatures)

    for epoch in range(epochs):
        learningRate = iLearningRate * (1 - epoch / epochs)  #decay learning rate
        for row in tdm:
            #calculate distances and find the winner
            distances = np.linalg.norm(weights - row, axis=1)
            winner = np.argmin(distances)

            #update winner's weights
            weights[winner] += learningRate * (row - weights[winner])
    
    #assign each  to the closest cluster
    paragraphCluster = [
        np.argmin(np.linalg.norm(weights - row, axis=1)) for row in tdm
    ]
    return paragraphCluster



def main():
    #extract frequency and create feature vector
    paragraphWordCounts = []
    i = 0
    for paragraph in paragraphs:
        cleanedParagraph = cleanText(paragraph)
        tokens = cleanedParagraph.split()  #further tokenize each paragraph into tokens
        tokens = removeStopWords(tokens)  
        stemmedWords = wordStemmer(tokens)  
        
        #get word frequency for each paragraph
        wordFreq = Counter(stemmedWords)
        paragraphWordCounts.append(wordFreq)



    #aggregate word counts across all paragraphs
    totalWordFreq = Counter()
    for wordCount in paragraphWordCounts:
        totalWordFreq.update(wordCount)

    T = 25  #threshold for feature vec


    print(f"total frequency of each word:", totalWordFreq)

    #generate feature vector by selecting words that appear at least T times across all paragraphs
    featureVec = [word for word, count in totalWordFreq.items() if count > T]

    print(f"Feature Vector (words with frequency > {T}):", featureVec)

    tdm = []
    for wordCount in paragraphWordCounts:
        row = [wordCount.get(word, 0) for word in featureVec]
        tdm.append(row)

    tdmDF = pd.DataFrame(tdm, columns=featureVec)
    print("\nTerm Document Matrix (TDM):")
    display(tdmDF)

    print(tdmDF.describe())


    tdmDF = tdmDF.to_numpy()

    nClusters = 10
    epochs = 200
    learningRate = 0.001

    clusters = kohonen(tdmDF, nClusters, epochs, learningRate)
    print("cluster assignments for each paragraph:", clusters)




if __name__ == "__main__":
    main()
