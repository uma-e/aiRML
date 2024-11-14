import re
from collections import Counter, defaultdict
import pandas as pd
from tabulate import tabulate
from IPython.display import display
from Porter_Stemmer_Python import PorterStemmer


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

#extract frequency and create feature vector
paragraphWordCounts = []
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

T = 30  #threshold for feature vec

#generate feature vector by selecting words that appear at least T times across all paragraphs
featureVec = [word for word, count in totalWordFreq.items() if count >= T]

print(f"Feature Vector (words with frequency >= {T}):", featureVec)

tdm = []
for wordCount in paragraphWordCounts:
    row = [wordCount.get(word, 0) for word in featureVec]
    tdm.append(row)

tdm_df = pd.DataFrame(tdm, columns=featureVec)
print("\nTerm Document Matrix (TDM):")
display(tdm_df)
