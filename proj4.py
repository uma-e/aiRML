import re
from collections import Counter
from Porter_Stemmer_Python import PorterStemmer


def load_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

textParagraphs = load_file('Project4_paragraphs.txt')
stopWordstxt = set(load_file('Project4_stop_words.txt').split())

#getting the stemmer from the provided file
stemmer = PorterStemmer()

#tokenizing
paragraphs = textParagraphs.split('\n\n')  #assuming paragraphs are separated by double newlines

def cleanText(text):
    text = re.sub(r'<.*?>', '', text)  #removing html tags
    text = re.sub(r'[^\w\s]', '', text)  #removing punctuation
    text = re.sub(r'\d+', '', text)  #removing numbers
    return text.lower()  #converting to lowercase

#removing the stop words using the given file
def removeStopWords(tokens):
    return [word for word in tokens if word not in stopWordstxt]

#using the port stemmer
def wordStemmer(tokens):
    return [stemmer.stem(word, 0, len(word) - 1) for word in tokens]

#extract frequency and create the feature vector
paragraphWordCounts = []
for paragraph in paragraphs:
    cleanedParagraph = cleanText(paragraph)
    tokens = cleanedParagraph.split() 
    tokens = removeStopWords(tokens)  
    stemmedWords = wordStemmer(tokens)  
    
    #get word frequency
    wordFreq = Counter(stemmedWords)
    paragraphWordCounts.append(wordFreq)

#aggregate word counts across all paragraphs
totalWordFreq = Counter()
for wordCount in paragraphWordCounts:
    totalWordFreq.update(wordCount)

T = 11

#feature vector: only include words with frequency >= T
featureVec = [word for word, count in totalWordFreq.items() if count > T]

print(f"Feature Vector (words with frequency >= {T}):", featureVec)

