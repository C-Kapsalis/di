# if you encounter a weird bug in calling T-SNE when your word_vectors contain negative
# values, do pip install --upgrade threadpoolctl


import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from num2words import num2words
from spellchecker import SpellChecker
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
import gensim.downloader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
import pickle
import warnings
# ML models
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
# Supress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)


# Download necessary NLTK data files
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('stopwords')

# Initialize objects
spell = SpellChecker()
stop_words = set(stopwords.words('english'))
wl = WordNetLemmatizer()
emojis_pattern = re.compile("["
		u"\U0001F600-\U0001F64F"  # emoticons 
		u"\U0001F300-\U0001F5FF"  # symbols & pictographs 
		u"\U0001F680-\U0001F6FF"  # transport & map symbols 
		u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
		u"\U00002702-\U000027B0"
		u"\U000024C2-\U0001F251"
	"]", 
	flags=re.UNICODE)
emoticons_pattern = re.compile(u'(' + u'|'.join(re.escape(k) for k in sorted(EMOTICONS_EMO.keys(), key=len, reverse=True)) + u')', 
	flags=re.UNICODE)
sequencePattern   = r"(.)\1\1+"  # this 'catches' the occurrence of triplets, quintets, etc of single characters, e.g. kkk, hhhh, zzzzz, 111111
seqReplacePattern = r"\1\1"  # maps the character matched by the sequencePattern (the one that got 'caught' using '.') to a double occurrence, e.g. 666 -> 66 
contractions_dict = {
	"ain't": "are_not", 
	"'s": "is",
	"aren't": "are not",
	"can't": "cannot",
    "can't've": "cannot have",
	"'cause": "because",
	"could've": "could have",
	"couldn't": "could not",
	"couldn't've": "could not have",
	"didn't": "did not",
	"doesn't": "does not",
	"don't": "do not",
	"hadn't": "had not",
	"hadn't've": "had not have",
	"hasn't": "has not",
	"haven't": "have not",
	"he'd": "he would",
	"he'd've": "he would have",
	"he'll": "he will",
	"he'll've": "he will have",
	"how'd": "how did",
	"how'd'y": "how do you",
	"how'll": "how will",
	"I'd": "I would",
	"I'd've": "I would have",
	"I'll": "I will",
	"y'all'd've": "you all would have",
	"y'all're": "you all are",
	"y'all've": "you all have",
	"you'd": "you would",
	"you'd've": "you would have",
	"you'll": "you will",
	"you'll've": "you will have",
	"you're": "you are",
	"you've": "you have"
}
contractions_pattern = re.compile(u'(' + u'|'.join(k for k in sorted(contractions_dict.keys(), key=len, reverse=True)) + u')', flags=re.UNICODE)

# function to replace emoticons
def replace_emoticons(match):
    emoticon = match.group(0)  # retrieve the matched emoticon 
    return "_".join(EMOTICONS_EMO[emoticon].replace(",", "").split())  # Return the corresponding text equivalent from the dictionary
    # the replace & split are needed because some emoticons/emojis correspond to multiple words
# function to replace emojis
def replace_emojis(match):
    emoji = match.group(0)  # retrieve the matched emoticon 
    return "_".join(UNICODE_EMOJI[emoji].replace(",", "").replace(":","").split())  # Return the corresponding text equivalent from the dictionary
# function to replace contractions
def replace_contractions(match):
	contr = match.group(0)
	return contractions_dict[contr]


def preprocess(text):
    text = text.lower()
    text = text.strip()
    text = re.sub(re.compile('<.*?>'), "", text)
    text = re.sub(re.compile(r'https?://\S+|www\.\S+'), "", text)
    text = re.sub(emoticons_pattern, replace_emoticons, text)  # replacing emoticons with their text equivalents
    text = re.sub(emojis_pattern, replace_emojis, text)  # same for emojis
    text = re.sub(re.compile(u'['+ ''.join([re.escape(c) for c in string.punctuation]) + u']'), "", text)
    text = re.sub(re.compile('[^\w\s]'), " ", text)
    text = re.sub(re.compile('\s+'), " ", text)
    text = ' '.join([spell.correction(word) for word in text.split()])
    text = ' '.join([num2words(int(word)) if word.isdigit() else word for word in text.split()])
    # replacing repetitive sequences of chars with doubles
    text = re.sub(sequencePattern, seqReplacePattern, tweet)
    # replacing contractions with the full version of the corresponding verb phrases
    text = re.sub(contractions_pattern, replace_contractions, text)
    text = re.sub(re.compile('\s+'), " ", text)
    return text

def stopword_removal(s):
    return ' '.join([word for word in s.split() if word not in stop_words])

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatizer(s):
    word_pos_tags = nltk.pos_tag(nltk.word_tokenize(s))
    a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for tag in word_pos_tags]
    return " ".join(a)

# Sample corpus
corpus = [
    "This is a sample text. It contains URLs like http://example.com and punctuation!",
    "Another example with numbers like 123 and 456.",
    "The quick brown fox jumps over the lazy dog."
]

# Create DataFrame
data = pd.DataFrame({'text': corpus})
data['processed_text'] = data['text'].apply(lambda x: lemmatizer(stopword_removal(preprocess(x))))

# Tokenize the processed text
sentences = [nltk.word_tokenize(text) for text in data['processed_text']]
tokens = list(set([word for sent in sentences for word in sent]))



########################## OneHotEncoding ###############################

data = pd.DataFrame({'Category': tokens})
encoder = OneHotEnoder(sparse_output=False, dtype=int)
endoded_data = encoder.fit_transform(data)
encoded_df = pd.DataFrame(encoded_data, columns=tokens)




########################## TF-IDF Vectorizer ##########################
# Custom tokenizer that does nothing because text is already preprocessed
def dummy_tokenizer(text):
    return text.split()

# Vectorize the text using TfIdfVectorizer
vectorizer = TfidfVectorizer(tokenizer=dummy_tokenizer, preprocessor=lambda x: x, lowercase=False)
tfidf_vectors = vectorizer.fit_transform(processed_texts)  # processed text is just X_train['processed_text'] - a list of strings, not a list of lists
tfidf_df = pd.DataFrame(tfidf_vectors.toarray(), columns=vectorizer.get_feature_names_out())


tsne = TSNE(n_components=2, perplexity=2, init="random", random_state=42)
tfidf_embedded = tsne.fit_transform(tfidf_vectors)

plt.figure(figsize=(8, 5))
plt.scatter(tfidf_embedded[:, 0], tfidf_embedded[:, 1])

for i, txt in enumerate(words):
    plt.annotate(txt, (tfidf_embedded[i, 0], tfidf_embedded[i, 1]))
    
plt.title('t-SNE Visualization of TF-IDF Vectors')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()



########################## Bag-of-Words Vectorizer ##########################
# Custom tokenizer that does nothing because text is already preprocessed
def dummy_tokenizer(text):
    return text.split()

# Initialize CountVectorizer
vectorizer = CountVectorizer(tokenizer=dummy_tokenizer, preprocessor=lambda x: x, lowercase=False)

# Transform the processed text into a document-term matrix
bow_vectors = vectorizer.fit_transform(df['processed_text'])

# Convert the matrix to a DataFrame for better visualization
count_df = pd.DataFrame(bow_vectors.toarray(), columns=vectorizer.get_feature_names_out())

# Display the DataFrame
print(count_df)

tsne = TSNE(n_components=2, perplexity=2, init="random", random_state=42)
bow_embedded = tsne.fit_transform(bow_vectors)

# Plot the t-SNE visualization in a scatterplot; use tfidf_embedded 
# Optionally, use text annotations in the plot 

plt.figure(figsize=(8, 5))
plt.scatter(bow_embedded[:, 0], bow_embedded[:, 1])

for i, txt in enumerate(documents):
    plt.annotate(txt, (bow_embedded[i, 0], bow_embedded[i, 1]))
    
plt.title('t-SNE Visualization of BOW Vectors')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()



########################## N-Gram Features ##########################
# Custom tokenizer that does nothing because text is already preprocessed
def dummy_tokenizer(text):
    return text.split()



ngram_vectorizer = CountVectorizer(ngram_range=(2,3), tokenizer=dummy_tokenizer, preprocessor=lambda x: x, lowercase=False)
ngram_vectors = ngram_vectorizer.fit_transform(data['processed_text'])
ngram_df = pd.DataFrame(ngram_vectors.toarray(), columns=ngram_vectorizer.get_feature_names_out())
            
# Display the DataFrame
print(ngram_df)

tsne = TSNE(n_components=2, perplexity=2, init="random", random_state=42)
ngram_embedded = tsne.fit_transform(ngram_vectors)

# Plot the t-SNE visualization in a scatterplot; use tfidf_embedded 
# Optionally, use text annotations in the plot 

plt.figure(figsize=(8, 5))
plt.scatter(ngram_embedded[:, 0], ngram_embedded[:, 1])

for i, txt in enumerate(documents):
    plt.annotate(txt, (ngram_embedded[i, 0], ngram_embedded[i, 1]))
    
plt.title('t-SNE Visualization of N-Gram Features')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()






########################## Word2Vec ###############################
# Train Word2Vec model
# w2v expects a list of tokenized sentences
sentences = [doc.split() for doc in df['processed_text']]
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

def get_mean_embedding_vector(doc, model):
    words = doc.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

data['vector'] = data['processed_text'].apply(lambda x: get_mean_embedding_vector(x, w2v))

print(data[['processed_text', 'vector']])

# List the vocabulary words from w2v.wv  
words = list(w2v.wv.key_to_index.keys())

# Retrieve all the word vectors of all the words in the corpus and store in a new variable "word_vectors"
word_vectors = np.array([w2v.wv[word] for word in words])




### Dimensionality reduction with T-SNE
tsne = TSNE(n_components=2, random_state=0, perplexity=10)
tsne_res = pd.DataFrame(tsne.fit_transform(word_vectors), columns=['tsne1', 'tsne2'])
tsne_res['word'] = words

# Print the tsne results
print(tsne_res)

# Create the scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x='tsne1', y='tsne2', data=tsne_res)

# Annotate the scatter plot with the words
for i, word in enumerate(tsne_res['word']):
    plt.annotate(word, (tsne_res['tsne1'][i], tsne_res['tsne2'][i]), fontsize=12)

plt.title('TSNE of Word2Vec Vectors')
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.show()




### Dimensionality reduction with pca
pca = PCA(n_components=2)
pca_scores = pd.DataFrame(pca.fit_transform(word_vectors), columns=['PC1','PC2'])
pca_scores['word'] = words
print(pca_scores)

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', data=pca_scores)

for i, word in enumerate(pca_scores['word']):
    plt.annotate(word, (pca_scores['PC1'][i], pca_scores['PC2'][i]), fontsize=12)

plt.title('PCA of Word2Vec Vectors')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()



### For larger datasets, I could get better performance by constructing 
# a scatterplot using Scattergl.
N = 1000000

fig = go.Figure(data=go.Scattergl(
    x = pca_df['PC1'],
    y = pca_df['PC2'],
    mode='markers',
    marker=dict(
        color=np.random.randn(N),
        colorscale='Viridis',
        line_width=1
    ),
    text=pca_df['word'],
    textposition="bottom center"
))

fig.show()



########################### FastText ###############################
ft_model = FastText(sentences, vector_size=100, window=5, min_count=1)

def get_mean_embedding_vector(doc, model):
    words = doc.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

data['vector'] = data['processed_text'].apply(lambda x: get_mean_embedding_vector(x, ft_model))

print(data[['processed_text', 'vector']])

# List the vocabulary words from w2v.wv  
words = list(ft_model.wv.key_to_index.keys())

# Retrieve all the word vectors of all the words in the corpus and store in a new variable "word_vectors"
word_vectors = ft_model.wv[words]

# Perform TSNE
tsne = TSNE(n_components=2, random_state=0, perplexity=1)
tsne_res = pd.DataFrame(tsne.fit_transform(word_vectors), columns=['tsne1', 'tsne2'])
tsne_res['word'] = words

# Print the tsne results
print(tsne_res)





########################### Working with Pretrained Models ###############################
# to inspect all the gensim pre-trained models available for download
print(list(gensim.downloader.info()['models'].keys()))

# downloading a specific model 
glove_vectors = gensim.downloader.load('glove-twitter-25')

# so now this is equivalent to having trained a 'w2v' model on our own corpus
# so I could do e.g. 
glove_vectors.most_similar('twitter')

# List the vocabulary words from w2v.wv  
words = list(glove_vectors.key_to_index.keys())

# Print the key_to_index from w2v.wv to get the mapping of a word into a number in a dictionary format
print(glove_vectors.key_to_index['great'])

# Retrieve and print the vector of a specific word (such as 'quick') from the corpus using w2v.wv  
print(glove_vectors['quick'])

# Retrieve and print the 10 top similar words of a specific word (such as 'dog') from the corpus using w2v.wv  
ms = glove_vectors.most_similar(positive=['dog'], topn=10)
plain_ms_words = [el[0] for el in ms]

# Get two words from the corpus and calculate the similarity between them using w2v.wv  
print(glove_vectors.similarity('joy', 'dog'))

# Retrieve all the word vectors of all the words in the corpus and store in a new variable "word_vectors"
word_vectors = np.array([glove_vectors[word] for word in words])



####################################################################################




