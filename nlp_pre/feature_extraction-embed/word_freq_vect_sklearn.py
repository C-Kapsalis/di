import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
stopwords_set = set(stopwords.words('english'))

# sample data to work with 
documents = ["This is the first document.",  "This document is the second document.",  "And this is the third one.", "Is this the first document?"]
# converting available text data to a df 
data = pd.DataFrame({'Text': documents})



	### BOW Implementation ###
# initializing the CountVectorizer 
vectorizer = CountVectorizer(stop_words=list(stopwords_set), ngram_range=(1,3))
bow_vectors = vectorizer.fit_transform(data['Text'])
bow_df = pd.DataFrame(bow_vectors.toarray(), 
						columns=vectorizer.get_feature_names_out())
						
# print the bow dataframe
print(bow_df.head())


	### TF-IDF Implementation ###
vectorizer2 = TfidfVectorizer(stop_words=list(stopwords_set), ngram_range=(1,3))
tfidf_vectors = vectorizer2.fit_transform(data['Text'])
tdidf_df = pd.DataFrame(tfidf_vectors.toarray(), 
						columns=vectorizer2.get_feature_names_out())
						
# print the bow dataframe
print(tdidf_df.head())
