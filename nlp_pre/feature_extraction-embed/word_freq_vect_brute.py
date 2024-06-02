import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from nltk.corpus import stopwords
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
from num2words import num2words 
from spellchecker import SpellChecker
from sklearn.manifold import TSNE

nlp = spacy.load('en_core_web_sm')
stopwords_set = set(stopwords.words('english'))

# instance of spellchecker for spelling and grammar corrections
spell = SpellChecker() 


text = ''
with open('/Users/chkapsalis/Desktop/nlp_pre/sample_text/difficult_text.txt', 'r') as fhand:
	for line in fhand:
		text += line.rstrip()

text = text.split('.')[:-1]

def replace_emoticons(match):
    emoticon = match.group(0)  # retrieve the matched emoticon 
    return "_".join(EMOTICONS_EMO[emoticon].replace(",", "").split())  # Return the corresponding text equivalent from the dictionary
def replace_emojis(match):
    emoji = match.group(0)  # retrieve the matched emoticon 
    return "_".join(UNICODE_EMOJI[emoji].replace(",", "").replace(":","").split())  # Return the corresponding text equivalent from the dictionary
def remove_all_instances(text_t, k):
	return [el for el in text_t if el != k]


def basic_preprocessing(s):
	# converting to lowercase
	s = (lambda x: x.lower() if isinstance(x, str) else x)(s)
	
	# removing punctuation but keeping some meaningful characters
	#alpnum_pattern = '^\w\s'
	alpnum_pattern = '\w\s'
	#spec_chars_to_keep = ['$', '+', '-', '=', '*', '/', '?', '!']
	spec_chars_to_keep=[]
	expanded_pattern_to_keep = re.compile(u'[^' + alpnum_pattern + ''.join(re.escape(k) for k in spec_chars_to_keep) + u']')
	s = re.sub(expanded_pattern_to_keep, '', s)
	
	# removing emojis/emoticons 
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
	s = re.sub(emojis_pattern, replace_emojis, s)
	s = re.sub(emoticons_pattern, replace_emoticons, s)

	# removing urls, html tags, triplets+ of chars with doubles
	urlPattern = re.compile('(http://)[^\s]+|(https://)[^\s]+|( www\.)[^\s]+')
	userHandlePattern = re.compile('@[^\s]+')
	s = re.sub(urlPattern, '', s)
	s = re.sub(userHandlePattern, '', s)
	sequencePattern   = re.compile('(.)\1\1+')
	seqReplacePattern = '\1\1'
	s = re.sub(sequencePattern, seqReplacePattern, s)
	
	# replacing numbers with words PLUS spell check
	wds = [] 
	for w in s.split():
		if w.isdigit():
			wds.append(num2words(w))
		else:
			if spell.unknown(w):	# will just ignore if it is empty
				w = spell.correction(w)
			# w is a correctly spelled actual word...
			wds.append(w)
	# Remove None values from wds - they indicate words that are not correctly spelled english words, but there is no known correction for them
	wds = remove_all_instances(wds, None)
	s = ' '.join(wds)
	
	# removing stop words - tokenization, and lemmatization
	doc = nlp(s)
	s = [token.lemma_ for token in doc if token.text.lower() not in stopwords_set and not token.is_punct]
	
	
	return s



#pre_proc = basic_preprocessing(text)
#print(pre_proc)

proc_sentences = [basic_preprocessing(sent) for sent in text]
print(proc_sentences)

unique_words = set([w for sent in proc_sentences for w in sent])



### finding the frequency of each word in each sentence ### 
w_dict = {w: [] for w in unique_words}
for t_sent in proc_sentences:
	word_counts = dict(Counter(t_sent))
	containing = set(word_counts.keys())
	missing = unique_words.difference(containing)
	for word,count_ in word_counts.items():  # by default works with the unique contained words
		w_dict[word].append(count_)
	for word in missing:
		w_dict[word].append(0)
		
# creating the BOW representation # 

bog_df = pd.DataFrame(w_dict, index = range(len(proc_sentences)))
#print(bog_df)

# aggregating to see the most frequent words # 
agg_df = pd.DataFrame({bog_df.columns[i]: bog_df.sum(axis=0)[i] for i in range(len(bog_df.columns))}, index=[0])
agg_df = agg_df.transpose()
agg_df.columns = ['count']
#print(agg_df.sort_values(by=['count'], axis=0, ascending=False)[:20])

# computing the IDF of each word # 
idf_df = pd.DataFrame({col: [np.log(bog_df.shape[0] / (bog_df[col] != 0).sum())] for col in bog_df.columns})
#print(idf_df)
# !!! sklearn uses a slightly different formula for smoothing: log((1+n)/(1+df(t)))+1,
# where n: the total number of documents, df(t): the number of documents containing the term t
#idf_df = pd.DataFrame({col: [np.log((1+bog_df.shape[0]) / (1+(bog_df[col] != 0).sum())) + 1] for col in bog_df.columns})


# constructing the TD-IDF dataframe # 
td_idf_df = pd.DataFrame({bog_df.columns[i]: [bog_df.iloc[i, j] * idf_df.iloc[0,i] for j in range(bog_df.shape[0])] for i in range(bog_df.shape[0])})
#print(td_idf_df)


# visualizing our results
n_comps = 2
tsne = TSNE(n_components=n_comps, perplexity=2, init='random', random_state=42)
tfidf_embedded = tsne.fit_transform(td_idf_df.values)
tfidf_embedded_df = pd.DataFrame(tfidf_embedded, columns=[f'feature_{i}' for i in range(n_comps)])


# each row of the tfidf_embedded_df (just like of td_idf_df) corresponds to a sentence/element of 'text'
# so I can add annotations to link specific 'dots' of the scatterplot to given initial sentences
fig, ax = plt.subplots(figsize=(12,10))
ax = sns.scatterplot(data=tfidf_embedded_df, x='feature_0', y='feature_1')
for i, txt in enumerate(text):
	#if i < 15:
	plt.annotate(txt, (tfidf_embedded[i, 0], tfidf_embedded[i, 1]), fontsize=9, color='red')
plt.show()




