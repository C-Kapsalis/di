import stanza
from nltk.stem import PorterStemmer 

# initialize the stanza pipeline and the porter stemmer 
nlp = stanza.Pipeline(lang='en', processors='tokenize')
porter = PorterStemmer()


text = "Running and jumping make a runner joyful."



### Stemming 
# process the text w/ stanza for tokenization 
doc = nlp(text)
stemmed_tokens = [porter.stem(word.text) for sent in doc.sentences for word in sent.words]

stemmed_text = ' '.join(stemmed_tokens)

print(stemmed_text)


### Lemmatization - NOTE: 
# It is provided in stanza's built-in models: 
nlp = stanza.Pipeline(lang='en', processors='tokenize,lemma')
text = "Running and jumping make a runner joyful."
doc = nlp(text)

lemmatized_tokens = [word.lemma for sent in doc.sentences for word in sent.words]
lemmatized_text = ' '.join(lemmatized_tokens)

print(lemmatized_text)

## !!! Here we use not sent.tokens, but sent.words - this is because in lemmatization we 
# typically need to know the part of speech of the word under study. 
# sent.tokens and sent.words represent different granularities of tokenization. 
# .words -> represent linguistic words. Each word has a lemma and a part-of-speech tag. 
# It may also split contractions into separate linguistic words. 
# .tokens -> Represents tokens as they appear in the original text. Each token is directly tied
# to a position in the original text. Contractions are NOT split. 


# To better understand token vs lemma, execute the following code: 
# Initialize the Stanza pipeline
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma')  # 'pos' here represents part-of-speech classification

# Example text containing contractions
text = "I can't believe it's already 10 o'clock."

# Process the text with Stanza
doc = nlp(text)

# Iterate through sentences, then through tokens and words
for sent in doc.sentences:
    print("Tokens:")
    for token in sent.tokens:
        print(f" - {token.text}")

    print("\nWords:")
    for word in sent.words:
        print(f" - {word.text}, lemma: {word.lemma}, pos: {word.pos}")

    print("\n---\n")
