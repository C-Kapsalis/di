import stanza 

nlp = stanza.Pipeline(lang='en', processors='tokenize')
doc = nlp('This is a test sentence. This is another.')
for i, sentence in enumerate(doc.sentences):
	print(f'==== Sentence {i+1} tokens ====')
	print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')
	
	
	
# doc = nlp(text)
# doc.sentences -> iterator of iterators (list of lists of dictionaries), each of which corresponds
# to a sentence of the given text. Each list corresponding ot a sentence contains one dict
# for each word in it. In the most basic form, each word dict contains its "id" (i.e. the 
# order in which it appears in the sentence), its full text without whitespaces, the location in 
# the given text (character position) that its first char is found, same for its last character,
# as well as an optional "misc" key-value pair containing flags for cases like a missing Space after 
# the given word. 

# The tokens of a sentence are given in a separate iterator (list of dictionaries, each of 
# which is enclosed in a dedicated list itself with length=1) through sentence.tokens
# e.g. doc.sentences[0].tokens 

# For each token, I can retrieve its id (i.e. position of appearance in the sentence it is contained)
# and text, e.g. toks = doc.sentences[0].tokens
# for token in toks:
# 	print(token.id, token.text)
# So toks = [
# [{}],  # token #1
# [{}],  # token #2
# [{}]   # token #3
# ]