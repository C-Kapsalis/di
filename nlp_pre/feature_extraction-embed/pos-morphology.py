import stanza 
from stanza.pipeline.core import DownloadMethod
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos', download_method=DownloadMethod.REUSE_RESOURCES)
doc = nlp('Barack Obama was born in Hawaii.')
print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for sent in doc.sentences for word in sent.words], sep='\n')

# word.upos is the same as the plain 'word.pos' 
# 'feats' corresponds to morphological features 
# upos -> universal part of speech tag, xpos -> treebank-specific part of speech tag 