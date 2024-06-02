import stanza
from stanza.pipeline.core import DownloadMethod


nlp = stanza.Pipeline(lang='en', processors='tokenize,ner', download_method=DownloadMethod.REUSE_RESOURCES)
doc = nlp("Chris Manning teaches at Stanford University. He lives in the Bay Area.")
print(*[f'token: {token.text}\tner: {token.ner}' for sent in doc.sentences for token in sent.tokens], sep='\n')


# or - more dedicated and better results, but less efficient? 
print(*[f'entity: {ent.text}\ttype: {ent.type}' for sent in doc.sentences for ent in sent.ents], sep='\n')
