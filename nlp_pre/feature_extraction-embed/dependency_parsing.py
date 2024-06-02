import stanza
from stanza.pipeline.core import DownloadMethod

nlp = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos,lemma,depparse', download_method=DownloadMethod.REUSE_RESOURCES)
doc = nlp('Nous avons atteint la fin du sentier.')
print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
