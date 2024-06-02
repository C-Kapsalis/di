import re

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

#contraction_occurrences = re.findall(contractions_pattern, text)
text = "I'd kill you if you've not tested's this"
def replace_contractions(match):
	contr = match.group(0)
	return contractions_dict[contr]
	
clean_text = re.sub(contractions_pattern, replace_contractions, text)
print(clean_text)
