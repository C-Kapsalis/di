# removing emojis and emoticons 

from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
emojis_uni = re.compile("["
		u"\U0001F600-\U0001F64F"  # emoticons 
		u"\U0001F300-\U0001F5FF"  # symbols & pictographs 
		u"\U0001F680-\U0001F6FF"  # transport & map symbols 
		u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
		u"\U00002702-\U000027B0"
		u"\U000024C2-\U0001F251"
	"]", 
	flags=re.UNICODE)

emoticons_uni = re.compile(u'(' + u'|'.join(re.escape(k) for k in sorted(EMOTICONS_EMO.keys(), key=len, reverse=True)) + u')', 
	flags=re.UNICODE)


# I want to catch all emoticons or emojis in a given text and turn them into words
# the proposed approach involves just iterating through all emojis/emoticons of the 
# 'emot' library of python, search the text for each of them, and then substitute that
# I will follow a more efficient approach: 
# I will first find all corresponding patters in the given string, and then map
# those to the strings proposed by the 'emot' library in a dictionary, and, ultimately, 
# substitute those using a DICT


# finding all emoji/emoticon occurrences in a given text 
emoji_occurrences = re.findall(emojis_uni, text) 
emoticon_occurrences = re.findall(emoticons_uni, text) 

	
# Function to replace emoticons with their respective text equivalents
def replace_emoticons(match):
    emoticon = match.group(0)  # retrieve the matched emoticon 
    return "_".join(EMOTICONS_EMO[emoticon].replace(",", "").split())  # Return the corresponding text equivalent from the dictionary
    # the replace & split are needed because some emoticons/emojis correspond to multiple words
# Substitute emoticons in the text using the mapping
cleaned_text = re.sub(emoticons_pattern, replace_emoticons, text)


# same for emojis
def replace_emojis(match):
    emoji = match.group(0)  # retrieve the matched emoticon 
    return "_".join(UNICODE_EMOJI[emoji].replace(",", "").replace(":","").split())  # Return the corresponding text equivalent from the dictionary
cleaned_text = re.sub(emojis_uni, replace_emojis, text)



print(cleaned_text)



## explanation: 
# `match` object: when using `re.sub` with a function as the replacement argument, python 
# passes each match found in the text to the function as a `match` object. 
# The `match` object provides info about the match through various methods, including 
# group(), span(), and start(). 
# `match.group()` method: this method retrieves specific matched groups based on the group's index. 
# `group(0)` returns the entire matched string. 
