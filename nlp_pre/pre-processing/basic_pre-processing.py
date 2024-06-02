### Case normalization (e.g. lowercase)
# df = df.applymap(lambda x: x.lower() if x.isinstance(x, str) else x)

text = '...'
text_ = lambda x: x.lower() if x.isinstance(x, str) else x 

######## ######## ########

### Punctuation removal 
alpnum_pattern = '\w\s'
# If i wanted to save any other special characters, depending on my domain: 
spec_chars_to_keep = ['$', '+', '-', '=', '*', '/', '?', '!']
expanded_pattern_to_keep = re.compile(u'[' + alpnum_pattern + ''.join(re.escape(k) for k in spec_chars_to_keep) + u']')


# if I only wanted alphanumeric characters and underscores: 
clean_text = re.sub(r'[^\w\s]', '', text)

######## ######## ########

### Removing HTML tags 
html_tag_pattern = '<.*?>'
re.sub(html_tag_pattern, "", text)

######## ######## ########

### Removing URLs

url_pattern = r'https?:\/\/\S+'
re.sub(url_pattern, "", text)


# def remove_urls(text):
# 	return re.sub(url_pattern, "", text)
# df.apply(remove_urls)

######## ######## ########

### Removing stop words 
# I will do it with sklearn, since it is what we will use for non-deep learning NLP ML tasks


from sklearn.feature_extraction.text import CountVectorizer

# Example text
text = ["This is a simple example to demonstrate the removal of stop words using sklearn."]

# Initialize CountVectorizer with the 'english' stop words list
vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the data
X = vectorizer.fit_transform(text)

# Get the feature names (words remaining after removing stop words)
remaining_words = vectorizer.get_feature_names_out()

# Reconstruct the filtered text
filtered_text = " ".join(remaining_words)

print(filtered_text)
# Output: "demonstrate example removal simple sklearn stop words"


### If i would like to define my own stop words to use: 
# Custom stop words list
custom_stop_words = ['simple', 'removal']
# Initialize CountVectorizer with custom stop words
vectorizer = CountVectorizer(stop_words=custom_stop_words)


### If i would like to EXTEND the default stop words list with my custom ones: 
# Custom stop words list
custom_stop_words = {'example', 'removal', 'using'}
# Combine with built-in English stop words
combined_stop_words = ENGLISH_STOP_WORDS.union(custom_stop_words)
# Initialize CountVectorizer with the combined stop words
vectorizer = CountVectorizer(stop_words=combined_stop_words)


### If i would like to DROP some of the stop words mentioned in the default list
 # Stop words to remove from the default list
words_to_drop = {'not', 'using'}
# Combine the remaining default stop words with the words_to_keep removed
custom_stop_words = ENGLISH_STOP_WORDS.difference(words_to_drop)
# Initialize CountVectorizer with the modified stop words
vectorizer = CountVectorizer(stop_words=custom_stop_words)




######## ######## ########  REMOVING DUPlICATE TEXT ######## ######## ########
### Catch triplets+ of chars and replace with double ones
# In some use cases -e.g. in sentiment analysis- it is important to retrieve this info
# because the divergence from standard spelling might be on purpose.

sequencePattern   = r"(.)\1\1+"  # this 'catches' the occurrence of triplets, quintets, etc of single characters, e.g. kkk, hhhh, zzzzz, 111111
seqReplacePattern = r"\1\1"  # maps the character matched by the sequencePattern (the one that got 'caught' using '.') to a double occurrence, e.g. 666 -> 66 

# Replace 3 or more consecutive letters (sequencePattern) by 2 letter (seqReplacePattern) - FILL YOUR SOLUTION HERE #
tweet = re.sub(sequencePattern, seqReplacePattern, tweet)