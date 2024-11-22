# NLTK (Natural Language Processing Toolkit)

## What is NLTK and why is it used?

NLTK is a leading platform for building Python programs to work with human language data. It's used because:
- Provides interfaces to over 50 corpora and lexical resources
- Offers a suite of text processing libraries
- Supports classification, tokenization, stemming, tagging, parsing
- Includes robust documentation and tutorials
- Has active community support
- Ideal for both research and production applications

## What are the core concepts in NLTK?

The core concepts include:
1. **Tokenization**: Breaking text into words or sentences
2. **Stemming**: Reducing words to their root/base form
3. **Lemmatization**: Converting words to their dictionary form
4. **Part-of-Speech Tagging**: Marking words with their grammatical parts
5. **Named Entity Recognition**: Identifying proper nouns
6. **Parsing**: Analyzing sentence structure

## How do you get started with NLTK?

```python
import nltk

# Download necessary NLTK data
nltk.download('popular')  # Downloads popular packages
# Or download specific packages
nltk.download('punkt')  # For tokenization
nltk.download('averaged_perceptron_tagger')  # For POS tagging
nltk.download('maxent_ne_chunker')  # For NER
nltk.download('words')  # For word corpus
nltk.download('wordnet')  # For lemmatization
```

## How do you perform tokenization?

1. **Word Tokenization**:
```python
from nltk.tokenize import word_tokenize, RegexpTokenizer

# Basic word tokenization
text = "Hello, how are you doing today? I'm doing great!"
words = word_tokenize(text)
# ['Hello', ',', 'how', 'are', 'you', 'doing', 'today', '?', 'I', "'m", 'doing', 'great', '!']

# Custom tokenization using regex
tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)
# ['Hello', 'how', 'are', 'you', 'doing', 'today', 'I', 'm', 'doing', 'great']
```

2. **Sentence Tokenization**:
```python
from nltk.tokenize import sent_tokenize

text = "This is first sentence. This is second sentence! Is this third sentence?"
sentences = sent_tokenize(text)
# ['This is first sentence.', 'This is second sentence!', 'Is this third sentence?']
```

## How do you perform stemming and lemmatization?

1. **Stemming**:
```python
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

# Porter Stemmer
porter = PorterStemmer()
words = ['running', 'runs', 'ran', 'runner']
[porter.stem(word) for word in words]
# ['run', 'run', 'ran', 'runner']

# Lancaster Stemmer
lancaster = LancasterStemmer()
[lancaster.stem(word) for word in words]
# ['run', 'run', 'ran', 'run']

# Snowball Stemmer (Porter2)
snowball = SnowballStemmer('english')
[snowball.stem(word) for word in words]
# ['run', 'run', 'ran', 'runner']
```

2. **Lemmatization**:
```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Default (noun) lemmatization
words = ['caring', 'cars', 'bikes', 'riding']
[lemmatizer.lemmatize(word) for word in words]

# Specifying part of speech
word = 'caring'
lemmatizer.lemmatize(word, pos='v')  # verb
lemmatizer.lemmatize(word, pos='n')  # noun
lemmatizer.lemmatize(word, pos='a')  # adjective
```

## How do you perform Part-of-Speech (POS) Tagging?

```python
from nltk import pos_tag
from nltk.tokenize import word_tokenize

text = "NLTK is a powerful tool for natural language processing"
tokens = word_tokenize(text)
tagged = pos_tag(tokens)
# [('NLTK', 'NNP'), ('is', 'VBZ'), ('a', 'DT'), ('powerful', 'JJ'),
#  ('tool', 'NN'), ('for', 'IN'), ('natural', 'JJ'), ('language', 'NN'),
#  ('processing', 'NN')]

# Common POS tags:
# NNP: Proper noun
# NN: Noun
# VB: Verb
# JJ: Adjective
# RB: Adverb
# IN: Preposition
# DT: Determiner
```

## How do you perform Named Entity Recognition (NER)?

```python
from nltk import ne_chunk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text = "John works at Google in New York"
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)
named_entities = ne_chunk(pos_tags)

# Parse tree representation
print(named_entities)

# Extract named entities
named_entities_list = []
for chunk in named_entities:
    if hasattr(chunk, 'label'):
        named_entities_list.append((chunk.label(), ' '.join(c[0] for c in chunk)))
# [('PERSON', 'John'), ('ORGANIZATION', 'Google'), ('GPE', 'New York')]
```

## How do you work with NLTK's built-in corpora?

```python
from nltk.corpus import gutenberg, brown, reuters, wordnet

# Gutenberg corpus
files = gutenberg.fileids()
hamlet = gutenberg.words('shakespeare-hamlet.txt')

# Brown corpus
categories = brown.categories()
words = brown.words(categories='news')

# Reuters corpus
files = reuters.fileids()
words = reuters.words('training/9865')

# WordNet
syns = wordnet.synsets("program")
lemmas = syns[0].lemmas()
definition = syns[0].definition()
examples = syns[0].examples()
```

## How do you perform frequency analysis?

1. **FreqDist**:
```python
from nltk import FreqDist
from nltk.tokenize import word_tokenize

text = "This is a sample text. This text is for frequency analysis."
tokens = word_tokenize(text.lower())

# Create frequency distribution
fdist = FreqDist(tokens)

# Most common words
print(fdist.most_common(5))

# Frequency of specific word
print(fdist['text'])

# Plot frequency distribution
fdist.plot(30)  # Plot top 30 words
```

2. **Collocations**:
```python
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.metrics import TrigramAssocMeasures

# Bigram collocations
bigram_measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(tokens)
finder.nbest(bigram_measures.pmi, 10)  # Top 10 collocations

# Trigram collocations
trigram_measures = TrigramAssocMeasures()
finder = TrigramCollocationFinder.from_words(tokens)
finder.nbest(trigram_measures.pmi, 10)
```

## How do you perform text classification?

```python
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

# Prepare training data
def word_feats(words):
    return dict([(word, True) for word in words])

positive_words = ['awesome', 'good', 'nice', 'great']
negative_words = ['bad', 'terrible', 'awful', 'horrible']

positive_features = [(word_feats(pos), 'pos') for pos in positive_words]
negative_features = [(word_feats(neg), 'neg') for neg in negative_words]

# Train classifier
train_set = positive_features + negative_features
classifier = NaiveBayesClassifier.train(train_set)

# Classify new text
test_sentence = "This movie is awesome"
features = word_feats(word_tokenize(test_sentence))
print(classifier.classify(features))

# Show most informative features
classifier.show_most_informative_features()
```

## How do you perform sentiment analysis?

```python
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Analyze sentiment
text = "This movie was really great! I enjoyed it a lot."
scores = sia.polarity_scores(text)
# Returns: {'neg': 0.0, 'neu': 0.446, 'pos': 0.554, 'compound': 0.8016}

# Interpret results
if scores['compound'] >= 0.05:
    sentiment = 'Positive'
elif scores['compound'] <= -0.05:
    sentiment = 'Negative'
else:
    sentiment = 'Neutral'
```

## How do you handle text preprocessing?

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation
    tokens = [token for token in tokens 
             if token not in string.punctuation]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens 
             if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens
```

## How do you perform text similarity analysis?

```python
from nltk.metrics.distance import edit_distance
from nltk.util import ngrams

# Levenshtein distance
word1 = "python"
word2 = "pytorch"
distance = edit_distance(word1, word2)

# N-gram similarity
def get_ngrams(text, n):
    tokens = word_tokenize(text.lower())
    return list(ngrams(tokens, n))

text1 = "The quick brown fox"
text2 = "The brown quick fox"
bigrams1 = set(get_ngrams(text1, 2))
bigrams2 = set(get_ngrams(text2, 2))

# Jaccard similarity
similarity = len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2)
```