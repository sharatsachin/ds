# NLP

Here are concise answers to the questions about natural language processing and text analysis:

## What is tf-idf?

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects the importance of a word in a document within a collection.

TF(t,d) = (Number of times term t appears in document d) / (Total number of terms in d)
IDF(t) = log(Total number of documents / Number of documents containing term t)

$$TF-IDF(t,d) = TF(t,d) * IDF(t)$$

## How is it different from Bag of words?

1. Bag of Words (BoW) only considers term frequency, while TF-IDF also accounts for term importance across documents.
2. BoW gives equal weight to all terms, whereas TF-IDF downweights common terms.
3. TF-IDF provides better feature importance for downstream tasks like classification.

## What are a few embedding techniques?

1. Word2Vec (CBOW and Skip-gram models)
2. GloVe (Global Vectors)
3. FastText
4. BERT embeddings
5. ELMo (Embeddings from Language Models)
6. Universal Sentence Encoder

## What is the difference between word embedding & sentence embedding?

Word embeddings represent individual words in a vector space, capturing semantic relationships between words.

Sentence embeddings represent entire sentences in a vector space, capturing the overall meaning of the sentence. They can be created by aggregating word embeddings or using more advanced techniques like BERT or Universal Sentence Encoder.

## What is Euclidean and cosine distance?

Euclidean distance: Straight-line distance between two points in Euclidean space.
For vectors a and b: $d(a,b) = \sqrt{\sum_{i=1}^n (a_i - b_i)^2}$

Cosine distance: Measure of similarity between two non-zero vectors based on the cosine of the angle between them.
$\cos(\theta) = \frac{a \cdot b}{\|a\| \|b\|}$
Cosine distance = 1 - cosine similarity

## Why do we use cosine distance for text not Euclidean distance?

1. Cosine similarity is invariant to vector magnitude, focusing on direction (content) rather than length.
2. Text vectors are often sparse; cosine similarity handles this well.
3. Euclidean distance is sensitive to document length, which may not be relevant for similarity.
4. Cosine similarity performs better in high-dimensional spaces typical in text analysis.

## Explain one classifier that can be used for sentiment analysis.

Naive Bayes Classifier:

1. Based on Bayes' theorem: $P(c|x) = \frac{P(x|c)P(c)}{P(x)}$
2. Assumes feature independence (naive assumption).
3. For text, often uses multinomial distribution for word counts.
4. Training: Estimate P(c) and P(word|c) from training data.
5. Prediction: Classify new text by calculating P(c|text) for each class and choosing the highest.
6. Fast and effective for sentiment analysis, especially with limited training data.

## What is n-gram? How do we choose the correct value for $n$?

N-gram: Contiguous sequence of n items (words or characters) from a text.

Choosing n:
1. Start with unigrams (n=1) and bigrams (n=2).
2. Increase n based on dataset size and specific task requirements.
3. Use cross-validation to compare performance of different n values.
4. Consider computational resources; higher n increases complexity.
5. Domain knowledge can guide choice (e.g., common phrases in the field).

Generally, n=1 to 3 works well for most tasks. Higher n may capture more context but risks overfitting.

## What is vector representation of text?

Vector representation of text converts text into numerical vectors for machine learning algorithms. Common methods include:

1. One-hot encoding: Binary vector with 1 for present words, 0 for absent.
2. Bag of Words: Vector of word counts.
3. TF-IDF: Vector of term frequency-inverse document frequency scores.
4. Word embeddings: Dense vectors learned from large corpora (e.g., Word2Vec, GloVe).
5. Document embeddings: Single vector representing entire document (e.g., Doc2Vec, BERT).

These representations capture different aspects of text (frequency, semantics, context) and are chosen based on the specific task and available resources.

## Give a short history of Transformer models?
- Introduced in the paper "Attention is All You Need" by Vaswani et al. (June 2017)
- Replaced recurrent neural networks (RNNs) and convolutional neural networks (CNNs) in many NLP tasks
- June 2018 - GPT (Generative Pre-trained Transformer) by OpenAI
- October 2018 - BERT (Bidirectional Encoder Representations from Transformers) by Google
- February 2019 - GPT-2 by OpenAI
- October 2019 - DistilBERT by Hugging Face
- November 2019 - T5 (Text-to-Text Transfer Transformer) by Google and BART (Bidirectional and Auto-Regressive Transformers) by Facebook
- May 2020 - GPT-3 by OpenAI (175 billion parameters)

## Different kinds of transformer models?

There are broadly 3 types of transformer models:
1. **Auto-regressive models**: Generate output sequentially (e.g., GPT, GPT-2, GPT-3).
2. **Auto-encoder models**: Encode and decode sequences (e.g., BERT, RoBERTa, DistilBERT).
3. **Seq2Seq models**: Translate sequences from one domain to another (e.g., T5, BART).







## How do Transformers work?

Transformers are a type of neural network architecture that uses self-attention mechanisms to process sequences of data. Key components include:
- Multi-head self-attention: Attend to different parts of the input sequence simultaneously.
- Position-wise feedforward networks: Apply non-linear transformations to each position independently.
- Layer normalization and residual connections: Stabilize training and facilitate deeper networks.
- Encoder-decoder architecture: Used in tasks like machine translation, summarization, and question answering.

Transformers have revolutionized natural language processing tasks by capturing long-range dependencies and contextual information effectively.

