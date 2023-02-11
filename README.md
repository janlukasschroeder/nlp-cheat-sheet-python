
# NLP Cheat Sheet - Introduction - Overview - Python - Starter Kit

Introduction to Natural Language Processing (NLP) tools, frameworks, concepts, resources for Python

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/janlukasschroeder/nlp-cheat-sheet-python/blob/master/NLP-Cheat-Sheet.ipynb)



# NLP Python Libraries

- [🤗 Models & Datasets](https://huggingface.co/) - includes all state-of-the models like BERT and datasets like CNN news
- [spacy](https://spacy.io/) - NLP library with out-of-the box Named Entity Recognition, POS tagging, tokenizer and more
- [NLTK](https://www.nltk.org/) - similar to spacy, simple GUI model download `nltk.download()`
- [gensim](https://radimrehurek.com/gensim) - topic modelling, accessing corpus, similarity calculations between query and indexed docs, SparseMatrixSimilarity, Latent Semantic Analysis
- [lexnlp](https://github.com/LexPredict/lexpredict-lexnlp) - information retrieval and extraction for real, unstructured legal text
- [Holmes](https://github.com/msg-systems/holmes-extractor#the-basic-idea) - information extraction, document classification, search in documents
- [fastText](https://fasttext.cc/) - library for efficient text classification and representation learning
- [Stanford's Open IE](https://nlp.stanford.edu/software/openie.html) - information extraction of relation tuples from plain text, such as (Mark Zuckerberg; founded; Facebook). "Barack Obama was born in Hawaii" would create a triple (Barack Obama; was born in; Hawaii). [Open IE in Python](https://github.com/philipperemy/stanford-openie-python).

# NLP Models

## BERT & Transformer Models
- [BERT](https://github.com/google-research/bert)
- [Alibaba’s StructBERT](https://github.com/alibaba/AliceMind)
- [FinBERT](https://github.com/ProsusAI/finBERT) - analyze sentiment of financial text
- [DistilBERT](https://arxiv.org/abs/1910.01108)
- [RoBERTa](https://arxiv.org/abs/1907.11692)
- [VideoBERT](https://arxiv.org/pdf/1904.01766.pdf)
- [XLnet](https://github.com/zihangdai/xlnet)

## GPT Variants
- [GPT-2](https://openai.com/blog/better-language-models/)
- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) - open source version of GPT

## Other Models
- [Flan-T5](https://arxiv.org/pdf/2210.11416.pdf) (most powerful state-of-the art model as of writing)
- [Vega v2](https://arxiv.org/abs/2212.01853)
- [ERNIE](https://github.com/PaddlePaddle/ERNIE/)
- [Microsoft's Turing URL v6](https://arxiv.org/abs/2210.14867)
- [Google's T5](https://github.com/google-research/text-to-text-transfer-transformer)
- [Flair](https://github.com/flairNLP/flair)
- [Whisper](https://github.com/openai/whisper) - speech-to-text transcription 
- [Exhaustive list of models (text, vision, audio) and systems](https://docs.google.com/spreadsheets/d/1AAIebjNsnJj_uKALHbXNfn3_YsT6sHXtCU0q7OIPuc4/edit#gid=0)

Uncased model is better unless you know that case information is important for your task (e.g., Named Entity Recognition or Part-of-Speech tagging)

# NLP Tasks

## Text Generation
- Text summarization, e.g. summarize an earnings call
- Question answering, e.g. a chatbot answering simple customer questions
- Google Ads copy generator, e.g. provide a text to the model for it to generate a Google Ad copy
- Translation
- Synonym finder

## Text Classifcation
- Sentiment analysis, e.g. assign sentiment (`positive`, `neutral`, `negative`) to a product review
- Support ticket classification, e.g. assign classes (`bug`, `feature request`) to a customer support ticket
- Document classification, e.g. find articles matching a search query
- Fact checking

## In-Text Analysis
- Spell checking
- Named entity recognition
- Part-of speech tagging

#  Framworks

- PyTorch
- TensorFlow
- Keras

## Training & ML Frameworks

- [Uber's Horovod](https://horovod.readthedocs.io/en/stable/): distributed deep learning training framework for TensorFlow, Keras, and PyTorch 
- [Amazon's SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html): distributed training libraries
- [DeepSpeed](https://www.deepspeed.ai/)
- [Uber's Michelangelo](https://www.uber.com/en-DE/blog/michelangelo-machine-learning-platform/)
- [Uber's Ludwig](https://ludwig.ai/latest/)
- [Google's TFX](https://www.tensorflow.org/tfx)
- [H2O AutoML](https://docs.h2o.ai/): you don’t need to specify the model structure or hyperparameters. It experiments with multiple model architectures and picks out the best model given the features and the task.
- [Google's AutoML](https://cloud.google.com/automl): for developers with limited machine learning expertise to train high-quality models.



# Datasets

- [Gutenberg Corpus](https://block.pglaf.org/germany.shtml) - contains 25,000 free electronic books. `from nltk.corpus import gutenberg`
- [OntoNotes 5](https://github.com/ontonotes/conll-formatted-ontonotes-5.0) - corpus comprising various genres of text (news, conversational telephone speech, weblogs, usenet newsgroups, broadcast, talk shows) in three languages (English, Chinese, and Arabic) with structural information (syntax and predicate argument structure) and shallow semantics (word sense linked to an ontology and coreference).
- [wiki_en_tfidf.mm in gensim](https://radimrehurek.com/gensim/wiki.html#latent-semantic-analysis) 3.9M documents, 100K features (distinct tokens) and 0.76G non-zero entries in the sparse TF-IDF matrix. The Wikipedia corpus contains about 2.24 billion tokens in total.
- [GPT-2 Dataset](https://github.com/openai/gpt-2-output-dataset)
- [Brown corpus](http://icame.uib.no/brown/bcm-los.html) - contains text from 500 sources, and the sources have been categorized by genre, such as news, editorial, and so on.
- [Reuters Corpus - 10,788 news documents totaling 1.3 million words](https://www.nltk.org/book/ch02.html)
- [Newsfilter.io stock market news corpus](https://developers.newsfilter.io/) - contains over 4 million press releases, earnings reports, FDA drug approvals, analyst ratings, merger agreements and many more covering all US companies listed on NASDAQ, NYSE, AMEX
- [Kaggle - All the news, 143K articles](https://www.kaggle.com/snapcrack/all-the-news)
- [Kaggle - Daily news for stock market prediction](https://www.kaggle.com/aaron7sun/stocknews)
- [CNN News](https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ)
- [AG News - PyTorch integrated](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
- [CommonCrawl News Dataset](https://commoncrawl.org/2016/10/news-dataset-available/)

# Benchmarks

Evaluate models performance on language understanding tasks, such as question answering and text summarization.

- [Glue](https://gluebenchmark.com/leaderboard)
- [SuperGlue](https://super.gluebenchmark.com/leaderboard)

# Other Resources

[PapersWithCode](https://paperswithcode.com/task/text-classification)



# Starting with Spacy

spacy (good for beginners; use NLTK for bigger projects)

```shell
pip install spacy
python -m spacy download en 
# python -m spacy download en_core_web_lg
```
LexNLP (good for dealing with legal and financial documents; [installation guide here](https://github.com/LexPredict/lexpredict-contraxsuite/blob/master/documentation/Installation%20and%20Configuration/Installation%20and%20Configuration%20Guide.pdf))
```shell
pip install https://github.com/LexPredict/lexpredict-lexnlp/archive/master.zip
python # to open REPL console
>>> import nltk
>>> nltk.download() # download all packages
```


# Concepts
---


# Word embeddings (=word vectors)

Visualizing word vectors using PCA ([link to paper](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)).

![word-embs](https://www.tensorflow.org/images/linear-relationships.png)

- Word embeddings are vector representation of words and learned from your data.
- Example sentence: word embeddings are words converted into numbers.
- A word in this sentence may be “Embeddings” or “numbers ” etc.
- A dictionary may be the list of all unique words in the sentence, eg [‘Word’,’Embeddings’,’are’,’Converted’,’into’,’numbers’]
- It’s common to see word embeddings that are 256-dimensional, 512-dimensional, or 1,024-dimensional
- Another vector representation of a word may be a one-hot encoded vector where 1 stands for the position where the word exists and 0 everywhere else. 
- The geometric relationships between word vectors should reflect the semantic relationships between these words.
- Words meaning different things are embedded at points far away from each other, whereas related words are closer.
- For instance, by adding a “female” vector to the vector “king,” we obtain the vector “queen.” By adding a “plural” vector, we obtain “kings.”
- The is a "perfect" word-embedding space for each task, for example the perfect word-embedding space for a movie-review sentiment-analysis model may look different from the perfect space for a legal-document-classification model.
- Learn a new embedding space with every new task.


**Example: one-hot encoded vector**

- `numbers` word is represented as one-hot encoded vector = [0,0,0,0,0,1] 
- `converted` = [0,0,0,1,0,0]

**Example: word-embeddings**

![word-emb](https://i.imgur.com/6dNjnO7_d.webp?maxwidth=760&fidelity=grand)

## One-Hot Vectors vs Word Embeddings

![img](https://i.imgur.com/qJzBY8N_d.webp?maxwidth=760&fidelity=grand)

Credits: Deep Learning with Python


## Pre-trained word embeddings

There are two ways to obtain word embeddings: 

1. Learn word embeddings jointly with the main task you care about (such as document classification or sentiment prediction). In this setup, you start with random word vectors and then learn word vectors in the same way you learn the weights of a neural network.
2. Load into your model word embeddings that were precomputed using a different machine-learning task than the one you’re trying to solve. These are called pretrained word embeddings.

**Pretrained word embeddings**

- [Word2Vec (Google, 2013)](https://code.google.com/archive/p/word2vec/), uses Skip Gram and CBOW
- [Vectors trained on Google News (1.5GB)](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) - vocabulary of 3 million words trained on around 100 billion words from the google news dataset
- [GloVe (Stanford)](https://nlp.stanford.edu/projects/glove), contains 100-dimensional embedding vectors for 400,000 words
- [Stanford Named Entity Recognizer (NER)](https://nlp.stanford.edu/software/CRF-NER.shtml)
- [LexPredict: pre-trained word embedding models for legal or regulatory text](https://github.com/LexPredict/lexpredict-lexnlp)
- [LexNLP legal models](https://github.com/LexPredict/lexpredict-legal-dictionary) - US GAAP, finaical common terms, US federal regulators, common law

> When parts of a model are pretrained (like your Embedding layer) and parts are randomly initialized (like your classifier), the pretrained parts shouldn’t be updated during training, to avoid forgetting what they already know.

## Comparison of Embedding Models

How OpenAI GPT-3 embeddings compare to Google and Sentence-Transformer embeddings.


![image.png](https://pbs.twimg.com/media/FKLqFkuXEAY5Ehq?format=png)

[Credit](https://twitter.com/Nils_Reimers/status/1487014195568775173?s=20&t=cXD8dnwWSbK_2fgw15yz5w)

### Embedding with SentenceTransformers

[SentenceTransformers](https://www.sbert.net/) is a Python framework for state-of-the-art sentence, text and image embeddings. The initial work is described in the paper [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084).


Install it with:



```python
pip install -U sentence-transformers
```


```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sentences to encode
sentences = [
  "This framework generates embeddings for each input sentence.",
  "Sentences are passed as a list of string.",
  "The quick brown fox jumps over the lazy dog."
]

embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding[:15])
    print("Embedding dimension", embedding.shape)
    print("")
```

    Sentence: This framework generates embeddings for each input sentence.
    Embedding: [-0.01195314 -0.05562933 -0.00824256  0.00889048  0.02768425  0.1139881
      0.01469875 -0.03189586  0.04145184 -0.08188552  0.01413268 -0.0203336
      0.04077511  0.02262853 -0.04784386]
    Embedding dimension (384,)
    
    Sentence: Sentences are passed as a list of string.
    Embedding: [ 0.0564525   0.05500239  0.03137959  0.03394853 -0.03542475  0.08346675
      0.09888012  0.00727544 -0.00668658 -0.0076581   0.07937384  0.00073965
      0.01492921 -0.01510471  0.03676743]
    Embedding dimension (384,)
    
    Sentence: The quick brown fox jumps over the lazy dog.
    Embedding: [ 0.04393354  0.05893442  0.04817837  0.07754811  0.02674442 -0.03762956
     -0.0026051  -0.05994309 -0.002496    0.02207284  0.04802594  0.05575529
     -0.03894543 -0.0266168   0.0076934 ]
    Embedding dimension (384,)
    


### GloVe Embeddings with Keras

Fine-tuning pre-trained GloVe embeddings with Keras.



```python
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

# download from https://nlp.stanford.edu/projects/glove
f = open('./glove.6B.100d.txt')

embeddings_index = {}

for line in f:
  values = line.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_index[word] = coefs

f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 100
maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in word_index.items():
  if i < max_words:
    embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
# option A)
model.add(Flatten())
model.add(Dense(32, activation='relu'))
# option B) LSTM
#model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

model.save_weights('pre_trained_glove_model.h5')
```

You can also train the same model without loading the pretrained word embeddings and without freezing the embedding layer. In that case, you’ll learn a task-specific embedding of the input tokens, which is generally more powerful than pretrained word embeddings when lots of data is available.

## Universal Sentence Encoder in TensorFlow (by Google)

The Universal Sentence Encoder ([Cer et al., 2018](https://arxiv.org/pdf/1803.11175.pdf)) (USE) is a model that encodes text into 512-dimensional embeddings.

[Source](https://github.com/tensorflow/tfjs-models/tree/master/universal-sentence-encoder)


```python
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

def embed(input):
  return model(input)

word = "Elephant"
sentence = "I am a sentence for which I would like to get its embedding."
paragraph = (
    "Universal Sentence Encoder embeddings also support short paragraphs. "
    "There is no hard limit on how long the paragraph is. Roughly, the longer "
    "the more 'diluted' the embedding will be.")

messages = [word, sentence, paragraph]

message_embeddings = embed(messages)

for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
  print("Message: {}".format(messages[i]))
  print("Embedding size: {}".format(len(message_embedding)))
  message_embedding_snippet = ", ".join(
      (str(x) for x in message_embedding[:3]))
  print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
```

    Message: Elephant
    Embedding size: 512
    Embedding: [0.008344486355781555, 0.00048085825983434916, 0.06595248728990555, ...]
    
    Message: I am a sentence for which I would like to get its embedding.
    Embedding size: 512
    Embedding: [0.050808604806661606, -0.016524329781532288, 0.01573779620230198, ...]
    
    Message: Universal Sentence Encoder embeddings also support short paragraphs. There is no hard limit on how long the paragraph is. Roughly, the longer the more 'diluted' the embedding will be.
    Embedding size: 512
    Embedding: [-0.028332693502306938, -0.0558621808886528, -0.012941480614244938, ...]
    


## Create word vectors yourself

```python
import gensim
word2vev_model = gensim.models.word2vec.Word2Vec(sentence_list)
```

[Source](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/)

## How to create word vectors?
- Count-based methods compute the statistics of how often some word co-occurs with its neighbor words in a large text corpus, and then map these count-statistics down to a small, dense vector for each word. 
- Predictive models directly try to predict a word from its neighbors in terms of learned small, dense embedding vectors (considered parameters of the model).
  - Example: Word2vec (Google)

### 1. Count based word embeddings

#### Count Vector (= Document Term Matrix)

![img](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/04164920/count-vector.png)



#### TF-IDF

Term Frequency - Inverse Document Frequency. The goal is to determine a TF-IDF vector for each document. The vectors are then used to calculate the similarity between documents.

- Term frequency (TF) is the number of times a word appears in a document divided by the total number of words in the document. 
- Inverse document frequency (IDF) = importance of the term across a corpus. It calculates the weight of rare words in all documents in the corpus, with rare words having a high IDF score, and words that are present in all documents (e.g. `a`, `the`, `is`) having IDF close to zero.

The TF is calculated for a term `t` in a document `d`. Hence, every term in every document has a TF and we need to calculate TFs for every term in every document.

![img](https://latex.codecogs.com/png.image?\small&space;\dpi{150}TF(t,&space;d)&space;=&space;\frac{\text{Number&space;of&space;occurences&space;of&space;term&space;\textit{t}&space;in&space;doc&space;\textit{d}}}{\text{Total&space;number&space;of&space;terms&space;in&space;doc&space;\textit{d}}})



The IDF score is calculated once for each term `t` occuring in the  corpus.

![img](https://i.imgur.com/Vy0ocKG_d.webp?maxwidth=760&fidelity=grand)

Combining these two, we get the TF-IDF score `w` for a term `t` in a document `d`:


![img](https://latex.codecogs.com/png.image?\small&space;\dpi{150}w(t,&space;d)&space;=&space;TF(t,&space;d)&space;*&space;IDF(t))

(sklearn) in Python has a function `TfidfVectorizer()` that will compute the TF-IDF values for you.

##### Example 1

The document size of our corpus is N=4.

![image.png](data:image/png;base64,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)

The 6 unique terms in our corpus are `dog`, `bites`, `man`, `eats`, `meat`, `food`.

Let's determine the TF-IDF scores `w` for all terms in the first document `Dog bites man.` 


Terms|TF |IDF Score|w = TF * IDF
--   |--|--|--------|
dog| 1/3|log(4/3)|0.138
bites| 1/3|log(4/2)|0.33
man| 1/3|log(4/3)|0.138
eats| 0 |log(4/2)|0
meat| 0 |log(4/1)|0
food| 0 |log(4/1)|0

The corresponding TF-IDF vector for document `D1` is:

`[0.138, 0.33, 0.138, 0, 0, 0]`


##### Example 2

Consider a document containing 100 words wherein the word cat appears 3 times. The term frequency (TF) for cat is then (3 / 100) = 0.03. 

Now, assume we have 10 million documents and the word cat appears in one thousand of these. Then, the inverse document frequency (IDF) is calculated as log(10,000,000 / 1,000) = 4. 

Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12

#### TD-IDF Implementation in Python 


```python
from sklearn.feature_extraction.text import TfidfVectorizer

document_corpus = [
  "Dog bites man", 
  "Man bites dog", 
  "Dog eats meat", 
  "Man eats food"
]

tfidf = TfidfVectorizer()
bow_rep_tfidf = tfidf.fit_transform(document_corpus)

print("IDF for all words in the vocabulary")
print(tfidf.idf_)
print("\nAll words in the vocabulary.")
print(tfidf.get_feature_names_out())

temp = tfidf.transform(["Dog bites man"])

print("\nTF-IDF representation for 'Dog bites man':\n", temp.toarray())
```

    IDF for all words in the vocabulary
    [1.51082562 1.22314355 1.51082562 1.91629073 1.22314355 1.91629073]
    
    All words in the vocabulary.
    ['bites' 'dog' 'eats' 'food' 'man' 'meat']
    
    TF-IDF representation for 'Dog bites man':
     [[0.65782931 0.53256952 0.         0.         0.53256952 0.        ]]


Notice that the TF-IDF scores that we calculated for our corpus doesn't match the TF-IDF scores given by scikit-learn. This is because scikit-learn uses a slightly modified version of the IDF formula. This stems from provisions to account for possible zero divisions and to not entirely ignore terms that appear in all documents.

TD-IDF with N-Gram


```python
from sklearn.feature_extraction.text import TfidfVectorizer
import re

document_corpus = [
  "Dog bites man", 
  "Man bites dog", 
  "Dog eats meat", 
  "Man eats food"
]

# Write a function for cleaning strings and returning an array of ngrams
def ngrams_analyzer(string):
    string = re.sub(r'[,-./]', r'', string)
    ngrams = zip(*[string[i:] for i in range(5)])  # N-Gram length is 5
    return [''.join(ngram) for ngram in ngrams]

# Construct your vectorizer for building the TF-IDF matrix
tfidf = TfidfVectorizer(analyzer=ngrams_analyzer)

bow_rep_tfidf = tfidf.fit_transform(document_corpus)

print("IDF for all words in the vocabulary")
print(tfidf.idf_)
print("\nAll words in the vocabulary.")
print(tfidf.get_feature_names_out())

temp = tfidf.transform(["Dog bites man"])

print("\nTF-IDF representation for 'Dog bites man':\n", temp.toarray())

# Credits: https://towardsdatascience.com/group-thousands-of-similar-spreadsheet-text-cells-in-seconds-2493b3ce6d8d
```

    IDF for all words in the vocabulary
    [1.51082562 1.51082562 1.91629073 1.91629073 1.91629073 1.91629073
     1.91629073 1.91629073 1.91629073 1.91629073 1.91629073 1.91629073
     1.51082562 1.51082562 1.91629073 1.91629073 1.91629073 1.91629073
     1.51082562 1.91629073 1.91629073 1.91629073 1.91629073 1.91629073
     1.91629073 1.91629073 1.91629073 1.91629073 1.91629073 1.91629073
     1.91629073]
    
    All words in the vocabulary.
    [' bite' ' eats' ' food' ' meat' 'Dog b' 'Dog e' 'Man b' 'Man e' 'an bi'
     'an ea' 'ats f' 'ats m' 'bites' 'eats ' 'es do' 'es ma' 'g bit' 'g eat'
     'ites ' 'n bit' 'n eat' 'og bi' 'og ea' 's dog' 's foo' 's man' 's mea'
     'tes d' 'tes m' 'ts fo' 'ts me']
    
    TF-IDF representation for 'Dog bites man':
     [[0.28113163 0.         0.         0.         0.35657982 0.
      0.         0.         0.         0.         0.         0.
      0.28113163 0.         0.         0.35657982 0.35657982 0.
      0.28113163 0.         0.         0.35657982 0.         0.
      0.         0.35657982 0.         0.         0.35657982 0.
      0.        ]]


#### Co-Occurrence Vector/Matrix

Words that are similar to each other will tend to co-occur together. 

Let’s call the context of the word, the two words that surround a specific word by each side. For example, in a sentence `I ate a peach yesterday`, the word `peach` is surrounded by the words: `ate`, `a`, `yesterday`.

To build a co-occurrence matrix, one has to start with the full vocabulary of words in a specific corpus.

**Example**

Let’s imagine some simple sentences:
- I’m riding in my car to the beach.
- I’m riding in my jeep to the beach.
- My car is a jeep.
- My jeep is a car.
- I ate a banana.
- I ate a peach.

The vocabulary of our group of sentences is:

`a, ate, banana, beach, car, in, is, I’m, jeep, my, riding, to, the`

Our co-occurence vector will be of of size 13, where 13 is the number of distinct words in our vocabulary.

The initialized co-occurence vector for the word `car` is:

`[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`

In our sentences, the word `car` shows up in 3 sentences:

- I’m riding **in my** car **to the** beach.
- **My** car **is a** jeep.
- My jeep **is a** car.

The highlighted words co-occur with the word `car`, i.e. the highlights represent the two words before and two words after `car`.

The co-occurence vector for `car` is: 

```
# Vocabulary + co-occurence vector below
 a, ate, banana, beach, car, in, is, I’m, jeep, my, riding, to, the
[2,   0,      0,     0,   0,  1,  2,   0,    0,  2,      0,  1,   1]
```

Each number represents the number of occurences in the context of the word. For example, `a` appears twice, whereas `ate` didn't appear at all.

[Credits](https://towardsdatascience.com/word-vectors-intuition-and-co-occurence-matrixes-a7f67cae16cd)

### 2. Prediction based word embeddings

- Uses Neural Networks
- CBOW predicts target words (e.g. 'mat') from source context words ('the cat sits on the')
- Skip-gram does the inverse and predicts source context-words from the target words

#### CBOW (Continuous Bag of words)

<img src="https://www.tensorflow.org/images/softmax-nplm.png" width="400">

#### Skip Gram

Skip – gram follows the same topology as of CBOW. It just flips CBOW’s architecture on its head. The aim of skip-gram is to predict the context given a word

<img src="https://www.tensorflow.org/images/nce-nplm.png" width="400">

#### Outcome

![out](https://github.com/sanketg10/deep-learning-repo/raw/6b207e326cc937930a0512a8c599e86e48c297b2/embeddings/assets/skip_gram_net_arch.png)

## Bag of Words


```python
# John likes to watch movies. Mary likes movies too.
BoW1 = {"John":1,"likes":2,"to":1,"watch":1,"movies":2,"Mary":1,"too":1};
```

# spacy


```python
import spacy
```


```python
# Import dataset
nlp = spacy.load("en")
# Import large dataset. Needs to be downloaded first.
# nlp = spacy.load("en_core_web_lg")
```

# Stop Words

Stop words are the very common words like ‘if’, ‘but’, ‘we’, ‘he’, ‘she’, and ‘they’.
We can usually remove these words without changing the semantics of a text and doing so often (but not always) improves the performance of a model.


```python
# spacy: Removing stop words
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

print('spacy: Number of stop words: %d' % len(spacy_stopwords))
```

    spacy: Number of stop words: 326



```python
# nltk: Removing stop words 
from nltk.corpus import stopwords
english_stop_words = stopwords.words('english')

print('ntlk: Number of stop words: %d' % len(english_stop_words))
```

    ntlk: Number of stop words: 179



```python
text = 'Larry Page founded Google in early 1990.'
doc = nlp(text)
tokens = [token.text for token in doc if not token.is_stop]
print('Original text: %s' % (text))
print()
print(tokens)
```

    Original text: Larry Page founded Google in early 1990.
    
    ['Larry', 'Page', 'founded', 'Google', 'early', '1990', '.']


# Spans
Part of a given text. So doc[2:4] is a span starting at token 2, up to – but not including! – token 4.

Docs: https://spacy.io/api/span


```python
doc = nlp("Larry Page founded Google in early 1990.")
span = doc[2:4]
span.text
```




    'founded Google'




```python
[(spans) for spans in doc]
```




    [Larry, Page, founded, Google, in, early, 1990, .]



# Token and Tokenization

Segmenting text into words, punctuation etc.
- Sentence tokenization
- Word tokenization

Docs: https://spacy.io/api/token


```python
doc = nlp("Larry Page founded Google in early 1990.")
[token.text for token in doc]
```




    ['Larry', 'Page', 'founded', 'Google', 'in', 'early', '1990', '.']




```python
# Load OpenAI GPT-2 using PyTorch Transformers
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
# https://huggingface.co/pytorch-transformers/serialization.html
```

# Tokenizers

- Byte-Pair Encoding (used by GPT-2)
- WordPiece (used by BERT)
- Unigram (used by T5)

[SentencePiece](https://github.com/google/sentencepiece) is a tokenization algorithm for the preprocessing of text.

[Unicode normalization](http://www.unicode.org/reports/tr15/) (such as NFC or NFKC), can also be applied by tokenizer.

HTML tokenizers 
- by W3: https://www.w3.org/TR/2011/WD-html5-20110113/tokenization.html
- by spact: https://pypi.org/project/spacy-html-tokenizer/

## Tokenization Process

![tokenizer](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/tokenization_pipeline.svg)

[Credits](https://huggingface.co/course/chapter6/4?fw=pt)

# Chunks and Chunking

Segments and labels multi-token sequences.

- Each of these larger boxes is called a chunk. 
- Like tokenization, which omits whitespace, chunking usually selects a subset of the tokens. 
- The pieces produced by a chunker do not overlap in the source text.

<img src="https://www.nltk.org/images/chunk-segmentation.png" width="400">
<center>Segmentation and Labeling at both the Token and Chunk Levels</center>

<img src="https://www.nltk.org/images/chunk-tagrep.png" width="400">
<center>Tag Representation of Chunk Structures</center>

<img src="https://www.nltk.org/images/chunk-treerep.png" width="400">
<center>Tree Representation of Chunk Structures</center>

Credits: https://www.nltk.org/book/ch07.html

# Chinks and Chinking

Chink is a sequence of tokens that is not included in a chunk.

Credits: https://www.nltk.org/book/ch07.html

# Part-of-speech (POS) Tagging

Assigning word types to tokens like verb or noun.

POS tagging should be done straight after tokenization and before any words are removed so that sentence structure is preserved and it is more obvious what part of speech the word belongs to.




```python
text = "Asian shares skidded on Tuesday after a rout in tech stocks put Wall Street to the sword"
doc = nlp(text)
[(x.orth_, x.pos_, spacy.explain(x.pos_)) for x in [token for token in doc]]
```




    [('Asian', 'ADJ', 'adjective'),
     ('shares', 'NOUN', 'noun'),
     ('skidded', 'VERB', 'verb'),
     ('on', 'ADP', 'adposition'),
     ('Tuesday', 'PROPN', 'proper noun'),
     ('after', 'ADP', 'adposition'),
     ('a', 'DET', 'determiner'),
     ('rout', 'NOUN', 'noun'),
     ('in', 'ADP', 'adposition'),
     ('tech', 'NOUN', 'noun'),
     ('stocks', 'NOUN', 'noun'),
     ('put', 'VERB', 'verb'),
     ('Wall', 'PROPN', 'proper noun'),
     ('Street', 'PROPN', 'proper noun'),
     ('to', 'ADP', 'adposition'),
     ('the', 'DET', 'determiner'),
     ('sword', 'NOUN', 'noun')]




```python
[(x.orth_, x.tag_, spacy.explain(x.tag_)) for x in [token for token in doc]]
```




    [('Asian', 'JJ', 'adjective'),
     ('shares', 'NNS', 'noun, plural'),
     ('skidded', 'VBD', 'verb, past tense'),
     ('on', 'IN', 'conjunction, subordinating or preposition'),
     ('Tuesday', 'NNP', 'noun, proper singular'),
     ('after', 'IN', 'conjunction, subordinating or preposition'),
     ('a', 'DT', 'determiner'),
     ('rout', 'NN', 'noun, singular or mass'),
     ('in', 'IN', 'conjunction, subordinating or preposition'),
     ('tech', 'NN', 'noun, singular or mass'),
     ('stocks', 'NNS', 'noun, plural'),
     ('put', 'VBD', 'verb, past tense'),
     ('Wall', 'NNP', 'noun, proper singular'),
     ('Street', 'NNP', 'noun, proper singular'),
     ('to', 'IN', 'conjunction, subordinating or preposition'),
     ('the', 'DT', 'determiner'),
     ('sword', 'NN', 'noun, singular or mass')]




```python
# using nltk
import nltk

tokens = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)
pos_tags
```




    [('Asian', 'JJ'),
     ('shares', 'NNS'),
     ('skidded', 'VBN'),
     ('on', 'IN'),
     ('Tuesday', 'NNP'),
     ('after', 'IN'),
     ('a', 'DT'),
     ('rout', 'NN'),
     ('in', 'IN'),
     ('tech', 'JJ'),
     ('stocks', 'NNS'),
     ('put', 'VBD'),
     ('Wall', 'NNP'),
     ('Street', 'NNP'),
     ('to', 'TO'),
     ('the', 'DT'),
     ('sword', 'NN')]



## BILUO tagging

- BEGIN - The first token of a multi-token entity.
- IN - An inner token of a multi-token entity.
- LAST - The final token of a multi-token entity.
- UNIT - A single-token entity.
- OUT - A non-entity token.


```python
[(token, token.ent_iob_, token.ent_type_) for token in doc]
```




    [(Asian, 'B', 'NORP'),
     (shares, 'O', ''),
     (skidded, 'O', ''),
     (on, 'O', ''),
     (Tuesday, 'B', 'DATE'),
     (after, 'O', ''),
     (a, 'O', ''),
     (rout, 'O', ''),
     (in, 'O', ''),
     (tech, 'O', ''),
     (stocks, 'O', ''),
     (put, 'O', ''),
     (Wall, 'O', ''),
     (Street, 'O', ''),
     (to, 'O', ''),
     (the, 'O', ''),
     (sword, 'O', '')]



# Stemming

Stemming is the process of reducing words to their root form.

Examples:
- cats, catlike, catty → cat
- fishing, fished, fisher → fish

There are two types of stemmers in NLTK: [Porter Stemmer](https://tartarus.org/martin/PorterStemmer/) and [Snowball stemmers](https://tartarus.org/martin/PorterStemmer/)

[Credits](https://stackabuse.com/python-for-nlp-tokenization-stemming-and-lemmatization-with-spacy-library/)


```python
import nltk
from nltk.stem.porter import *

stemmer = PorterStemmer()
tokens = ['compute', 'computer', 'computed', 'computing']
for token in tokens:
    print(token + ' --> ' + stemmer.stem(token))
```

    compute --> comput
    computer --> comput
    computed --> comput
    computing --> comput


# Lemmatization

Assigning the base form of word, for example: 
- "was" → "be"
- "rats" → "rat"


```python
doc = nlp("Was Google founded in early 1990?")
[(x.orth_, x.lemma_) for x in [token for token in doc]]
```




    [('Was', 'be'),
     ('Google', 'Google'),
     ('founded', 'found'),
     ('in', 'in'),
     ('early', 'early'),
     ('1990', '1990'),
     ('?', '?')]



# Sentence Detection

Finding and segmenting individual sentences.


```python
doc = nlp("Larry Page founded Google in early 1990. Sergey Brin joined.")
[sent.text for sent in doc.sents]
```




    ['Larry Page founded Google in early 1990.', 'Sergey Brin joined.']



# Dependency Parsing	

Assigning syntactic dependency labels, describing the relations between individual tokens, like subject or object.


```python
doc = nlp("We are reading a text.")
# Dependency labels
[(x.orth_, x.dep_, spacy.explain(x.dep_)) for x in [token for token in doc]]
```




    [('We', 'nsubj', 'nominal subject'),
     ('are', 'aux', 'auxiliary'),
     ('reading', 'ROOT', None),
     ('a', 'det', 'determiner'),
     ('text', 'dobj', 'direct object'),
     ('.', 'punct', 'punctuation')]




```python
# Syntactic head token (governor)
[token.head.text for token in doc]
```




    ['reading', 'reading', 'reading', 'text', 'reading', 'reading']



# Base noun phrases



```python
doc = nlp("I have a red car")
[chunk.text for chunk in doc.noun_chunks]
```




    ['I', 'a red car']



# Named Entity Recognition (NER)

What is NER? Labeling "real-world" objects, like persons, companies or locations.

2 popular approaches:
- Rule-based
- ML-based:
    - Multi-class classification
    - Conditional Random Field (probabilistic graphical model)

Datasets:
- [Kaggle, IOB, POS tags](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/home)

Credits: https://medium.com/@yingbiao/ner-with-bert-in-action-936ff275bc73


Entities supported by spacy:
- PERSON	People, including fictional.
- NORP	Nationalities or religious or political groups.
- FAC	Buildings, airports, highways, bridges, etc.
- ORG	Companies, agencies, institutions, etc.
- GPE	Countries, cities, states.
- LOC	Non-GPE locations, mountain ranges, bodies of water.
- PRODUCT	Objects, vehicles, foods, etc. (Not services.)
- EVENT	Named hurricanes, battles, wars, sports events, etc.
- WORK_OF_ART	Titles of books, songs, etc.
- LAW	Named documents made into laws.
- LANGUAGE	Any named language.
- DATE	Absolute or relative dates or periods.
- TIME	Times smaller than a day.
- PERCENT	Percentage, including ”%“.
- MONEY	Monetary values, including unit.
- QUANTITY	Measurements, as of weight or distance.
- ORDINAL	“first”, “second”, etc.
- CARDINAL	Numerals that do not fall under another type.

## Alternatives to spacy

[LexNLP](https://lexpredict-lexnlp.readthedocs.io/en/latest/modules/extract/extract.html#pattern-based-extraction-methods) entities:
- acts, e.g., “section 1 of the Advancing Hope Act, 1986”
- amounts, e.g., “ten pounds” or “5.8 megawatts”
- citations, e.g., “10 U.S. 100” or “1998 S. Ct. 1”
- companies, e.g., “Lexpredict LLC”
- conditions, e.g., “subject to …” or “unless and until …”
- constraints, e.g., “no more than” or “
- copyright, e.g., “(C) Copyright 2000 Acme”
- courts, e.g., “Supreme Court of New York”
- CUSIP, e.g., “392690QT3”
- dates, e.g., “June 1, 2017” or “2018-01-01”
- definitions, e.g., “Term shall mean …”
- distances, e.g., “fifteen miles”
- durations, e.g., “ten years” or “thirty days”
- geographic and geopolitical entities, e.g., “New York” or “Norway”
- money and currency usages, e.g., “$5” or “10 Euro”
- percents and rates, e.g., “10%” or “50 bps”
- PII, e.g., “212-212-2121” or “999-999-9999”
- ratios, e.g.,” 3:1” or “four to three”
- regulations, e.g., “32 CFR 170”
- trademarks, e.g., “MyApp (TM)”
- URLs, e.g., “http://acme.com/”

Stanford NER entities:
- Location, Person, Organization, Money, Percent, Date, Time

NLTK
- NLTK maximum entropy classifier

Transformer Models (on HuggingFace)
- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox)
- [BERT](https://huggingface.co/dslim/bert-base-NER?text=My+name+is+Sarah+and+I+live+in+London)
- [BERT-large](https://huggingface.co/dslim/bert-large-NER)
- [camemBERT](https://huggingface.co/Jean-Baptiste/camembert-ner)
- [Electra](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english)
- [NER-English-Large](https://huggingface.co/flair/ner-english-large) (best so far for org NER)
- [XML-RoBERTa](https://huggingface.co/xlm-roberta-large-finetuned-conll03-english)


```python
doc = nlp("Larry Page founded Google in the US in early 1990.")
# Text and label of named entity span
[(ent.text, ent.label_) for ent in doc.ents]
```




    [('Larry Page', 'PERSON'),
     ('Google', 'ORG'),
     ('US', 'GPE'),
     ('early 1990', 'DATE')]




```python
doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
[(X.text, X.label_) for X in doc.ents]
```




    [('European', 'NORP'),
     ('Google', 'ORG'),
     ('$5.1 billion', 'MONEY'),
     ('Wednesday', 'DATE')]




```python
from collections import Counter

labels = [x.label_ for x in doc.ents]
Counter(labels)
```




    Counter({'NORP': 1, 'ORG': 1, 'MONEY': 1, 'DATE': 1})




```python
[(X, X.ent_iob_, X.ent_type_) for X in doc]
```




    [(European, 'B', 'NORP'),
     (authorities, 'O', ''),
     (fined, 'O', ''),
     (Google, 'B', 'ORG'),
     (a, 'O', ''),
     (record, 'O', ''),
     ($, 'B', 'MONEY'),
     (5.1, 'I', 'MONEY'),
     (billion, 'I', 'MONEY'),
     (on, 'O', ''),
     (Wednesday, 'B', 'DATE'),
     (for, 'O', ''),
     (abusing, 'O', ''),
     (its, 'O', ''),
     (power, 'O', ''),
     (in, 'O', ''),
     (the, 'O', ''),
     (mobile, 'O', ''),
     (phone, 'O', ''),
     (market, 'O', ''),
     (and, 'O', ''),
     (ordered, 'O', ''),
     (the, 'O', ''),
     (company, 'O', ''),
     (to, 'O', ''),
     (alter, 'O', ''),
     (its, 'O', ''),
     (practices, 'O', '')]




```python
# Show Begin and In entities
items = [x.text for x in doc.ents]
print(items)
Counter(items).most_common(3)
```

    ['European', 'Google', '$5.1 billion', 'Wednesday']





    [('European', 1), ('Google', 1), ('$5.1 billion', 1)]




```python
import lexnlp.extract.en as lexnlp
import nltk
```


```python
text = "There are ten cows in the 2 acre pasture."
print(list(lexnlp.amounts.get_amounts(text)))
```

    [10, 2.0]



```python
import lexnlp.extract.en.acts
text = "test section 12 of the VERY Important Act of 1954."
lexnlp.extract.en.acts.get_act_list(text)
```




    [{'location_start': 5,
      'location_end': 49,
      'act_name': 'VERY Important Act',
      'section': '12',
      'year': '1954',
      'ambiguous': False,
      'value': 'section 12 of the VERY Important Act of 1954'}]



## NER with flair's NER-English-Large

Source: https://huggingface.co/flair/ner-english-large

Available tags:
- PER, person
- LOC, location
- ORG, organization
- MISC, other name


```python
pip install flair
```


```python
from flair.data import Sentence
from flair.models import SequenceTagger

# load tagger
tagger = SequenceTagger.load("flair/ner-english-large")
```


```python
# make example sentence
sentence = Sentence("George Washington went to Washington")

# predict NER tags
tagger.predict(sentence)

# print sentence
print(sentence)

# print predicted NER spans
print('The following NER tags are found:')

# iterate over entities and print
for entity in sentence.get_spans('ner'):
    print(entity)
```

    Sentence: "George Washington went to Washington" → ["George Washington"/PER, "Washington"/LOC]
    The following NER tags are found:
    Span[0:2]: "George Washington" → PER (1.0)
    Span[4:5]: "Washington" → LOC (1.0)



```python
text = "We are the platform of choice for customers' SAP workloads in the cloud, companies like Thabani, Munich Re's, Sodexo, Volvo Cars, all run SAP on Azure. We are the only cloud provider with direct and secure access to Oracle databases running an Oracle Cloud infrastructure, making it possible for companies like FedEx, GE, and Marriott to use capabilities from both companies. And with Azure Confidential Computing, we're enabling companies in highly regulated industries, including RBC, to bring their most sensitive applications to the cloud. Just last week, UBS said it will move more than 50% of its applications to Azure."

# make example sentence
sentence = Sentence(text)

# predict NER tags
tagger.predict(sentence)

# print sentence
print(sentence)

# print predicted NER spans
print('\nThe following NER tags are found:\n')

# iterate over entities and print
for entity in sentence.get_spans('ner'):
    print(entity)
```

    Sentence: "We are the platform of choice for customers' SAP workloads in the cloud , companies like Thabani , Munich Re 's , Sodexo , Volvo Cars , all run SAP on Azure . We are the only cloud provider with direct and secure access to Oracle databases running an Oracle Cloud infrastructure , making it possible for companies like FedEx , GE , and Marriott to use capabilities from both companies . And with Azure Confidential Computing , we 're enabling companies in highly regulated industries , including RBC , to bring their most sensitive applications to the cloud . Just last week , UBS said it will move more than 50 % of its applications to Azure ." → ["SAP"/ORG, "Thabani"/ORG, "Munich Re"/ORG, "Sodexo"/ORG, "Volvo Cars"/ORG, "SAP"/ORG, "Azure"/MISC, "Oracle"/ORG, "Oracle Cloud"/MISC, "FedEx"/ORG, "GE"/ORG, "Marriott"/ORG, "Azure Confidential Computing"/MISC, "RBC"/ORG, "UBS"/ORG, "Azure"/MISC]
    
    The following NER tags are found:
    
    Span[8:9]: "SAP" → ORG (0.9945)
    Span[16:17]: "Thabani" → ORG (1.0)
    Span[18:20]: "Munich Re" → ORG (0.9604)
    Span[22:23]: "Sodexo" → ORG (1.0)
    Span[24:26]: "Volvo Cars" → ORG (1.0)
    Span[29:30]: "SAP" → ORG (0.9995)
    Span[31:32]: "Azure" → MISC (0.9974)
    Span[45:46]: "Oracle" → ORG (0.9997)
    Span[49:51]: "Oracle Cloud" → MISC (1.0)
    Span[59:60]: "FedEx" → ORG (1.0)
    Span[61:62]: "GE" → ORG (1.0)
    Span[64:65]: "Marriott" → ORG (1.0)
    Span[74:77]: "Azure Confidential Computing" → MISC (0.999)
    Span[88:89]: "RBC" → ORG (1.0)
    Span[104:105]: "UBS" → ORG (1.0)
    Span[117:118]: "Azure" → MISC (0.9993)


# Text Classification

Two types:
- binary classification (text only belongs to one class)
- multi-class classification (text can belong to multiple classes)

Assigning categories or labels to a whole document, or parts of a document.

Approach:
- calculate document vectors for each document
- use kNN to calculate clusters based on document vectors
- each cluster represents a class of documents that are similar to each other


```python
# Credits: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
import re
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer

ag_news_label = {1 : "World",
                 2 : "Sports",
                 3 : "Business",
                 4 : "Sci/Tec"}

def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

vocab = train_dataset.get_vocab()
model = model.to("cpu")

print("This is a %s news" %ag_news_label[predict(ex_text_str, model, vocab, 2)])

# Output: This is a Sports news
```

## CountVectorizer 
- Convert a collection of text documents to a matrix of token counts
- [skikitLearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

# Text Similarity Metrics

We can calculate the similarity between two (and more):
- Characters: `A` vs `a`
- Words: `cat` vs `cats`
- Tokens: `co-chief` vs `-chief`
- Sentences: `I love cats` vs `I love dogs`
- Documents: a set of sentences

Two types of measures exist returing one of two outputs:
1. `True` or `False` (binary), i.e. two inputs are exactly the same or they are not - nothing in between.
2. Floating, i.e. 95% of the two given inputs are the same, 5% are not the same. Let's take `cat` vs `cats`: 3 out of 4 (75%) characters are the same, 1 out of 4 (25%) is different. 

It makes only sense to use type `1.` metrics when the inputs are always of equal length. For example, using any type `1.` metric to compare `cat` and `cats` always results in "not equal". On the other hand, type `2.` metrics are applicable to both, equal and non-equal length inputs.

[> Overview of semantic similarity](https://en.wikipedia.org/wiki/Semantic_similarity)

## Popular Similarity Metrics

Similarity metrics are mostly calculated with vector representations of words, sentences or documents. For example,
- One-hot encoded vector of two words 
- Bag of words vector of two sentences
- TF-IDF vector of two sentences or documents

### Metrics for equal length inputs

The following metrics assume two inputs having equal length. If two inputs don't have the same length, they can be normalizable by using padding characters.

**Hamming distance** is measured between two strings of equal length and defined as the number of positions that have different characters.

**Manhatten distance** (L1 norm, city block distance, taxicab) counts the number of mismatches by subtracting the difference between each pair of characters at each position of two strings.

**Euclidean distance** (L2 norm) is defined as the shortest straight-line distance between two points.

### Metrics for (non-)equal length inputs

The following metrics work with inputs of equal and non-equal length.

**[Cosine distance](https://studymachinelearning.com/cosine-similarity-text-similarity-metric/)** (L2-normalized dot product of vectors) measures the similarity by using the normalized length of two input vectors. Order of characters/words are not taken into account.

**[Jaccard](https://studymachinelearning.com/jaccard-similarity-text-similarity-metric-in-nlp/) similarity** indicates how many words two documents share by using the intersection and unions of the words.

**Levenshtein distance** measures the minimum number of edits needed to transform one input into the other. Considers order of characters or words in input.

**Jaro Winkler distance** minimum edit distance, considers prefixes.

**Okapi BM25(F) ranking** takes token distributions across corpus into account.

```
q_idf * dot(q_tf, d_tf[i]) * 1.5 (dot(q_tf, d_tf[i]) + .25 + .75 * d_num_words[i] / d_num_words.mean()))
```

## Cosine Similarity

A document, sentence or word is represented as a vector and the Cosine sim calculates the angle (="similarity") between two vectors. 

The resulting similarity ranges from:

- 1 if the vectors are the same
- 0 if the vectors don’t have any relationship (orthogonal vectors)

![cos-sim](https://studymachinelearning.com/wp-content/uploads/2019/09/Cosine-similarity-Wikipedia.png)

Similarity measurements for:
- Term frequency vectors (bag of words)
- TF-IDF vectors
- Oc-occurence vectors


### Example 1

Figure below shows three word vectors and Cosine distance (=similarity) between 
- "I hate cats" and "I love dogs" (result: not very similar)
- "I love dogs" and "I love, love, love, .. dogs" (result: similar)

<img src="https://miro.medium.com/max/700/1*QNp4TNCNwo1HMqjFV4jq1g.png" width="600">

[Credits](https://towardsdatascience.com/group-thousands-of-similar-spreadsheet-text-cells-in-seconds-2493b3ce6d8d)

### Example 2

```
doc_1 = "Data is the oil of the digital economy" 
doc_2 = "Data is a new oil" 

# Vector representation of the document
doc_1_vector = [1, 1, 1, 1, 0, 1, 1, 2]
doc_2_vector = [1, 0, 0, 1, 1, 0, 1, 0]
```

![vec](https://studymachinelearning.com/wp-content/uploads/2019/09/word_cnt_vect_cosine_similarity.png)

![vec-calc](https://studymachinelearning.com/wp-content/uploads/2019/09/cosine_similarity_example_1.png)

[Credits](https://studymachinelearning.com/cosine-similarity-text-similarity-metric/)

---

Cosine sim with scikit: [sklearn.metrics.pairwise.cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html#sklearn.metrics.pairwise.cosine_similarity)

## Jaccard Similarity

Jaccard sim is calculated by dividing the number of words occuring in both documents/sentences (intersection) with the number of all words in both docs/sents (union).

### Example 
```
doc_1 = "Data is the new oil of the digital economy"
doc_2 = "Data is a new oil"
```

Each sentence is tokenized into words.

```
words_doc1 = {'data', 'is', 'the', 'new', 'oil', 'of', 'digital', 'economy'}
words_doc2 = {'data', 'is', 'a', 'new', 'oil'}
```

Four words occur in both sentences (intersection): data, is, new, oil.

Across both sentences, nine unique words exist (union): data, a, of, is, economy, the, new, digital, oil.

![jac-sim1](https://studymachinelearning.com/wp-content/uploads/2020/04/jaccard_example.png)

Visualized:

![jac-sim2](https://studymachinelearning.com/wp-content/uploads/2020/04/intersection_example.png)

## Levenshtein Distance

Levenshtein distance between two words is the minimum number of single-character edits (i.e. insertions, deletions or substitutions) required to change one word into the other. 

The distance can also be calculated for sentences, i.e. the minimum number of single token edits required to change one sentence into the other.

Levenshtein distance for two strings `a`, `b` of length `|a|` and `|b|` is given by `lev`:

![lev-dic](https://wikimedia.org/api/rest_v1/media/math/render/svg/70962a722b0b682e398f0ee77d60c714a441c54e)

### Example 1

Levenshtein distance between `HONDA` and `HYUNDAI` is 3 because it takes three transformations to change `HONDA` to `HYUNDAI`:

1. Add `Y` to `HONDA` => `HYONDA`
2. Substitue `O` with `U` in `HYONDA` => `HYUNDA`
3. Add `I` to `HYUNDA` => `HYUNDAI`

### Example 2

Lev distance between two sentences `I love cats` and `I love dogs` is 1.

Step 1: tokenize both sentences to `["I", "love", "cats"]`, `["I", "love", "docs"]`

Step 2: perform one transformation, i.e. replace `cats` with `dogs`.


## Jaro-Winkler Distance

![img](https://miro.medium.com/max/720/1*0efk-HIrKHqv_uimWWwhDQ.jpeg)

## L2 Norm

Length of a word vector. Also known as **Euclidean norm**.

Example:
- length of "I like cats" is 4.7


```python
doc1 = nlp("I like cats")
doc2 = nlp("I like dogs")
# Compare 2 documents
doc1.similarity(doc2)
```




    0.957709143352323




```python
# "cats" vs "dogs"
doc1[2].similarity(doc2[2])
```




    0.83117634




```python
# "I" vs "like dogs"
doc1[0].similarity(doc2[1:3])
```




    0.46475163




```python
doc = nlp("I like cats")
# L2 norm of "I like cats"
doc.vector_norm
```




    4.706799587675896




```python
# L2 norm of "cats"
doc[2].vector_norm
```




    6.933004




```python
# Vector representation of "cats"
doc[2].vector
```




    array([-0.26763  ,  0.029846 , -0.3437   , -0.54409  , -0.49919  ,
            0.15928  , -0.35278  , -0.2036   ,  0.23482  ,  1.5671   ,
           -0.36458  , -0.028713 , -0.27053  ,  0.2504   , -0.18126  ,
            0.13453  ,  0.25795  ,  0.93213  , -0.12841  , -0.18505  ,
           -0.57597  ,  0.18538  , -0.19147  , -0.38465  ,  0.21656  ,
           -0.4387   , -0.27846  , -0.41339  ,  0.37859  , -0.2199   ,
           -0.25907  , -0.019796 , -0.31885  ,  0.12921  ,  0.22168  ,
            0.32671  ,  0.46943  , -0.81922  , -0.20031  ,  0.013561 ,
           -0.14663  ,  0.14438  ,  0.0098044, -0.15439  ,  0.21146  ,
           -0.28409  , -0.4036   ,  0.45355  ,  0.12173  , -0.11516  ,
           -0.12235  , -0.096467 , -0.26991  ,  0.028776 , -0.11307  ,
            0.37219  , -0.054718 , -0.20297  , -0.23974  ,  0.86271  ,
            0.25602  , -0.3064   ,  0.014714 , -0.086497 , -0.079054 ,
           -0.33109  ,  0.54892  ,  0.20076  ,  0.28064  ,  0.037788 ,
            0.0076729, -0.0050123, -0.11619  , -0.23804  ,  0.33027  ,
            0.26034  , -0.20615  , -0.35744  ,  0.54125  , -0.3239   ,
            0.093441 ,  0.17113  , -0.41533  ,  0.13702  , -0.21765  ,
           -0.65442  ,  0.75733  ,  0.359    ,  0.62492  ,  0.019685 ,
            0.21156  ,  0.28125  ,  0.22288  ,  0.026787 , -0.1019   ,
            0.11178  ,  0.17202  , -0.20403  , -0.01767  , -0.34351  ,
            0.11926  ,  0.73156  ,  0.11094  ,  0.12576  ,  0.64825  ,
           -0.80004  ,  0.62074  , -0.38557  ,  0.015614 ,  0.2664   ,
            0.18254  ,  0.11678  ,  0.58919  , -1.0639   , -0.29969  ,
            0.14827  , -0.42925  , -0.090766 ,  0.12313  , -0.024253 ,
           -0.21265  , -0.10331  ,  0.91988  , -1.4097   , -0.0542   ,
           -0.071201 ,  0.66878  , -0.24651  , -0.46788  , -0.23991  ,
           -0.14138  , -0.038911 , -0.48678  ,  0.22975  ,  0.36074  ,
            0.13024  , -0.40091  ,  0.19673  ,  0.016017 ,  0.30575  ,
           -2.1901   , -0.55468  ,  0.26955  ,  0.63815  ,  0.42724  ,
           -0.070186 , -0.11196  ,  0.14079  , -0.022228 ,  0.070456 ,
            0.17229  ,  0.099383 , -0.12258  , -0.23416  , -0.26525  ,
           -0.088991 , -0.061554 ,  0.26582  , -0.53112  , -0.4106   ,
            0.45211  , -0.39669  , -0.43746  , -0.6632   , -0.048135 ,
            0.23171  , -0.37665  , -0.38261  , -0.29286  , -0.036613 ,
            0.25354  ,  0.49775  ,  0.3359   , -0.11285  , -0.17228  ,
            0.85991  , -0.34081  ,  0.27959  ,  0.03698  ,  0.61782  ,
            0.23739  , -0.32049  , -0.073717 ,  0.015991 , -0.37395  ,
           -0.4152   ,  0.049221 , -0.3137   ,  0.091128 , -0.38258  ,
           -0.036783 ,  0.10902  , -0.38332  , -0.74754  ,  0.016473 ,
            0.55256  , -0.29053  , -0.50617  ,  0.83599  , -0.31783  ,
           -0.77465  , -0.0049272, -0.17103  , -0.38067  ,  0.44987  ,
           -0.12497  ,  0.60263  , -0.12026  ,  0.37368  , -0.079952 ,
           -0.15785  ,  0.37684  , -0.18679  ,  0.18855  , -0.4759   ,
           -0.11708  ,  0.36999  ,  0.54134  ,  0.42752  ,  0.038618 ,
            0.043483 ,  0.31435  , -0.24491  , -0.67818  , -0.33833  ,
            0.039218 , -0.11964  ,  0.8474   ,  0.09451  ,  0.070523 ,
           -0.2806   ,  0.296    , -0.17554  , -0.41087  ,  0.70748  ,
            0.17686  ,  0.043479 , -0.31902  ,  0.64584  , -0.45268  ,
           -0.7967   ,  0.099817 , -0.1734   ,  0.11404  , -0.36809  ,
            0.12035  , -0.048582 ,  0.55945  , -0.51508  ,  0.072704 ,
            0.18106  ,  0.07802  , -0.31526  ,  0.38189  ,  0.092801 ,
           -0.044227 , -0.66154  , -0.020428 ,  0.059836 , -0.23628  ,
           -0.017592 , -0.56481  , -0.52934  , -0.16392  ,  0.077331 ,
            0.24583  , -0.32195  , -0.36811  , -0.037208 ,  0.26702  ,
           -0.57907  ,  0.46457  , -0.54636  ,  0.11855  ,  0.092475 ,
           -0.10469  ,  0.03319  ,  0.62616  , -0.33684  ,  0.045742 ,
            0.25089  ,  0.28973  ,  0.060633 , -0.4096   ,  0.39198  ,
            0.58276  ,  0.496    , -0.75881  ,  0.13655  ,  0.21704  ,
           -0.37978  , -0.54051  , -0.22813  ,  0.28393  , -0.58739  ,
            1.0472   , -0.13318  , -0.07325  ,  0.12991  , -0.44999  ],
          dtype=float32)




```python
# can also be done using sklearn's linear kernel (equivilant to cosine similarity)
```

# n-grams: Unigram, bigrams, trigrams

- Unigram = one word, eg the, and, of, hotel
- Bigrams = two consecutive words, eg the hotel, in seattle, the city
- Trigrams = three consecutive words, eg easy access to, high speed internet, the heart of

Credits: https://towardsdatascience.com/building-a-content-based-recommender-system-for-hotels-in-seattle-d724f0a32070

## Get all unigrams


```python
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd 

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

document_corpus = [
  "Dog bites man",
  "Dog bites man after man eats fish", 
  "Dog bites fish",
  "Man bites dog", 
  "Dog eats meat", 
  "Man eats food",
  "Man eats fish"
]

common_words = get_top_n_words(document_corpus, 5) # or use df['desc'] 

df2 = pd.DataFrame(common_words, columns = ['desc' , 'count'])
df2.groupby('desc').sum()['count'].sort_values().plot(kind='barh', title='Top 5 words in document corpus')

```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fbae1ff3510>




![png](README_files/README_102_1.png)


## Get all bigrams


```python
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

document_corpus = [
  "Dog bites man",
  "Dog bites man after man eats fish", 
  "Dog bites fish",
  "Man bites dog", 
  "Dog eats meat", 
  "Man eats food",
  "Man eats fish"
]

common_words = get_top_n_bigram(document_corpus, 5)

df4 = pd.DataFrame(common_words, columns = ['desc' , 'count'])
df4.groupby('desc').sum()['count'].sort_values().plot(kind='barh', title='Top 5 bigrams in our corpus after removing stop words')

```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fbae2086950>




![png](README_files/README_104_1.png)


## Get all trigrams


```python
def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

document_corpus = [
  "Dog bites man",
  "Dog bites man after man eats fish", 
  "Dog bites fish",
  "Man bites dog", 
  "Dog eats meat", 
  "Man eats food",
  "Man eats fish"
]

common_words = get_top_n_trigram(document_corpus, 5)

df6 = pd.DataFrame(common_words, columns = ['desc' , 'count'])
df6.groupby('desc').sum()['count'].sort_values().plot(kind='barh', title='Top 5 trigrams in our corpus after removing stop words')

```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fbae20fb150>




![png](README_files/README_106_1.png)


# Visualization


```python
from spacy import displacy
```


```python
doc = nlp("This is a sentence")
displacy.render(doc, style="dep")
```


<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:lang="en" id="0bf72b5ccfc143b4871b353f00ec3470-0" class="displacy" width="750" height="312.0" direction="ltr" style="max-width: none; height: 312.0px; color: #000000; background: #ffffff; font-family: Arial; direction: ltr">
<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="50">This</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="50">DET</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="225">is</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="225">VERB</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="400">a</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="400">DET</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="575">sentence</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="575">NOUN</tspan>
</text>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-0bf72b5ccfc143b4871b353f00ec3470-0-0" stroke-width="2px" d="M70,177.0 C70,89.5 220.0,89.5 220.0,177.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-0bf72b5ccfc143b4871b353f00ec3470-0-0" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">nsubj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M70,179.0 L62,167.0 78,167.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-0bf72b5ccfc143b4871b353f00ec3470-0-1" stroke-width="2px" d="M420,177.0 C420,89.5 570.0,89.5 570.0,177.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-0bf72b5ccfc143b4871b353f00ec3470-0-1" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">det</textPath>
    </text>
    <path class="displacy-arrowhead" d="M420,179.0 L412,167.0 428,167.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-0bf72b5ccfc143b4871b353f00ec3470-0-2" stroke-width="2px" d="M245,177.0 C245,2.0 575.0,2.0 575.0,177.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-0bf72b5ccfc143b4871b353f00ec3470-0-2" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">attr</textPath>
    </text>
    <path class="displacy-arrowhead" d="M575.0,179.0 L583.0,167.0 567.0,167.0" fill="currentColor"/>
</g>
</svg>



```python
doc = nlp("Larry Page founded Google in the US in early 1990.")
displacy.render(doc, style="ent")
```


<div class="entities" style="line-height: 2.5; direction: ltr">
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    Larry Page
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 founded 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    Google
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 in the 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    US
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 in 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    early 1990
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
.</div>


Inspired by: https://www.datacamp.com/community/blog/spacy-cheatsheet

# Kernels 

Used by 
- Support Vector Machines (SVMs)
- Principal Component Analysis (PCA)

Useful for
- classification tasks

Also known as
- kernel function
- similarity function

Opposite of kernels: vectors

Source:
- [Wikipedia](https://en.wikipedia.org/wiki/Kernel_method)

## Linear Kernel

Linear Kernel is used when the data is Linearly separable, that is, it can be separated using a single Line.

Compute the linear kernel between X and Y: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.linear_kernel.html#sklearn.metrics.pairwise.linear_kernel

## Non-linear Kernel

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Kernel_trick_idea.svg/1280px-Kernel_trick_idea.svg.png" width="600">

[Credits](https://en.wikipedia.org/wiki/Kernel_method)

# Spearman's Rank Correlation Coefficient



<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Spearman_fig1.svg/1280px-Spearman_fig1.svg.png" width="300">

Credits: https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient

# kNN

k-nearest neighbors algoritm

Useful for
- classification

# Text Summarization

- [How to Make a Text Summarizer](https://github.com/llSourcell/How_to_make_a_text_summarizer)
- [How to Prepare News Articles for Text Summarization](https://machinelearningmastery.com/prepare-news-articles-text-summarization/)

# Sentiment Analysis

Is text fact or opinion? Only perform sentiment analysis on opinion, not facts.

Sentiments:
- positive
- neutral
- negative

2 ways:
- rule-based uses lexicon with polarity score per word. Count positive and negative words. Doesn't provide training data.
- automatic using machine learning (=classification problem). Needs training data.

Sentiment analysis can be performed with ntlk's `SentimentIntensityAnalyzer`

See: https://www.nltk.org/api/nltk.sentiment.html#module-nltk.sentiment.vader

Learning resources: 
- https://www.youtube.com/watch?v=3Pzni2yfGUQ
- https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184


```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')
```

    /Users/jan/PycharmProjects/playground/nlp-cheat-sheet/venv/lib/python3.6/site-packages/nltk/twitter/__init__.py:20: UserWarning: The twython library has not been installed. Some functionality from the twitter package will not be available.
      warnings.warn("The twython library has not been installed. "


# Logistic Regression

A classification model that uses a sigmoid function to convert a linear model's raw prediction () into a value between 0 and 1. You can interpret the value between 0 and 1 in either of the following two ways:

- As a probability that the example belongs to the positive class in a binary classification problem.
- As a value to be compared against a classification threshold. If the value is equal to or above the classification threshold, the system classifies the example as the positive class. Conversely, if the value is below the given threshold, the system classifies the example as the negative class.

https://developers.google.com/machine-learning/glossary/#logistic-regression

# RNN

Recurrent neural networks
- Size changes depending on input/output (in contrast to neural network like CNN)

## LSTM

Long Short-Term Mermoy

ToDo


```python
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, Merge, TimeDistributedDense
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-63-61823dfb33e8> in <module>
    ----> 1 from keras.layers.recurrent import LSTM
          2 from keras.models import Sequential
          3 from keras.layers.core import Dense, Activation, Dropout, RepeatVector, Merge, TimeDistributedDense


    ModuleNotFoundError: No module named 'keras'


# Levenshtein distance


```python
import Levenshtein
```

# Regularization

# Markov Decision Process

- State -> action -> state -> action ...
- Agent
- Set of actions
- Transitions
- Discount factor
- Reward

# Probability to discard words to reduce noise

![prob](https://miro.medium.com/max/460/1*h4xJftToHQRc_sl1ejpmfA.png)

Credits: https://towardsdatascience.com/how-to-train-custom-word-embeddings-using-gpu-on-aws-f62727a1e3f6

# Loss functions

A measure of how far a model's predictions are from its label.

In contrast to:
- reward function

## SSE (sum of squared of the errors)

## Mean Squared Errors (MSE)

Mean Squared Error (MSE) is a common loss function used for regression problems.

Mean squared error of an estimator measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.

Can be used for regression problems (say, to predict the price of a house).

Alternatives:
- Binary Crossentropy Loss (is better for dealing with probabilities)

## Binary Crossentropy Loss

Used in binary classification tasks, ie model outputs a probability (a single-unit layer with a sigmoid activation), we'll use the binary_crossentropy loss function.

## Cross-entropy loss

## Sparse Categorical Crossentropy

Used in image classification task

## Log loss

Used in logistic regression tasks

# Optimizer

This is how the model is updated based on the data it sees and its loss function.

## Gradient Descent

Optimization algorithm for finding the minimum of a function.

## Stochastic Gradient Descent (SGD)

## Adam

## AdaBoost

## AdaGrad

# NN Frameworks
- Keras (best learning tool for beginners)
- PyTorch (dynamic)
- Tensorflow (declerative programming, can run on Apache Spark)

# Classification

- Binary
- Not binary

# Activation function

A function (for example, ReLU or sigmoid) that takes in the weighted sum of all of the inputs from the previous layer and then generates and passes an output value (typically nonlinear) to the next layer.

https://developers.google.com/machine-learning/glossary/#activation_function

## Softmax Function

A function that provides probabilities for each possible class in a multi-class classification model. The probabilities add up to exactly 1.0. For example, softmax might determine that the probability of a particular image being a dog at 0.9, a cat at 0.08, and a horse at 0.02.

Example: last layer is a 10-node softmax layer—this returns an array of 10 probability scores that sum to 1.


## Sigmoid

A function that maps logistic or multinomial regression output (log odds) to probabilities, returning a value between 0 and 1

Sigmoid function converts /sigma into a probability between 0 and 1.

## ReLU (Rectified Linear Unit)

- If input is negative or zero, output is 0.
- If input is positive, output is equal to input.

# Performance measure

## Accuracy

Used when taining a neural network.

- training loss decreases with each epoch
- training accuracy increases with each epoch

![acc](https://www.tensorflow.org/tutorials/keras/basic_text_classification_files/output_6hXx-xOv-llh_0.png)


## Precision

TP/(TP+FP)

- TP=true positive
- FP=false positive

## Recall

TP/(TP+FN)

## F1 score

(2 × Precision × Recall) / (Precision + Recall)

## Mean Absolute Error

A common regression metric is Mean Absolute Error (MAE).

## Mean Squared Error



# Early stopping

Early stopping is a useful technique to prevent overfitting.

# Regularization

## L1 Regularization
penalizes weights in proportion to the sum of the absolute values of the weights

https://developers.google.com/machine-learning/glossary/#L1_regularization

## L2 Regularization

penalizes weights in proportion to the sum of the squares of the weights

# Sparsity

The number of elements set to zero (or null) in a vector or matrix divided by the total number of entries in that vector or matrix.

# Ranking

## Wilson-Score Interval

Used by Reddit to rank comments.

## Euclidean Ranking

## Cosine Ranking

# XLNet + BERT in spacy

https://spacy.io/models/en#en_pytt_xlnetbasecased_lg

# Latent Dirichlet Allocation

# Confusion Matrix

A confusion matrix is a table where each cell `[i,j]` indicates how often label `j` was predicted when the correct label was `i`.

# Entropy & Information Gain

Information gain measures how much more organized the input values become when we divide them up using a given feature. To measure how disorganized the original set of input values are, we calculate **entropy** of their labels, which will be high if the input values have highly varied labels,

Entropy is defined as the sum of the probability of each label times the log probability of that same label:

![image.png](data:image/png;base64,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)

Labels that have low frequency do not contribute much to the entropy (since P(l) is small), and labels with high frequency also do not contribute much to the entropy (since log2P(l) is small).



![image.png](data:image/png;base64,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)

Credits: Natural Language Processing with Python book

# Naive Bayes Classifiers

- Every feature gets a say in determining which label should be assigned to a given input value. 
- To choose a label for an input value, the naive Bayes classifier begins by calculating the prior probability of each label, which is determined by checking frequency of each label in the training set.

Credits: https://www.nltk.org/book/ch06.html
