import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

import re

contractions:
def clean_contractions(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

stemming:
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import *

def stem_and_clean(text):
    sno = SnowballStemmer('english', ignore_stopwords=True)
    punctuations="?:!.,;"

    sentences = sent_tokenize(clean_contractions(text))
    for si, sentence in enumerate(sentences):
        sentence_words = word_tokenize(sentence)
        new_words = []
        for word in sentence_words:
            if word in punctuations:
                sentence_words.remove(word)
            else:
                new_words.append(sno.stem(word))
        sentences[si] = ' '.join(new_words)
    return ' '.join(sentences)

Stem movie plots:
df_lyrics = df_lyrics.astype(str)
df_lyrics['lyrics'] = df_lyrics['lyrics'].map(lambda com : stem_and_clean(com))

!pip install textblob
!python -m textblob.download_corpora

from textblob import TextBlob
Sentiment = []

for l in df_lyrics['lyrics']:
   Sentiment.append(TextBlob(l).sentiment.polarity)
print(Sentiment)

m = min(Sentiment)
M = max(Sentiment)
mn = sum(Sentiment) / len(Sentiment)
print('min:', m)
print('max:', M)
print('mean:', mn)

print('size', df_lyrics['lyrics'].size)

ltz = 0
htz = 0
eqz = 0

for i in Sentiment:
  if i < 0:
    ltz = ltz + 1
  elif i == 0:
    eqz = eqz + 1
  else:
    htz = htz +1

print('less than zero', ltz)
print('equal to zero', htz)
print('higher than zero', htz)

PolarizedSentiment = []
NegativeCount = 0;
PositiveCount = 0;

for i in Sentiment:
  if i < 0:
    PolarizedSentiment.append(-1)
    NegativeCount = NegativeCount + 1
  else:
    PolarizedSentiment.append(1)
    PositiveCount = PositiveCount + 1 

print('PolarizedSentiment', PolarizedSentiment)
print('number of songs with positive sentiment', PositiveCount)
print('number of songs with negative sentiment', NegativeCount)

PreviousValue = -1;
CountInRange = []
r = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for i in r:
  count = 0
  for j in Sentiment:
    if (PreviousValue <= j) & (j < i):
      count = count + 1
    if (i == 1) & (j == i):
      count = count + 1
  print('[', PreviousValue, ', ', i, ']: ', count)
  CountInRange.append(count)
  PreviousValue = i

import matplotlib.pyplot as plt

plt.bar(r, CountInRange, width = 0.07)

plt.show()
