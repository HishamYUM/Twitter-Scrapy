from spacy_langdetect import LanguageDetector
import spacy
import os
import pandas as pd
from datetime import date
import numpy as np
import re
import threading
from random import randint
from time import sleep
import string
import nltk
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
# Initial Parameters
today = date.today()
end_date = today
search_term = 'samsung'
from_date = '2021-01-01'
max_results = 5000


#_______________________________________________________________________________________
def myfunct_scrapy(search_term) :
  # Scrapping Part
  extracted_tweets = "snscrape --format '{content!r}'"+ f" --max-results {max_results} --since {from_date} twitter-search '{search_term} until:{end_date}' > extracted-tweets.csv"
  os.system(extracted_tweets)
  if os.stat("extracted-tweets.csv").st_size == 0:
    print('No Tweets found')
  else:
    df = pd.read_csv('extracted-tweets.csv', names=['content'])
    for row in df['content'].iteritems():
      print(row)

  # Converting to Dataframe
  df = pd.read_csv('extracted-tweets.csv', names=['content'])
  Data = pd.DataFrame(df['content'])
  r = []
  for item in range(0, len(Data.content)):
    r.append(Data.content[item])
  return r
#___________________________________________________________________________________________

# Cleanning Part
def clean_tweet_whole(r):
  df = list()
  def clean_tweet(tweet):
    if type(tweet) == np.float:
      return ""
    temp = tweet.lower()
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)
    return temp
  
  for i in range(0 , len(r)):
    df.append(clean_tweet(r[i]))
  
  # Duplicate remove
  df = list(set(df))
  return df

#_____________________________________________________________________________________________
#Check english :

def check_en(df) :
  result = []
  nlp = spacy.load('en')  # 1
  nlp.add_pipe(LanguageDetector(), name='language_detector', last=True) #2
  for element in df:
    text_content = element
    doc = nlp(text_content) #3
    detect_language = doc._.language #4
    if (detect_language['language'] == 'en' ) and (detect_language['score'] > 0.95) :
      result.append(text_content) 
  return result

#_________________________________________________________________________________________________
#Word tokenizer
def prepro(df):
  list_words = []
  sentences_base = []
  def preprocess_tweet_text(tweet):
      tweet.lower()
      # Remove urls
      tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
      # Remove user @ references and '#' from tweet
      tweet = re.sub(r'\@\w+|\#','', tweet)
      # Remove punctuations
      tweet = tweet.translate(str.maketrans('', '', string.punctuation))
      # Remove stopwords
      tweet_tokens = word_tokenize(tweet)
      filtered_words = [w for w in tweet_tokens if not w in stop_words]
      
      #ps = PorterStemmer()
      #stemmed_words = [ps.stem(w) for w in filtered_words]
      #lemmatizer = WordNetLemmatizer()
      #lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
      
      return " ".join(filtered_words)

  b = []
  for element in df:
      k = preprocess_tweet_text(element)
      b.append(k)
  Data = pd.DataFrame()
  Data['text_cleaned_stop_words'] = b

  def remove_punct(text):
      text  = "".join([char for char in text if char not in string.punctuation])
      text = re.sub('[0-9]+', '', text)
      return text
  Data['text_punct'] = Data['text_cleaned_stop_words'].apply(lambda x: remove_punct(x))

  def tokenization(text):
      text = re.split('\W+', text)
      return text
      
  Data['text_tokenized'] = Data['text_punct'].apply(lambda x: tokenization(x.lower()))

  def remove_stopwords(text):
      text = [word for word in text if word not in stop_words]
      return text
      
  Data['text_nonstop'] = Data['text_tokenized'].apply(lambda x: remove_stopwords(x))

  # List of vocabulary 

  list_words = []
  for element in Data['text_nonstop']:
    for k in element :
      list_words.append(k) 

  list_words = list(set(list_words))

  return list_words

#_________________________________________________________________________________________________

base_s = []
base_w = ["samsung" , "phone"]

#_____________________________________________________________________________________________
start_time = time.time()
seconds = 36000

for i in base_w:
  thread_list = []

  search_term = i 
  df = myfunct_scrapy(search_term)
  #____
  n = 10
  ten_split = [df[i::n] for i in range(n)]
  #____
  for i in range(1, 11):
      locals()[str((str('df')+str(i)))] = ten_split[i-1]

  def main(df):

    df = clean_tweet_whole(df)
    df = check_en(df)
    words = prepro(df)
    base_s.extend(df)
    base_w.extend(words)

  for i in range(1, 10):
    t = threading.Thread(target=main, args=(locals()[str('df')+str(i)],))
    thread_list.append(t)

  for thread in thread_list:
    if (thread.is_alive() == False):
      thread.start()

  for thread in thread_list:
   thread.join()

  current_time = time.time()
  elapsed_time = current_time - start_time

  if elapsed_time > seconds:
    break
  
  print ('/n Length of vocabulary before remove duplicate: ' , len(base_w) , '/n')

  base_w = list(set(base_w))

  print ('/n Length of vocabulary before after duplicate: ' , len(base_w) , '/n')


  print ('/n Length of vocabulary before remove duplicate: ' , len(base_s) , '/n')

  base_s = list(set(base_s))

  print ('/n Length of vocabulary after remove duplicate: ' , len(base_s) , '/n')

#_________________________________________________________________________________________________

words = pd.DataFrame(base_w , columns= ['words'])
sents = pd.DataFrame(base_s , columns= ['sentences'])
words.to_csv('words.csv', index=False)
sents.to_csv('sents.csv', index=False)



