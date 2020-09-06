#!/usr/bin/env python
# coding: utf-8

# Author:
# Carlos Catalao Alves

# Emotional Sentiment on Twitter
# A coronavirus vaccine online firestorm

# In this python script you will find examples of some of the most common 
# NLP (Natural Language Processing) techniques used to uncover patterns of 
# sentiment and emotion on social media microblogging platforms like Twitter. 

# It is organized as follows:

# - Step 1: Exploratory analysis
# - Step 2: Text processing
# - Step 3: Sentiment analysis 
# - Step 4: Word frequency 
# - Step 5: LDA topics extraction
# - Step 6: Emotion analysis
# 

# ## Step 1:  EXPLORATORY ANALYSIS

import pandas as pd 
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from collections import defaultdict
from datetime import date

import re # for regular expressions
import string


# Importing the data
tweets = pd.read_csv('input/tweets.csv')

# getting the date column ready for datetime operations
tweets['datetime']= pd.to_datetime(tweets['datetime'])

# A plot of the tweets with the word "CureVac" over the past 6 years.
fig = plt.figure(figsize=(15, 10))
ax = sns.lineplot(data=tweets.set_index("datetime").groupby(pd.Grouper(freq='Y')).count())
plt.title('Tweets with "CureVac" from 2014 to 2020', fontsize=20)
plt.xlabel('Years', fontsize=15)
plt.ylabel('Tweets', fontsize=15)
fig.savefig("images/All_Tweets_2014-2020.png")


# creating a column to filter the online storm period (from 15 and 18 March)
def make_onlinestorm_field():
    for i, row in tweets.iterrows():
        if pd.to_datetime(tweets.at[i, 'datetime']) > pd.Timestamp(date(2020,3,15)):
            tweets.at[i, 'onlinestorm'] = True
        else:
            tweets.at[i, 'onlinestorm'] = False  
            
make_onlinestorm_field()

# counting tweets during the three days online storm
print('In three days, tweets went over {}, all around the world.'.format(tweets[tweets['onlinestorm']]['onlinestorm'].count()))

tweets[tweets['onlinestorm']]

# Let's now have a look at the distribution of the tweets, by the hour, during the online storm.
fig = plt.figure(figsize=(15, 10))
ax = sns.lineplot(data=tweets[tweets['onlinestorm'] == True].set_index("datetime").groupby(pd.Grouper(freq='H')).onlinestorm.count())
plt.title('Tweets per hour from 15 to 18 March 2020', fontsize=20)
plt.xlabel('Time (hours)', fontsize=15)
plt.ylabel('No. Tweets', fontsize=15)
fig.savefig("images/All_Tweets_Onlinestorm.png")


# It is time to have a first look at the content of the tweets and do some descriptive statistics. 
# For now, I will focus only on features like hastags, mentions, urls, capital words and words in general.

# A function to count tweets based on regular expressions
def count_tweets(reg_expression, tweet):
    tweets_list = re.findall(reg_expression, tweet)
    return len(tweets_list)

# Creating a dictionary to hold these counts
content_count = {
    'words' : tweets['text'].apply(lambda x: count_tweets(r'\w+', x)),
    'mentions' : tweets['text'].apply(lambda x: count_tweets(r'@\w+', x)),
    'hashtags' : tweets['text'].apply(lambda x: count_tweets(r'#\w+', x)),
    'urls' : tweets['text'].apply(lambda x: count_tweets(r'http.?://[^\s]+[\s]?', x)),   
}
df = pd.concat([tweets, pd.DataFrame(content_count)], axis=1)

# Tweets descriptive statistics

# Display descriptive statistics fdor words, mentions,
# hashtags and urls
for key in content_count.keys():
    print()
    print('Descriptive statistics for {}'.format(key))
    print(df.groupby('onlinestorm')[key].describe())

# Now plot them 
for key in content_count.keys():

    bins = np.arange(df[key].min(), df[key].max() + 1)
    g = sns.FacetGrid(df, col='onlinestorm', height=5, hue='onlinestorm', palette="RdYlGn")
    g = g.map(sns.distplot, key, kde=False, norm_hist=True, bins=bins)
    plt.savefig('images/Descriptive_stats_for_' + key + '.png')


# Step 2: TEXT PROCESSING

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

# I am adding my own stopwords list to the NLTK list.
# This way we can drop words that are irrelevant for text processing
MY_STOPWORDS = ['curevac','vaccine','german','mrna','biotech','cancer', 'lilly','eli','ag','etherna_immuno', 'translatebio',                 'mooreorless62','boehringer', 'ingelheim','biopharmaceutical', 'company']
STOPLIST = set(stopwords.words('english') + list(MY_STOPWORDS))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "``", ",", ".", ":", "''","#","@"]

# The NLTK lemmatizer and stemmer classes
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')

# read english selected tweets, no duplicates
tweets = pd.read_csv('input/tweets_en.csv')

# I use the POS tagging from NLTK to retain only adjectives, verbs, adverbs 
# and nouns as a base for for lemmatization.
def get_lemmas(tweet): 
    
    # A dictionary to help convert Treebank tags to WordNet
    treebank2wordnet = {'NN':'n', 'JJ':'a', 'VB':'v', 'RB':'r'}
    
    postag = ''
    lemmas_list = []
    
    for word, tag in pos_tag(word_tokenize(tweet)):
        if tag.startswith("JJ") or tag.startswith("RB") or tag.startswith("VB") or tag.startswith("NN"):
                
            try:
                postag = treebank2wordnet[tag[:2]]
            except:
                postag = 'n'                
                            
            lemmas_list.append(lemmatizer.lemmatize(word.lower(), postag))    
    
    return lemmas_list


# We will now pre-process the tweets, following a pipeline of tokenization, 
# filtering, case normalization and lemma extraction.

# This is the function to clean and filter the tokens in each tweet
def clean_tweet(tokens):
    
    filtered = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if token not in STOPLIST:
                if token[0] not in SYMBOLS:
                    if not token.startswith('http'):
                        if  '/' not in token:
                            if  '-' not in token:
                                filtered.append(token)
                                        
    return filtered


# Prior to lemmatization, I apply POS (part-of-speech) tagging to make sure that only the   
# adjectives, verbs, adverbs and nouns are retained.

# Starts the lemmatization process
def get_lemmatized(tweet):
   
    all_tokens_string = ''
    filtered = []
    tokens = []

    # lemmatize
    tokens = [token for token in get_lemmas(tweet)]
    
    # filter
    filtered = clean_tweet(tokens)

    # join everything into a single string
    all_tokens_string = ' '.join(filtered)
    
    return all_tokens_string


# get the lemmatized tweets and puts the result in an "edited" text column
# for future use in this script
edited = ''
for i, row in tweets.iterrows():
    edited = get_lemmatized(tweets.loc[i]['text'])
    if len(edited) > 0:
        tweets.at[i,'edited'] = edited
    else:
        tweets.at[i,'edited'] = None        


# After lemmatization, some tweets may end up with the same words
# Let's make sure that we have no duplicates
tweets.drop_duplicates(subset=['edited'], inplace=True)
tweets.dropna(inplace=True)


# With these text processing steps, and the removal of duplicates, 
# the final sample counts 5,508 English-language tweets, 
# with an average of 30 words (SD 12.5, ranging from 4 to 61 words). 

# Using apply/lambda to create a new column with the number of words in each tweet
tweets['word_count'] = tweets.apply(lambda x: len(x['text'].split()),axis=1)
t = pd.DataFrame(tweets['word_count'].describe()).T

tweets.head()


# Step 3: SENTIMENT ANALYSIS

# Let us import the VADER analyser.
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# For the puropose of the timeseries analysis, we must make sure that the tweets are all correctly sorted.
tweets['datetime']=pd.to_datetime(tweets['datetime']) 
tweets.sort_values('datetime', inplace=True, ascending=True)
tweets = tweets.reset_index(drop=True)

# Creating a column to "filter" the online storm period.
make_onlinestorm_field()

# To avoid repetitions in our code, here are some plotting functions 
# that will be called often ...

def plot_sentiment_period(df, info):
    
    # Using the mean values of sentiment for each period
    df1 = df.groupby(df['datetime'].dt.to_period(info['period'])).mean()

    df1.reset_index(inplace=True)
    df1['datetime'] = pd.PeriodIndex(df1['datetime']).to_timestamp()
    plot_df = pd.DataFrame(df1, df1.index, info['cols'])

    plt.figure(figsize=(15, 10))
    ax = sns.lineplot(data=plot_df, linewidth = 3, dashes = False)
    plt.legend(loc='best', fontsize=15)
    plt.title(info['title'], fontsize=20)
    plt.xlabel(info['xlab'], fontsize=15)
    plt.ylabel(info['ylab'], fontsize=15)
    plt.tight_layout()
    plt.savefig('images/' + info['fname'])    
    return


def plot_fractions(props, title, fname):
    
    plt1 = props.plot(kind='bar', stacked=False, figsize=(16,5), colormap='Spectral') 
    plt.legend(bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Online storm', fontweight='bold', fontsize=18)
    plt.xticks(rotation=0,fontsize=14)
    #plt.ylim(0, 0.5)
    plt.ylabel('Fraction of Tweets', fontweight='bold', fontsize=18)
    plt1.set_title(label=title, fontweight='bold', size=20)
    plt.tight_layout()
    plt.savefig('images/' + fname + '.png')
    
    return


def plot_frequency_chart(info):
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.set_context("notebook", font_scale=1)    
    ax = sns.barplot(x=info['x'], y=info['y'], data=info['data'], palette=(info['pal']))
    ax.set_title(label=info['title'], fontweight='bold', size=18)
    plt.ylabel(info['ylab'], fontsize=16)
    plt.xlabel(info['xlab'], fontsize=16)
    plt.xticks(rotation=info['angle'],fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('images/' + info['fname'])
    
    return

# Calling VADER
analyzer = SentimentIntensityAnalyzer()

# Get VADER Compound value for sentiment intensity
tweets['sentiment_intensity'] = [analyzer.polarity_scores(v)['compound'] for v in tweets['edited']]

# This function returns the sentiment category
def get_sentiment(intensity):
    if intensity >= 0.05:
        return 'Positive'
    elif (intensity >= -0.05) and (intensity < 0.05):
        return 'Neutral'
    else:
        return 'Negative'

# Using pandas apply/lambda to speed up the process
tweets['sentiment'] = tweets.apply(lambda x: get_sentiment(x['sentiment_intensity']),axis=1)


#  The next plot gives us a clear image of the “explosion” of contradictory sentiments in this period:
df=tweets.loc[:,['datetime','sentiment_intensity']]
# filter for these dates
df.set_index('datetime',inplace=True)
df=df[(df.index>='2020-03-12') & (df.index<'2020-03-18')]
df.plot(figsize=(12,6));
plt.ylabel('Compoud score', fontsize=15)
plt.xlabel('Tweets', fontsize=15)
plt.legend().set_visible(False)
plt.title('Sentiment on tweets with CureVac (12 March to 18 March)', fontsize=20)
plt.tight_layout()
sns.despine(top=True)
plt.savefig('images/Sentiment_during_onlinestorm.png')   
plt.show()


# And this one will shows us a comparison of the sentiments before and during the online strom.
# Values are normalized to take into account the number of tweets in each 
# of the two different periods
props = tweets.groupby('onlinestorm')['sentiment'].value_counts(normalize=True).unstack()
plot_fractions(props,'Percentage of sentiments before and during the online storm',
               'Fraction_sentiments_before_and_during_onlinestorm')


# Step 4: Word frequency

# We need these imports for the wordcloud representation:
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib.colors import makeMappingArray
from palettable.colorbrewer.diverging import Spectral_4

from collections import Counter    # Counts the most common items in a list

def display_wordcloud(tokens, title, fname):
    
    tokens_upper = [token.upper() for token in tokens]

    cloud_mask = np.array(Image.open("images/cloud_mask.png"))
    wordcloud = WordCloud(max_font_size=100, 
                          max_words=50, width=2500, 
                          height=1750,mask=cloud_mask, 
                          background_color="white").generate(" ".join(tokens_upper))
    plt.figure()
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.title(title, fontsize=20)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('images/'+ fname  + '.png')   
    plt.show()

    return


def join_edited_string(edited_tweets):
    
    edited_string = ''
    for row in edited_tweets:
        edited_string = edited_string + ' ' + row
        
    return edited_string
    

def get_trigrams(trigrams, top_grams):
    
    grams_str = []
    data = []

    gram_counter = Counter(trigrams)
    
    for grams in gram_counter.most_common(10):
        gram = ''
        grams_str = grams[0]
        grams_str_count = []
        for n in range(0,3):
            gram = gram + grams_str[n] + ' '
        grams_str_count.append(gram)
        grams_str_count.append(grams[1])
        data.append(grams_str_count)
        print(grams_str_count)

    df = pd.DataFrame(data, columns = ['Grams', 'Count'])

    return df


# Let’s have a look at the 20 most frequent words in tweets before the online storm.

# Filtering the tweets of the 6 years before the online storm
df = tweets[tweets['onlinestorm'] == False]

# Join all the edited tweets in one single string
joined_string = join_edited_string(df['edited'])

# Get tokens
tokens = joined_string.split(' ')

# get trigrams
trigrams = nltk.trigrams(tokens)


# plot word frequency during online storm
word_counter = Counter(tokens)
df_counter = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
info = {'data': df_counter, 'x': 'freq', 'y': 'word',
       'xlab': 'Count', 'ylab': 'Words', 'pal':'viridis',
       'title': 'Most frequent words before online storm',
       'fname':'word_frequency_before_onlinestorm.png',
       'angle': 90}
plot_frequency_chart(info)


# plot trigram frequency
df_trigrams = get_trigrams(trigrams, 10)
info = {'data': df_trigrams, 'x': 'Grams', 'y': 'Count',
       'xlab': 'Trigrams', 'ylab': 'Count', 'pal':'viridis',
       'title': 'Most frequent trigrams before online storm',
       'fname':'trigrams_frequency_before_onlinestorm.png',
       'angle': 40}
plot_frequency_chart(info)


# And the wordcloud ...
display_wordcloud(tokens, 'Wordcloud of most frequent words before online storm',
                 'WordCloud_before_onlinestorm')

# Filtering the tweets of the 3 days of the online storm
df =tweets[tweets['onlinestorm']]

# Join all the edited tweets in one single string
joined_string = join_edited_string(df['edited'])

# Get tokens
tokens = joined_string.split(' ')

# get trigrams
trigrams = nltk.trigrams(tokens)

# plot word frequency during online storm
word_counter = Counter(tokens)
df_counter = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
info = {'data': df_counter, 'x': 'freq', 'y': 'word',
       'xlab': 'Count', 'ylab': 'Words', 'pal':'inferno',
       'title': 'Most frequent words during online storm',
       'fname':'word_frequency_during_onlinestorm.png',
       'angle': 90}
plot_frequency_chart(info)


# In[139]:


# plot trigrams frequency
df_trigrams = get_trigrams(trigrams, 10)
info = {'data': df_trigrams, 'x': 'Grams', 'y': 'Count',
       'xlab': 'Trigrams', 'ylab': 'Count', 'pal':'inferno',
       'title': 'Most frequent trigrams during online storm',
       'fname':'trigrams_frequency_during_onlinestorm.png',
       'angle': 40}
plot_frequency_chart(info)


# In[140]:


display_wordcloud(tokens, 'Wordcloud of most frequent words during online storm',
                 'WordCloud_during_onlinestorm')


# Step 5: LDA topics extraction

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer


# I am using here Susan Li's functions to get the top words from a topic:

def get_keys(topic_matrix):
    '''
    returns an integer list of predicted topic 
    categories for a given topic matrix
    '''
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys

def keys_to_counts(keys):
    '''
    returns a tuple of topic categories and their 
    accompanying magnitudes for a given list of keys
    '''
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)

def get_top_n_words(n, n_topics, keys, document_term_matrix, tfidf_vectorizer):
    '''
    returns a list of n_topic strings, where each string contains the n most common 
    words in a predicted category, in order
    '''
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
                temp_vector_sum += document_term_matrix[i]
        temp_vector_sum = temp_vector_sum.toarray()
        top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:],0)
        top_word_indices.append(top_n_word_indices)   
    top_words = []
    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = np.zeros((1,document_term_matrix.shape[1]))
            temp_word_vector[:, index] = 1
            the_word = tfidf_vectorizer.inverse_transform(temp_word_vector)[0][0]
            try:
                topic_words.append(the_word.encode('ascii').decode('utf-8'))
            except:
                pass
        top_words.append(", ".join(topic_words))         
    return top_words


# And here is a function for topics extraction using LDA, in which I produce a dataframe 
# with the topics and their top words to facilitate the plotting that follows.

# LDA topics
def get_topics(edited, n_topics, n_words):

    eds = edited.values
    
    vec = TfidfVectorizer(use_idf=True, smooth_idf=True)
    document_term_matrix = vec.fit_transform(eds)
    
    model = LatentDirichletAllocation(n_components=n_topics)
    topic_matrix = model.fit_transform(document_term_matrix)
    
    keys = get_keys(topic_matrix)
    categories, counts = keys_to_counts(keys)
    top_n_words = get_top_n_words(n_words, n_topics, keys, document_term_matrix, vec)

    topics = ['Topic {}: \n'.format(i + 1) + top_n_words[i] for i in categories]
    data=[]
    for i, topic in enumerate(topics):
        tmp = []
        tmp.append(topic)
        tmp.append(counts[i])
        data.append(tmp)
    df_topics = pd.DataFrame(data, columns = ['Topics', 'Count'])
    
    return df_topics


# Topics before the online storm

# Filtering the tweets of the 6 years before the online storm
df = tweets[tweets['onlinestorm'] == False]

# LDA topics
df_topics = get_topics(df['edited'], 5, 5)
info = {'data': df_topics, 'x': 'Topics', 'y': 'Count',
       'xlab': 'Topics', 'ylab': 'Count', 'pal':'viridis',
       'title': 'LDA Topics before Online Storm',
       'fname':'LDA_Topics_before_onlinestorm.png',
       'angle': 40}
plot_frequency_chart(info)


# Topics during the online storm

# Filtering the tweets of the 3 days of the online storm
df =tweets[tweets['onlinestorm']]

# LDA topics
df_topics = get_topics(df['edited'], 5, 5)
info = {'data': df_topics, 'x': 'Topics', 'y': 'Count',
       'xlab': 'Topics', 'ylab': 'Count', 'pal':'inferno',
       'title': 'Main Topics during Online Storm',
       'fname':'LDA_Topics_during_onlinestorm.png',
       'angle': 40}
plot_frequency_chart(info)


# Step 6: Emotion analysis

import termcolor
import sys
from termcolor import colored, cprint
plt.style.use('fivethirtyeight')

# Importing the data from the NCR lexicon
ncr = pd.read_csv('input/NCR-lexicon.csv', sep =';')

# Let's create a list of the emotions
emotions = ['Anger', 'Anticipation','Disgust','Fear', 'Joy','Sadness', 'Surprise', 'Trust']

# Join all the edited tweets in one single string
joined_string = join_edited_string(df['edited'])

# Get tokens
tokens = joined_string.split(' ')

# We build now two dictionaries with indexes and unique words, for future reference
unique_words = set(tokens)

word_to_ind = dict((word, i) for i, word in enumerate(unique_words))
ind_to_word = dict((i, word) for i, word in enumerate(unique_words))


def plot_emotions_period(df, cols, title, fname, period = 'h' ):

    df1 = df.groupby(df['datetime'].dt.to_period(period)).mean()

    df1.reset_index(inplace=True)
    df1['datetime'] = pd.PeriodIndex(df1['datetime']).to_timestamp()
    plot_df = pd.DataFrame(df1, df1.index, cols)

    plt.figure(figsize=(15, 10))
    ax = sns.lineplot(data=plot_df, linewidth = 3,dashes = False)
    plt.legend(loc='best', fontsize=15)
    plt.title(title, fontsize=20)
    plt.xlabel('Time (hours)', fontsize=15)
    plt.ylabel('Z-scored Emotions', fontsize=15)
    plt.savefig('images/'+ fname  + '.png')       
    
    return


def get_tweet_emotions(df, emotions, col):

    df_tweets = df.copy()
    df_tweets.drop(['sentiment','sentiment_intensity'], axis=1, inplace=True)
    
    emo_info = {'emotion':'' , 'emo_frq': defaultdict(int) }    

    list_emotion_counts = []

    # creating a dictionary list to hold the frequency of the words
    # contributing to the emotions
    for emotion in emotions:
        emo_info = {}
        emo_info['emotion'] = emotion
        emo_info['emo_frq'] = defaultdict(int)
        list_emotion_counts.append(emo_info)
    
    # bulding a zeros matrix to hold the emotions data
    df_emotions = pd.DataFrame(0, index=df.index, columns=emotions)

    
    # stemming the word to facilitate the search in NRC
    stemmer = SnowballStemmer("english")
    
    # iterating in the tweets data set
    for i, row in df_tweets.iterrows(): # for each tweet ...
        tweet = word_tokenize(df_tweets.loc[i][col])
        for word in tweet: # for each word ...
            word_stemmed = stemmer.stem(word.lower())
            # check if the word is in NRC
            result = ncr[ncr.English == word_stemmed]
            # we have a match
            if not result.empty:
                # update the tweet-emotions counts
                for idx, emotion in enumerate(emotions):
                    df_emotions.at[i, emotion] += result[emotion]
                    
                    # update the frequencies dictionary list
                    if result[emotion].any():
                        try:
                            list_emotion_counts[idx]['emo_frq'][word_to_ind[word]] += 1
                        except:
                            continue
    
    # append the emotions matrix to the tweets data set
    df_tweets = pd.concat([df_tweets, df_emotions], axis=1)

    return df_tweets, list_emotion_counts


# Create a list of words to highlight 
def get_words(word_list, emotions):
    
    words_emotion_idx = []
    
    for i, word in enumerate(word_list):
        word = stemmer.stem(word.lower())
        result = ncr[ncr.English == word]
        if not result.empty:
            for emotion in emotions:
                if result[emotion].any() > 0:
                    words_emotion_idx.append(i)
                
    return words_emotion_idx


def get_top_emotion_words(word_counts, n = 5):

    # Here I map the numpy array "words" with the index and word frequency
    words = np.zeros((len(word_counts), 2), dtype=int)
    for i, w in enumerate(word_counts):
        words[i][0] = w
        words[i][1] = word_counts[w]

    # From the indexes generated by the argsort function, 
    # I get the order of the top n words in the list
    top_words_idx = np.flip(np.argsort(words[:,1])[-n:],0)

    # The resulting indexes are now used as keys in the dic to get the words
    top_words = [words[ind][0] for ind in top_words_idx]
    
    return words, top_words, top_words_idx


# This is now the function to display and highlight 
# the words associated to specific emotions
def print_colored_emotions(tweets, emotions, color, on_color):
    
    for tweet in tweets:

        word_list = word_tokenize(tweet)

        word_emotion_idx = get_words(word_list, emotions)

        for i, w in enumerate(word_list):
            if i in word_emotion_idx:
                w=colored(w, color=color, on_color=on_color)
            print(w, end='') 
            print(' ', end='')  

        print('\n')

        
    return


# Connecting words to emotions

# We are using the NCR lexicon to associate words to emotions 
# Be patient, this will take some time ...

df_emo, list_emotion_counts = get_tweet_emotions(tweets, emotions, 'edited')

# Preparing for time series
df_emo['datetime']= pd.to_datetime(df_emo['datetime']) 


# For a better understanding of the word-emotions associations, 
# I produce here the plots showing what are the 10 words 
# that contributed the most for each of the 8 emotions.

# Plotting the 10 words that contribute the most for each of the 8 emotions
fig, axs = plt.subplots(figsize=(15, 25), frameon=False) 
plt.box(False)
plt.axis('off')
plt.subplots_adjust(hspace = 1.6)
counter = 0

for i, emotion in enumerate(emotions): # for each emotioin

    # This is the dict that holds the top 10 words 
    words, top_words, top_words_indices = get_top_emotion_words(list_emotion_counts[i]['emo_frq'], 10)
    
    info = {'values' : [words[ind][1] for ind in top_words_indices], 
                      'labels' : [ind_to_word[word] for word in top_words]}
    sns.set(style="whitegrid")
    sns.set_context("notebook", font_scale=1.25)
    ax = fig.add_subplot(4, 2, counter+1) # plot 2 charts in each of the 4 rows
    sns.despine()
    ax = sns.barplot(x='labels', y='values', data=info, palette=("cividis"))
    plt.ylabel('Top words', fontsize=12)
    ax.set_title(label=str('Emotion: ') + emotion, fontweight='bold', size=13)
    plt.xticks(rotation=45, fontsize=14)
    counter += 1

axs.set_title(label='\nTop 10 words for each emotion\n', 
             fontweight='bold', size=20, pad=40)
plt.tight_layout()
plt.savefig('images/Top10_words_per_emotion.png')


# Aggregating negative and positive emotions
df_emo['neg_emotions'] = df_emo['Sadness'] + df_emo['Fear'] + df_emo['Disgust'] + df_emo['Anger']
df_emo['pos_emotions'] = df_emo['Joy'] + df_emo['Anticipation'] + df_emo['Trust'] + df_emo['Surprise']

df_emo['total_neg_emotions'] = df_emo['neg_emotions'].apply(lambda x: x > 0)
df_emo['total_pos_emotions'] = df_emo['pos_emotions'].apply(lambda x: x > 0)


# I use here the pandas groupby feature to obtain a normalized account of the emotions 
# as a proportion that takes into account the number of tweets in each of the two periods
# (before and during the online storm).

props = df_emo.groupby('onlinestorm')['total_neg_emotions'].value_counts(normalize=True).unstack()
props

# plot it
plot_fractions(props,'Percentage of tweets with negative emotions','Percentage_of_Tweets_with_negative_emotions')

props = df_emo.groupby('onlinestorm')['total_pos_emotions'].value_counts(normalize=True).unstack()
props

plot_fractions(props,'Percentage of tweets with positive emotions','Percentage_of_Tweets_with_positive_emotions')


# Word - emotion connections in the tweets

df = df_emo[df_emo['Sadness'] > 3]
print_colored_emotions(df['text'], ['Disgust','Sadness','Anger','Fear'], 'white', 'on_red')

# And here some positive ones ...
df = df_emo[df_emo['Anticipation'] > 4]
print_colored_emotions(df['text'], ['Joy','Trust','Anticipation'], 'white', 'on_green')


# Proportion of emotions in relation to number of tweets, before and during the online storm 

df1 = df_emo.groupby(df_emo['onlinestorm'])[emotions].apply(lambda x:( x.sum()/x.count())*100)

df1.index = ['before_onlinestorm', 'during_onlinestorm']

df1.head()

df_ =df1.T
df_.reset_index()


fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.set_title(label='Comparing percentage of emotion-related words before and during online storm\n', 
    fontweight='bold', size=18)
df_.reset_index().plot(
    x="index", y=["before_onlinestorm", "during_onlinestorm"], kind="bar", ax=ax
)

plt.xlabel("Emotions",fontsize = 16)
plt.ylabel("Percentage of emotion-related words",fontsize = 16)
plt.xticks(rotation=45,fontsize=14)
plt.tight_layout()
plt.savefig('images/Percentage_emotions_before_and_during_onlinestorm.png')

# Applying a Z-score normalization
df_zscore = df_emo.groupby(df_emo['onlinestorm'])[emotions].apply(lambda x:(x - x.mean()) / x.std())  
df_emo = pd.concat([df_emo[['datetime','text','edited', 'onlinestorm']], df_zscore], axis=1)
df_emo.head()

plot_emotions_period(df_emo[df_emo['onlinestorm']], emotions,
                    'Emotions time series during online storm','Timeseries_Emotions_OnlineStorm')

# Plotting emotions during online storm

fig, axs = plt.subplots(figsize=(15, 25), frameon=False) 
plt.box(False)
plt.axis('off')
plt.subplots_adjust(hspace = 1.6)
counter = 0

df = df_emo[df_emo['onlinestorm']]
df1 = df.groupby(df['datetime'].dt.to_period('h')).mean()
df1.reset_index(inplace=True)
df1['datetime'] = pd.PeriodIndex(df1['datetime']).to_timestamp()

for i, emotion in enumerate(emotions): # for each emotion
    
    emo = []  
    emo.append(emotion)
    plot_df = pd.DataFrame(df1, df1.index, emo)
    
    sns.set(style="whitegrid")
    sns.set_context("notebook", font_scale=1.25)
    ax = fig.add_subplot(4, 2, counter+1) # plot 2 charts in each of the 4 rows
    sns.despine()
    ax = sns.lineplot(data=plot_df, linewidth = 3,dashes = False)
    plt.ylabel('Time by the hour', fontsize=12)
    ax.set_title(label=str('Emotion: ') + emotion, fontweight='bold', size=13)
    counter += 1

axs.set_title(label='\nPlot for each emotion during online storm\n', 
             fontweight='bold', size=20, pad=40)
plt.tight_layout()
plt.savefig('images/Emotions_during_onlinestorm.png')


# Another way of looking at it is by plotting contrasts of emotions, like joy and sadness ...
plot_emotions_period(df_emo[df_emo['onlinestorm']], ['Joy', 'Sadness'],
                    'Joy and Sadness time series during online storm','Joy_Sadness_Emotions_OnlineStorm')


# And now trust and fear ...
plot_emotions_period(df_emo[df_emo['onlinestorm']], ['Trust', 'Fear'],
                    'Trust and Fear time series during online storm','Trust_Fear_Emotions_OnlineStorm')

