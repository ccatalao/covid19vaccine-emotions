#!/usr/bin/env python
# coding: utf-8

# # Emotional Sentiment on Twitter
# ## CureVac: A coronavirus vaccine online firestorm

# <img src="../images/Average_sentiment_during_onlinestorm.png">

# The ongoing competition for a viable vaccine against coronavirus is arguably the race of the century. With its hundred millions of users, Twitter is particularly well-suited for research into sentiment and emotions running in social media. 

# This is how it all begun: an exercise of 'real politiks' that is likely to change dramatically the way science, politics and business colide in a pos-covid19 world. As we will see below, the 15th March 2020 will go down in history as a shift of political tone that is at odds with the collaborative, responsible and ethical behaviour of scientific research.

# I collected the data scraping tweets from Twitter’s application program inter-face (API), using TwitterScraping. Tweets were saved on a daily basis using the fol-lowing search term “Curevac”, the name of a German vaccine maker backed by Bill & Melinda Gates Foundation, and currently working on a Covid-19 vaccine. The post covers tweets from a 6-year period from March 3, 2014 to March 18, 2020 (N = 15,036).

# The post covers tweets from a 6-year period from March 3, 2014 to March 18, 2020.

# Results include 15,036 tweets in a wide range of languages. 

# In this notebook you will find examples of some of the most common NLP (Natural Language Processing) techniques used to uncover patterns of sentiment and emotion in the kind of unstructured data that is predominant in Twitter. It is organized as follows:

# - Step 1: Exploratory analysis
# - Step 2: Text processing
# - Step 3: Word frequency
# - Step 4: LDA Topics extraction
# - Step 5: Sentiment analysis
# - Step 6: Emotion analysis
# 

# ## Step 1:  EXPLORATORY ANALYSIS

# After scrapping the Twitter API, the retained tweets were gathered in an excel file (tweets_curevac.xlsx).

# Below we have the major Python packages required for data handling (pandas), scientific computing (numpy) and data visualization (matplotlib and seaborn).

# In[1]:


import pandas as pd 
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from collections import defaultdict
from datetime import date

import re # for regular expressions
import string


# In[3]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Let us start by reading the data and drop some unused fields.

# In[4]:


tweets = pd.read_excel('../input/Tweets_CureVac.xlsx')
# droping unused fields
tweets.drop(tweets.columns[tweets.columns.str.contains('Unnamed',case = False)],axis = 1, inplace = True)
tweets.drop(['ID','usernameTweet','user_id','lang'], axis=1, inplace=True)


# In[5]:


# droping missing data
tweets.dropna(inplace=True)


# In[6]:


# getting the date column ready for datetime operations
tweets['datetime']= pd.to_datetime(tweets['datetime'])


# Here is a view of the first rows:

# In[7]:


tweets.head()


# And here is a plot of the tweets with thw word "CureVac" over the past 6 years.

# In[8]:


# A simple timeseries plot
fig = plt.figure(figsize=(15, 10))
ax = sns.lineplot(data=tweets.set_index("datetime").groupby(pd.Grouper(freq='Y')).count())
plt.title('Tweets with "CureVac" from 2014 to 2020', fontsize=20)
plt.xlabel('Years', fontsize=15)
plt.ylabel('Tweets', fontsize=15)
fig.savefig("../images/All_Tweets_2014-2020.png")


# For several years, the rate of tweets went on at a regular pace, until one day ... everything changed!

# Digital marketing researchers call these events “online firestorms”, referring to negative word of mouth (eWOM) that suddenly attract thousands of expres-sions of support from other clients through social [1].

# <img src="../images/fig01_washingtonpost.png">

# Let us create a column to identify this three-days event.
# 

# In[9]:


# creating a column to hold True for date between 15 and 18 March
for i, row in tweets.iterrows():
	if pd.to_datetime(tweets.at[i, 'datetime']) > pd.Timestamp(date(2020,3,15)):
		tweets.at[i, 'onlinestorm'] = True
	else:
		tweets.at[i, 'onlinestorm'] = False  


# In[10]:


# count tweets during the three days online storm
print('In three days, tweets went over {}, all around the world.'.format(tweets[tweets['onlinestorm']]['onlinestorm'].count()))


# Here we have them ..

# In[11]:


tweets[tweets['onlinestorm']]


# Let's have a look at the distribution of tweets by the hour.

# In[12]:


# plot it
fig = plt.figure(figsize=(15, 10))
ax = sns.lineplot(data=tweets[tweets['onlinestorm'] == True].set_index("datetime").groupby(pd.Grouper(freq='H')).onlinestorm.count())
plt.title('Tweets per hour from 15 to 18 March 2020', fontsize=20)
plt.xlabel('Time per hour', fontsize=15)
plt.ylabel('No. Tweets', fontsize=15)
fig.savefig("../images/All_Tweets_15-18_March_2020.png")


# It is now time to have a first look at the content of the tweets and do some descriptive statistics. For now, I will focus only on features like hastags, mentions, urls, capital words and words in general.

# In[13]:


# A function to count tweets based on regular expressions
def count_tweets(reg_expression, tweet):
	tweets_list = re.findall(reg_expression, tweet)
	return len(tweets_list)


# In[14]:


# Creating a dictionary to hold the counts
content_count = {
	'words' : tweets['text'].apply(lambda x: count_tweets(r'\w+', x)),
	'mentions' : tweets['text'].apply(lambda x: count_tweets(r'@\w+', x)),
	'hashtags' : tweets['text'].apply(lambda x: count_tweets(r'#\w+', x)),
	'urls' : tweets['text'].apply(lambda x: count_tweets(r'http.?://[^\s]+[\s]?', x)),   
}


# In[15]:


df = pd.concat([tweets, pd.DataFrame(content_count)], axis=1)


# In[16]:


df


# In[17]:


# Display tweets' descriptive statistics  
for key in content_count.keys():
	print('Descriptive statistics for {}'.format(key))
	print(df.groupby('onlinestorm')[key].describe())


# In[18]:


# Now plot them 
for key in content_count.keys():

	bins = np.arange(df[key].min(), df[key].max() + 1)
	g = sns.FacetGrid(df, col='onlinestorm', height=5, hue='onlinestorm', palette="RdYlGn")
	g = g.map(sns.distplot, key, kde=False, norm_hist=True, bins=bins)
	plt.savefig('../images/Descriptive_stats_for_' + key + '.png')


# In[ ]:





# ## Part 2: TEXT PROCESSING

# In[ ]:





# In[19]:


import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag


# In[20]:


import spacy
from spacy.lang.en import English


# In[51]:


import string
import re
from collections import Counter    # Look at the most common item in a list


# In[21]:


MY_STOPWORDS = ['curevac','vaccine','german','mrna','biotech','cancer','lilly','eli','ag','etherna_immuno', 'translatebio', 'mooreorless62','boehringer', 'ingelheim','biopharmaceutical', 'company']
STOPLIST = set(stopwords.words('english') + list(MY_STOPWORDS))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "``", ",", ".", ":", "''","#","@"]


# In[22]:


lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')


# In[23]:


# read english selected tweets, no duplicates
tweets = pd.read_excel('../input/Tweets_CureVac_en.xlsx')

#remove unused fields
tweets.drop(tweets.columns[tweets.columns.str.contains('Unnamed',case = False)],axis = 1, inplace = True)
tweets.drop(['ID','user_id','usernameTweet'], axis=1, inplace=True)
tweets.dropna(inplace=True)
tweets.head()


# In[24]:


def penn2morphy(penntag):
	#Penn Treebank to WordNet
	morphy_tag = {'NN':'n', 'JJ':'a', 'VB':'v', 'RB':'r'}
	try:
		return morphy_tag[penntag[:2]]
	except:
		return 'n' 


# In[25]:


def get_lemmas(tweet): 
	
	lemmas_list = []
	
	for w, t in pos_tag(word_tokenize(tweet)):
		if t.startswith("JJ") or t.startswith("RB") or t.startswith("VB") or t.startswith("NN") :         
			pos = penn2morphy(t)
			lemmas_list.append(lemmatizer.lemmatize(w.lower(), pos))    
	return lemmas_list


# In[26]:


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


# In[27]:


# Cleanning text function
def get_lemmatized(tweet):
   
	all_tokens_string = ''
	filtered = []
	tokens = []

	# lemmatize
	tokens = [token for token in get_lemmas(tweet)]
	
		# filter
	filtered = clean_tweet(tokens)

	all_tokens_string = ' '.join(filtered)
	
	return all_tokens_string


# In[28]:


# get lemmatized tweet and put it in an "edited" text column
edited = ''
for i, row in tweets.iterrows():
	edited = get_lemmatized(tweets.loc[i]['text'])
	if len(edited) > 0:
		tweets.at[i,'edited'] = edited
	else:
		tweets.at[i,'edited'] = None        


# In[29]:


tweets.drop_duplicates(subset=['edited'], inplace=True)
tweets.dropna(inplace=True)
tweets.to_csv('../input/tweets_edited.csv')
tweets.head()


# ## Step 3: SENTIMENT ANALYSIS

# In[30]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[31]:


# read english selected tweets, no duplicates
tweets = pd.read_csv('../input/tweets_edited.csv')

tweets.dropna(inplace=True)


# In[32]:


tweets['datetime']=pd.to_datetime(tweets['datetime']) 
tweets.sort_values('datetime', inplace=True, ascending=True)
tweets = tweets.reset_index(drop=True)


# In[ ]:





# In[33]:


# creating a column to "filter" dates between 15 and 18 March: thebonline storm
for i, row in tweets.iterrows():
	if pd.to_datetime(tweets.at[i, 'datetime']) > pd.Timestamp(date(2020,3,15)):
		tweets.at[i, 'onlinestorm'] = True
	else:
		tweets.at[i, 'onlinestorm'] = False  


# In[34]:


def plot_sentiment_period(df, info):
	
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
	plt.savefig('../images/' + info['fname'])    
	return


# In[35]:


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
	plt.savefig('../images/' + info['fname'])
	
	return


# In[36]:


# Calling VADER
analyzer = SentimentIntensityAnalyzer()


# In[37]:


# get VADER Compound value for sentiment intensity
tweets['sentiment_intensity'] = [analyzer.polarity_scores(v)['compound'] for v in tweets['edited']]


# The output of VADER are the positive, negative, and neutral ratios of sentiment. The most useful metric is the compound score, which is computed by summing the valence scores of each word in the lexicon, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive). This can be considered a ‘normalized, weighted composite score’.
# 
# The threshold values for the compound score are as follows:
# 
# * Positive sentiment : (compound score >= 0.05).
# 
# * Neutral sentiment : (compound score > -0.05) and (compound score < 0.05).
# 
# * Negative sentiment : (compound score <= -0.05)
# 
# Since we want to calculate the sentiment of an entire tweet, we weight scores at the sentence and word level to get the weighted compound scores.

# In[38]:


def get_sentiment(intensity):
	if intensity >= 0.05:
		return 'Positive'
	elif (intensity >= -0.05) and (intensity < 0.05):
		return 'Neutral'
	else:
		return 'Negative'
		
tweets['sentiment'] = tweets.apply(lambda x: get_sentiment(x['sentiment_intensity']),axis=1)


# ## The Online Storm

# In[39]:


df=tweets.loc[:,['datetime','sentiment_intensity']]
df.set_index('datetime',inplace=True)
df=df[(df.index>='2020-03-12') & (df.index<'2020-03-18')]
df.plot(figsize=(12,5));
plt.ylabel('Sentiment Intensity')
plt.legend().set_visible(False)
plt.tight_layout()
sns.despine(top=True)
plt.savefig('../images/Average_sentiment_during_onlinestorm.png')   
plt.show();


# In[40]:


props = tweets.groupby('onlinestorm')['sentiment'].value_counts(normalize=True).unstack()
plt1 = props.plot(kind='bar', stacked=False, figsize=(16,5), colormap='Spectral') 

plt.legend(bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)
plt.xlabel('Online storm', fontweight='bold', fontsize=18)
plt.xticks(rotation=0,fontsize=14)
plt.ylim(0, 0.5)
plt.ylabel('Fraction of Tweets', fontweight='bold', fontsize=18)
plt1.set_title(label='Fraction of tweets, per online storm', fontweight='bold', size=20)
plt.tight_layout()
plt.savefig('../images/percentage_tweets_vs_online_storm.png')


# In[ ]:





# ## Step 4: Word Frequency

# In[72]:


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer


# In[86]:


from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[73]:


from matplotlib.colors import makeMappingArray
from palettable.colorbrewer.diverging import Spectral_4


# ### Some basic word statistics 

# In[74]:


tweets['word_count'] = tweets.apply(lambda x: len(x['text'].split()),axis=1)
t = pd.DataFrame(tweets['word_count'].describe()).T
t


# In[ ]:





# In[75]:


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
	plt.savefig('../images/' + info['fname'])
	
	return


# In[ ]:





# In[76]:


def display_wordcloud(tokens):
	
	tokens_upper = [token.upper() for token in tokens]

	cloud_mask = np.array(Image.open("../images/cloud_mask.png"))
	wordcloud = WordCloud(max_font_size=100, 
						  max_words=50, width=2500, 
						  height=1750,mask=cloud_mask, 
						  background_color="white").generate(" ".join(tokens_upper))
	plt.figure()
	fig, ax = plt.subplots(figsize=(14, 8))
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	plt.show()

	return


# In[ ]:





# In[77]:


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


# In[ ]:





# In[78]:


# Susan Li's predefined functions

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


# In[ ]:





# In[79]:


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


# In[ ]:





# ## Before the online storm

# In[80]:


df = tweets[tweets['onlinestorm'] == False]


# In[81]:


# Merging all the requests into a single line
text_merged = ''
for line in df['edited']:
	text_merged = text_merged + ' ' + line
	
# get tokens and trigrams
tokens = text_merged.split(' ')
trigrams = nltk.trigrams(tokens)


# In[ ]:





# In[82]:


# plot word frequency before online storm
word_counter = Counter(tokens)
df_counter = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
info = {'data': df_counter, 'x': 'freq', 'y': 'word',
	   'xlab': 'Count', 'ylab': 'Words', 'pal':'viridis',
	   'title': 'Most frequent words before online storm',
	   'fname':'word_frequency_before_onlinestorm.png',
	   'angle': 90}
plot_frequency_chart(info)


# In[ ]:





# In[83]:


# plot trigram frequency
df_trigrams = get_trigrams(trigrams, 10)
info = {'data': df_trigrams, 'x': 'Grams', 'y': 'Count',
	   'xlab': 'Trigrams', 'ylab': 'Count', 'pal':'viridis',
	   'title': 'Most frequent trigrams before online storm',
	   'fname':'trigrams_frequency_before_onlinestorm.png',
	   'angle': 40}
plot_frequency_chart(info)


# In[ ]:





# In[87]:


display_wordcloud(tokens)


# In[85]:


# LDA topics
df_topics = get_topics(df['edited'], 5, 5)
info = {'data': df_topics, 'x': 'Topics', 'y': 'Count',
	   'xlab': 'Topics', 'ylab': 'Count', 'pal':'viridis',
	   'title': 'Main Topics before Online Storm',
	   'fname':'LDA_Topics_before_onlinestorm.png',
	   'angle': 40}
plot_frequency_chart(info)


# In[ ]:





# ## During the online storm

# In[89]:


df =tweets[tweets['onlinestorm']]


# In[91]:


# Merging all the requests into a single line
text_merged = ''
for line in df['edited']:
	text_merged = text_merged + ' ' + line
	
# get tokens and trigrams
tokens = text_merged.split(' ')
trigrams = nltk.trigrams(tokens)


# In[92]:


# plot word frequency during online storm
word_counter = Counter(tokens)
df_counter = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
info = {'data': df_counter, 'x': 'freq', 'y': 'word',
	   'xlab': 'Count', 'ylab': 'Words', 'pal':'inferno',
	   'title': 'Most frequent words during online storm',
	   'fname':'word_frequency_during_onlinestorm.png',
	   'angle': 90}
plot_frequency_chart(info)


# In[93]:


# plot trigrams frequency
df_trigrams = get_trigrams(trigrams, 10)
info = {'data': df_trigrams, 'x': 'Grams', 'y': 'Count',
	   'xlab': 'Trigrams', 'ylab': 'Count', 'pal':'inferno',
	   'title': 'Most frequent trigrams during online storm',
	   'fname':'trigrams_frequency_during_onlinestorm.png',
	   'angle': 40}
plot_frequency_chart(info)


# In[94]:


display_wordcloud(tokens)


# In[95]:


# LDA topics
df_topics = get_topics(df['edited'], 5, 5)
info = {'data': df_topics, 'x': 'Topics', 'y': 'Count',
	   'xlab': 'Topics', 'ylab': 'Count', 'pal':'inferno',
	   'title': 'Main Topics during Online Storm',
	   'fname':'LDA_Topics_during_onlinestorm.png',
	   'angle': 40}
plot_frequency_chart(info)


# ## Step 5: EMOTIONS ANALYSIS

# In[97]:


import termcolor
import sys
from termcolor import colored, cprint
plt.style.use('fivethirtyeight')


# In[ ]:





# In[98]:


ncr = pd.read_csv('../input/NCR-lexicon.csv', sep =';')


# In[100]:


emotions = ['Anger', 'Anticipation','Disgust','Fear', 'Joy','Sadness', 'Surprise', 'Trust']


# In[101]:


# Merging all the requests into a single line
all_text = ''
for line in tweets['edited']:
	all_text = all_text + ' ' + line
	
# get tokens and trigrams
tokens = all_text.split(' ')

# Criam-se agora, a partir das palavras únicas, os dicionários de índices para referência futura
unique_words = set(tokens)
word_to_ind = dict((word, i) for i, word in enumerate(unique_words))
ind_to_word = dict((i, word) for i, word in enumerate(unique_words))


# In[102]:


def plot_emotions_period(df, cols, period = 'h' ):

	df1 = df.groupby(df['datetime'].dt.to_period(period)).mean()

	df1.reset_index(inplace=True)
	df1['datetime'] = pd.PeriodIndex(df1['datetime']).to_timestamp()
	plot_df = pd.DataFrame(df1, df1.index, cols)

	plt.figure(figsize=(15, 10))
	ax = sns.lineplot(data=plot_df, linewidth = 3,dashes = False)
	plt.legend(loc='best', fontsize=15)
	plt.title('Emotions in Tweets with CureVac', fontsize=20)
	plt.xlabel('15 March', fontsize=15)
	plt.ylabel('Average Emotions', fontsize=15)
	plt.savefig('../images/Emotions_during_onlinestorm.png')       
	return


# In[ ]:





# In[109]:


def get_tweet_emotions(df, emotions, col):

	df_base = df.copy()
	emo_info = {'emotion':'' , 'emo_frq': defaultdict(int) }    

	list_emotion_counts = []

	for emotion in emotions:
		emo_info = {}
		emo_info['emotion'] = emotion
		emo_info['emo_frq'] = defaultdict(int)
		list_emotion_counts.append(emo_info)
	
	#criamos um dataframe de zeros com a dimensão de df
	df_emotions = pd.DataFrame(0, index=df.index, columns=emotions)

	stemmer = SnowballStemmer("english")
	x = 0
	for i, row in df_base.iterrows():
		tweet = word_tokenize(df_base.loc[i][col])
		for word in tweet:
			word_stemmed = stemmer.stem(word.lower())
			result = ncr[ncr.English == word_stemmed]
			if not result.empty:
				for idx, emotion in enumerate(emotions):
					df_emotions.at[i, emotion] += result[emotion]
									   
					if result[emotion].any():
						try:
							list_emotion_counts[idx]['emo_frq'][word_to_ind[word]] += 1
						except:
							continue
								
	df_base = pd.concat([df_base, df_emotions], axis=1)

	return df_base, list_emotion_counts


# In[ ]:





# In[106]:



def get_top_emotion_words(word_counts, n = 5):

	# Passamos finalmente o dicionário para uma numpy array "words", com o indice da palavra e respectiva frequência
	words = np.zeros((len(word_counts), 2), dtype=int)
	for i, w in enumerate(word_counts):
		words[i][0] = w
		words[i][1] = word_counts[w]

	# A partir dos indices gerados pela função argsort, sabemos a posição 
	# das "n" palavras mais frequentes na array words
	top_words_indices = np.flip(np.argsort(words[:,1])[-n:],0)

	# Com estas posições (indices), obtemos os indices que funcionam como keys no dicionário ind_to_word,
	# e nos devolvem, como "value", as palavras como strings
	top_words = [words[ind][0] for ind in top_words_indices]
		
	return words, top_words, top_words_indices


# In[ ]:





# In[110]:


# Criação de uma dataframe de tweets associados a emoções
# A execução pode ser longa, dependendo do tamanho da amostra de tweets

df_emo, list_emotion_counts = get_tweet_emotions(tweets, emotions, 'edited')

#df_emo = pd.read_csv('../data/df_emotions.csv')
df_emo['datetime']= pd.to_datetime(df_emo['datetime']) 

# gravar em csv
df_emo.to_csv('../input/tweets_emotions.csv')


# In[ ]:





# In[111]:


###############################################################################
# Mostra distribuição das 10 palavras mais frequentes em cada emoção
############################################################################### 

fig, axs = plt.subplots(figsize=(15, 25), frameon=False) 
plt.box(False)
plt.axis('off')
plt.subplots_adjust(hspace = 1.6)
counter = 0

for i, emotion in enumerate(emotions): # para cada emoção ...

	# Para esta emoção, cria um dicionário com as 10 palavras mais frequentes
	words, top_words, top_words_indices = get_top_emotion_words(list_emotion_counts[i]['emo_frq'], 10)
	dados = {'valores' : [words[ind][1] for ind in top_words_indices], 
					  'labels' : [ind_to_word[word] for word in top_words]}
	sns.set(style="whitegrid")
	sns.set_context("notebook", font_scale=1.25)
	ax = fig.add_subplot(4, 2, counter+1) # faz 4 gráficos por linha
	sns.despine()
	ax = sns.barplot(x='labels', y='valores', data=dados, palette=("cividis"))
	plt.ylabel('Top words', fontsize=12)
	ax.set_title(label=str('Emotion: ') + emotion, fontweight='bold', size=13)
	plt.xticks(rotation=45, fontsize=14)
	counter += 1

axs.set_title(label='\nFrequência das 10 palavras mais usadas em cada emoção\n', 
			 fontweight='bold', size=20, pad=40)
plt.tight_layout()
plt.savefig('../images/Top10_palavras_em_cada_emocao.png')


# In[ ]:





# ### Filter for March 2020

# In[112]:


plot_emotions_period(df_emo[df_emo['onlinestorm']], emotions)


# In[ ]:





# In[113]:


df_emo['neg_emotions'] = df_emo['Sadness'] + df_emo['Fear'] + df_emo['Disgust'] + df_emo['Anger']
df_emo['pos_emotions'] = df_emo['Joy'] + df_emo['Anticipation'] + df_emo['Trust']


# In[114]:


plot_emotions_period(df_emo[df_emo['onlinestorm']], ['neg_emotions', 'pos_emotions'])


# In[ ]:





# In[115]:


props = df_emo.groupby('onlinestorm')['Anger'].value_counts(normalize=True).unstack()
plt1 = props.plot(kind='bar', stacked=False, figsize=(16,5), colormap='Spectral') 

plt.legend(bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)
plt.xlabel('Online storm', fontweight='bold', fontsize=18)
plt.xticks(rotation=0,fontsize=14)
plt.ylim(0, 0.5)
plt.ylabel('Fraction of Tweets', fontweight='bold', fontsize=18)
plt1.set_title(label='Fraction of tweets, per online storm', fontweight='bold', size=20)
plt.tight_layout()
plt.savefig('../images/percentage_tweets_vs_online_storm.png')


# In[116]:


df_emo['total_neg_emotions'] = df_emo['neg_emotions'].apply(lambda x: x > 0)


# In[117]:


props = df_emo.groupby('onlinestorm')['total_neg_emotions'].value_counts(normalize=True).unstack()
plt1 = props.plot(kind='bar', stacked=False, figsize=(16,5), colormap='Spectral') 

plt.legend(bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)
plt.xlabel('Online storm', fontweight='bold', fontsize=18)
plt.xticks(rotation=0,fontsize=14)
plt.ylim(0, 0.5)
plt.ylabel('Fraction of Tweets', fontweight='bold', fontsize=18)
plt1.set_title(label='Fraction of tweets with negative emotions', fontweight='bold', size=20)
plt.tight_layout()
plt.savefig('../images/percentage_tweets_vs_online_storm.png')


# In[ ]:





# In[118]:


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


# In[ ]:





# In[119]:


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


# In[ ]:





# In[125]:


df = df_emo[df_emo['Sadness'] > 3]
print_colored_emotions(df['text'], ['Disgust','Sadness','Anger','Fear'], 'white', 'on_red')


# In[122]:


df = df_emo[df_emo['Anticipation'] > 4]
print_colored_emotions(df['text'], ['Joy','Trust','Anticipation'], 'white', 'on_green')


# In[ ]:




