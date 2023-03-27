#!/usr/bin/env python
# coding: utf-8

# ## İmport libary and Dataset load

# In[93]:


import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity 
import pyspark
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.tuning import ParamGridBuilder , CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import dill as pickle
from joblib import dump, load


import psycopg2
from urllib.parse import urlparse

import findspark
findspark.init()
#İgnore Error
import warnings
warnings.simplefilter(action="ignore")
plt.rcParams['figure.figsize'] = [4, 4]
#plt.figure(figsize =(5, 3))


# param={
#     'dbname':'book',
#     'user': 'masal',
#     'password': 'Masal2020',
#     'port': 5432,
#     'host': 'localhost'
# }
# connection = psycopg2.connect(**param )

# books_sql=pd.read_sql("SELECT * FROM public.books",connection)
# books_sql

# In[2]:


books = pd.read_csv('books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv('users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']


# In[3]:


print(users.shape)
print(books.shape)
print(ratings.shape)


# In[4]:


books.info()


# In[5]:


books.head()


# ## Exploration Data Analiysis

# #### Books Dataset check

# In[6]:


#BOOK VALUE CHECK
total = books.isnull().sum().sort_values(ascending=False)
percent_1 = books.isnull().sum()/books.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(12)


# sns.boxplot(x=books.yearOfPublication)#I want learn anomaly data year publication

# books.yearOfPublication=books.yearOfPublication.astype(int)#I cant convert because it has got string value 

# In[7]:


books.loc[books.yearOfPublication == 'DK Publishing Inc',:]


# In[8]:


#update wrong value 
#ISBN 0789466953
books.loc[books.ISBN == '0789466953','yearOfPublication'] = 2000
books.loc[books.ISBN == '0789466953','bookAuthor'] = "James Buckley"
books.loc[books.ISBN == '0789466953','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953','bookTitle'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"


# In[9]:


#ISBN '078946697X'
books.loc[books.ISBN == '078946697X','yearOfPublication'] = 2000
books.loc[books.ISBN == '078946697X','bookAuthor'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '078946697X','bookTitle'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"


# In[10]:


books.loc[(books.ISBN == '0789466953') | (books.ISBN == '078946697X'),:]


# In[11]:


#Other anomaly value
books.loc[books.yearOfPublication == 'Gallimard',:]


# In[12]:


#Update ISBN 2070426769
books.loc[books.ISBN == '2070426769','yearOfPublication'] = 2003
books.loc[books.ISBN == '2070426769','bookAuthor'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books.loc[books.ISBN == '2070426769','publisher'] = "Gallimard"
books.loc[books.ISBN == '2070426769','bookTitle'] = "Peuple du ciel, suivi de 'Les Bergers"


# In[13]:


books.yearOfPublication=pd.to_numeric(books.yearOfPublication, errors='coerce') #Convert int


# In[14]:


sns.distplot(books['yearOfPublication'],kde=True,bins=25);
plt.savefig('Year Publisher.png')


# In[15]:


books.yearOfPublication.describe()#Minimum year value 0


# In[16]:


books.loc[(books.yearOfPublication > 2006) | (books.yearOfPublication == 0),'yearOfPublication'] = np.NAN#We learned that the dataset was prepared until 2006


# In[17]:


books.yearOfPublication.fillna(round(books.yearOfPublication.mean()), inplace=True)#use mean values


# In[18]:


books.yearOfPublication.describe()


# In[19]:


books.yearOfPublication = books.yearOfPublication.astype(np.int32)


# In[20]:


#saw 2 null values above
books.loc[books.publisher.isnull(),:]


# In[21]:


#change the other publisher
books.loc[(books.ISBN == '193169656X'),'publisher'] = 'other'
books.loc[(books.ISBN == '1931696993'),'publisher'] = 'other'


# In[22]:


plt.figure(figsize=(10, 5))
plt.title("5 Publishers by Year ")
publisher = books.groupby(['publisher']).yearOfPublication.agg(['sum']).reset_index()
publisher =publisher.nlargest(5,'sum')
sns.barplot(x="sum", y="publisher", orient = "h", data=publisher, palette = "Blues_d")
plt.savefig('Top5publisher.png')


# In[23]:


#USER VALUE CHECK
total = users.isnull().sum().sort_values(ascending=False)
percent_1 = users.isnull().sum()/users.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(12)


# In[24]:


users.Age.describe()#Age anormaly


# In[25]:


users.loc[(users.Age > 90) | (users.Age < 10), 'Age'] = np.nan#min 10 , maximum 90 years


# In[26]:


users.Age=users.Age.fillna(users.Age.mean())


# In[27]:


users.Age.describe()


# In[28]:


users.Age = users.Age.astype(np.int32)


# In[29]:


plt.hist(x=users.Age);
plt.savefig('Agehist.png')


# In[30]:


#Rating Data set valu check
total = ratings.isnull().sum().sort_values(ascending=False)
percent_1 = ratings.isnull().sum()/ratings.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(12)


# In[31]:


ratings.info()


# In[32]:


sns.countplot(x=ratings['bookRating']);#i want to see Rating the distribution


# In[33]:


#if i wanted to get the ones with equal rating isbn numbers with the books
ratings_new = ratings[ratings.ISBN.isin(books.ISBN)]


# In[34]:


print(ratings.shape)
print(ratings_new.shape)


# In[35]:


#our goal is to get user ratings
ratings = ratings[ratings.userID.isin(users.userID)]


# In[36]:


#Since i will create a recommendation system, we cannot recommend a book with 0 ratings.
explicitdf = ratings_new[ratings_new.bookRating != 0]
implicitdf = ratings_new[ratings_new.bookRating == 0]


# In[37]:


sns.countplot(x=explicitdf['bookRating']);


# In[38]:


explicitdf['bookRating'].value_counts(normalize=True)


# In[39]:


explicitdf


# ## Item Based recommendation

# In[40]:


#Top 10 best rated books 
popular_products = pd.DataFrame(explicitdf.groupby('ISBN')['bookRating'].count())
most_popular = popular_products.sort_values('bookRating', ascending=False).head(10)
top10=most_popular.merge(books,left_index = True, right_on = 'ISBN')
top10


# In[41]:


top10.to_pickle("top10.pkl")  


# ## Content based

# In[42]:


#bought 1200 books against and  memory errors because there is a recommendation system.
products = pd.DataFrame(explicitdf.groupby('ISBN')['bookRating'].count())
popular = products.sort_values('bookRating', ascending=False).head(1200)
most=popular.merge(books,left_index = True, right_on = 'ISBN')


# In[43]:


most.bookTitle.nunique()


# In[44]:


n = 1200
top_n=most.bookTitle.value_counts().index[:n]
books_df=most[most['bookTitle'].isin(top_n)]


# In[45]:


books_df.shape


# In[46]:


#Since I will use similarity rates in the content recommendation system, I gathered all the features together.
books_df['text']=books_df['bookTitle']+' '+books_df['bookAuthor']+' '+books_df['publisher']


# In[47]:


books_df['text']


# In[48]:


#I removed the books to avoid recommending the same books.
books_df.drop_duplicates(subset=['text'], inplace=True)
books_df.drop_duplicates(subset=['bookTitle'], inplace=True)
books_df.info()


# In[49]:


books_df.dropna(subset='bookTitle', inplace=True)
books_df.reset_index(drop=True, inplace=True)
books_df


# In[50]:


#I vectorized the book and features to use similarity criteria.
tfidf = TfidfVectorizer(stop_words='english') # 
books_df['text'] = books_df['text'].fillna('')

tfidf_matrix = tfidf.fit_transform(books_df['text'])
tfidf_matrix.shape


# In[51]:


#Cosine similarity rates
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim


# In[52]:


#Create Books İndex
indices = pd.Series(books_df.index, index=books_df['bookTitle'])
indices


# In[53]:


#for Recommendation function
def get_recommendations(book, cosine_sim=cosine_sim):
    
    idx = indices[book] # Her bir biraya karşılık gelen index değerleri

    sim_scores = list(enumerate(cosine_sim[idx])) # Biralar arasındaki ikili benzerlik puanları

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # Benzerlik oranlarına göre sıralama

    sim_scores = sim_scores[1:11] # En benzer 10 bira

    book_indices = [i[0] for i in sim_scores] # Bu biraların index değerleri

    return books_df['bookTitle'].iloc[book_indices] # En benzer 10 birayı göster


# In[54]:


get_recommendations('The Lovely Bones: A Novel')


# In[55]:


get_recommendations('Wild Animus')


# In[56]:


get_recommendations('Airframe')


# In[94]:


#content=pickle.dumps()
dump(get_recommendations, 'content.pkl')


# ## Collabarity conttent

# In[57]:


books_df


# In[58]:


collabarity = books.set_index('ISBN').join(ratings.set_index('ISBN'))


# In[59]:


collabarity


# In[60]:


collabarity.info()


# In[61]:


n = 1200
columns=['imageUrlS','imageUrlM','imageUrlL','yearOfPublication']
collabarity.drop(columns=columns,inplace=True)
top_m = collabarity.bookTitle.value_counts().index[:n] # En çok değerlendirilen 250 bira

collabarity = collabarity[collabarity['bookTitle'].isin(top_n)]
collabarity.head()


# In[62]:


df_wide = pd.pivot_table(collabarity, values=['bookRating'],
        index=['bookTitle', 'userID'],
        aggfunc=np.mean).unstack() # Tablonun görünümünü düzenlemek için küçük bir düzeltme
df_wide.shape


# In[63]:


df_wide


# In[64]:


df_wide = df_wide.fillna(0)
df_wide


# In[65]:


cosine_sim2 = cosine_similarity(df_wide)
cosine_sim2


# In[66]:


cosine_sim2_df = pd.DataFrame(cosine_sim2, index=df_wide.index, columns=df_wide.index)
cosine_sim2_df.head()


# In[95]:


def find_similar_films(films, count=1):
    """
    Parameters
    ----------
    beers: list
        some beer names!
        
    count: int (default=1)
        count of similar beer!
    
    Returns
    -------
    ranked_beers: list
        rank ordered beers
    """
    films = [films for films in films if films in cosine_sim2_df.columns]
    films_summed = cosine_sim2_df[films].apply(lambda row: np.sum(row), axis=1)
    films_summed = films_summed.sort_values(ascending=False) # Yüksek skorlar daha iyi
    ranked_films = films_summed.index[films_summed.index.isin(films)==False]
    ranked_films = ranked_films.tolist()

    if count is None:
        return ranked_films
    else:
        return ranked_films[:count]


# In[96]:


find_similar_films(['Liar'],5)


# In[97]:


find_similar_films(['The Body Farm'],5)


# In[99]:


dump(find_similar_films, 'collabarity.pkl')


