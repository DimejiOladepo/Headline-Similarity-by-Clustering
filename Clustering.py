
# coding: utf-8

# In[2]:


#Importing Libraries
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile


# In[91]:


#Importing the dataset and specifying no header
data = "news_training.xlsx"
news_headlines = pd.read_excel(data, header= None)


# In[92]:


#Adding column names
news_headlines.columns = ['Date', 'Headline']
news_headlines.head()


# In[93]:


#Converting datetime to month/day/year format
news_headlines['Date'] = pd.to_datetime(news_headlines['Date'], infer_datetime_format=True)
news_headlines['Date'] = news_headlines['Date'].dt.strftime('%m/%d/%Y')


# In[94]:


news_headlines.head()


# In[96]:


#Import regex library 
import re

#Cleaning Headlines
news_headlines["Headline"] = news_headlines['Headline'].map(lambda x: re.sub(r'^b', '', x))
news_headlines["Headline"] = news_headlines['Headline'].map(lambda x: re.sub(r'\b["",\']', '', x))
news_headlines.head()
                                        


# In[97]:


#Listing dataframe
prev = news_headlines.values.tolist()
prev


# In[11]:


#Pre-processing news headlines 
features = news_headlines.iloc[:,1].values
processed_features = []
for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))
     # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
     # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
     # Converting to Lowercase
    processed_feature = processed_feature.lower()
    processed_features.append(processed_feature)


# In[12]:


processed_features_arr = np.asarray(processed_features)
processed_features_arr


# In[15]:


from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

#Specifying module parameter
stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

#defining function for tokenizing input text
def tokenize(text):
    return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower()) if not word in set(stopwords.words('english'))]


# In[16]:


from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer 
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']',  '{', '}', "%"]

#Vectorizing headlines array 
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
vectorizer = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize)        
X = vectorizer.fit_transform(processed_features_arr)


# In[17]:


#Extracting word features and number of features
word_features = vectorizer.get_feature_names()
print(len(word_features))
print(word_features[:100])


# In[18]:


# Applying K-means and plotting graph to determine Elbow and optimum number of clusters
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init= 'k-means++', max_iter= 100, n_init=5, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[19]:


#Fitting K-means model on vectorized headlines array
from sklearn.cluster import KMeans
true_k = 3
model = KMeans(n_clusters= true_k, init= 'k-means++', max_iter= 100, n_init=1)
model.fit(X)


# In[20]:


# Outputting top 10 terms for each cluster  
print ("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print ("\nCluster %d:" % i, end= " ")
    for ind in order_centroids[i, :10]:
        print (' %s' % terms[ind], end= " ")
    print


# In[153]:


#Predicting cluster for news text 
print("Prediction")
P_text = tokenize("Finance quarter result")
P_text = ' '.join(P_text)
Y = vectorizer.transform([P_text])
predicted = model.predict(Y)
print(predicted)


# In[154]:


#Zipping new headlines list with kmens labels and sorting on labels
labels = model.labels_
result = zip(prev, labels)
sortedR = sorted(result, key=lambda x: x[1], reverse = False)
sortedR


# In[160]:


#Headline history suggestions 
for i in range(0, labels.shape[0]):
    if labels[i] == predicted:
        print(prev[i])
        

