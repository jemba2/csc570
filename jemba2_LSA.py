
# coding: utf-8

# In[1]:

from sklearn.datasets import fetch_20newsgroups
categories = ['rec.sport.baseball']
dataset = fetch_20newsgroups(subset='all',shuffle=True, random_state=42, categories=categories)
corpus = dataset.data


# In[2]:

from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# In[4]:

nltk.download('stopwords')


# In[29]:

stopset = set(stopwords.words('english'))
stopset.update(['lt','p','/p','br','amp','quot','field','font','normal','span','0px','rgb','style','51', 
                'spacing','text','helvetica','size','family', 'space', 'arial', 'height', 'indent', 'letter',
                'line','none','sans','serif','transform','line','variant','weight','times', 'new','strong', 'video', 'title',
                'white','word','letter', 'roman','0pt','16','color','12','14','21', 'neue', 'apple', 'class', 'nntp', '00',
                'posting', '@', 'edu', 'com', '000', 'cs', 'net', 'from', 'subject', 'organization', 'uiuc', 'morris',
                'tc', 'rose', 'jhunix', 'rose', 'duke', 'hulman', 'hcf', 'cc', 'bob', 'cornell', 'stanford', 'hp', 'ca',
                'netcom', 'williams', 'university', 'ted', 'aix', 'ibm', 'scott', 'roger', 'vb30', 'lafibm', '\n'])


# In[30]:

vectorizer = TfidfVectorizer(stop_words=stopset, use_idf=True, ngram_range=(1, 3))
X = vectorizer.fit_transform(corpus)


# In[31]:

X[0]


# In[43]:

print(X[0])


# In[32]:

X.shape


# In[33]:

lsa = TruncatedSVD(n_components=100, n_iter=100)
lsa.fit(X)


# In[34]:

terms = vectorizer.get_feature_names()
for i, comp in enumerate(lsa.components_): 
    termsInComp = zip (terms,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=True) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print (" ")


# In[28]:

corpus[40]


# In[ ]:



