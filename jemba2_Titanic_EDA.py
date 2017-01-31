
# coding: utf-8

# In[50]:

import pandas as pd
get_ipython().magic('pylab inline')


# In[2]:

df = pd.read_csv("train.csv")


# ### EDA
# 
# Learn about the data!
# <li>Is the variable categorical or continuous
# <li>Are there missing values?
# <li>Min, Max, Mean, and Standard Deviation of the continuous variables.
# <li>Histograms describing the distribution of the variable.
# <li>EDA should be completed for these fields:  'PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', and 'Embarked'

# In[7]:

dfAttrib = list(df)
dfAttrib.remove('Name')
dfAttrib


# ### Exploration of the dataset values
# 
# ##### Lets loop through the attributes of the dataset and perform value_counts() to get some context of the data values

# In[58]:

x=0
for i in dfAttrib:   
    print(dfAttrib[x] + ": \n")
    print(df[i].value_counts())
    print("\n")
    x=x+1


# 
# ### Categorical Attributes
# 
# <li> PassengerId
# <li> Survived
# <li> Pclass
# <li> Sex
# <li> SibSp
# <li> Parch
# <li> Embarked
# <li> Ticket
# <li> Cabin
# 
# 
# ### Continuous
# 
# 
# <li> Age
# <li> Fare
# 
# 
# 
# 

# ### More Exploration of the dataset values
# 
# ##### Lets loop through the attributes of the dataset and look for missing values

# In[68]:

y=0
for i in dfAttrib:   
    print(dfAttrib[y] + ": \n")
    print(df[df[i].isnull()])
    print("\n")
    y=y+1


# 
# ### There are missing values for the following attributes:
# 
# <li> Age (177 rows)
# <li> Cabin (687 rows)
# <li> Embarked (2 rows)

# In[69]:

df.describe()


# # Exploration of the continuous variables
# 
# ### Age
# <li> Min = 0.42
# <li> Max = 80.0
# <li> Mean = 29.699118
# <li> Standard Deviation = 14.526497
# ### Fare
# <li> Min = 0.0
# <li> Max = 512.329200
# <li> Mean = 32.204208
# <li> Standard Deviation = 49.693429

# # Histograms and Plots
# 

# In[90]:

df.PassengerId.hist()


# In[91]:

df.Survived.hist()


# In[92]:

df.Pclass.hist()


# In[94]:

df.Age.hist()


# In[95]:

df.SibSp.hist()


# In[96]:

df.Parch.hist()


# In[98]:

df.Fare.hist()


# In[115]:

df.Sex.value_counts().plot(kind='bar')


# In[116]:

df.Cabin.value_counts().plot(kind='bar')


# In[117]:

df.Embarked.value_counts().plot(kind='bar')


# # Additional Analysis
# 

# In[121]:

fig, axs = plt.subplots(1,2)
df[df.Survived ==1].Age.value_counts().plot(kind='barh', ax=axs[0], title="Age of Survivors")
df[df.Survived ==0].Age.value_counts().plot(kind='barh', ax=axs[1], title="Age of those who died")


# In[122]:

fig, axs = plt.subplots(1,2)
df[df.Survived ==1].Embarked.value_counts().plot(kind='barh', ax=axs[0], title="Survivors by port")
df[df.Survived ==0].Embarked.value_counts().plot(kind='barh', ax=axs[1], title="Fatalities by port")


# In[123]:

fig, axs = plt.subplots(1,2)
df[df.Survived ==1].Pclass.value_counts().plot(kind='barh', ax=axs[0], title="Survivors by Pclass")
df[df.Survived ==0].Pclass.value_counts().plot(kind='barh', ax=axs[1], title="Fatalities by Pclass")


# In[126]:

fig, axs = plt.subplots(1,2)
df[df.Fare > 32.204208].Survived.value_counts().plot(kind='barh', ax=axs[0], title="Survivors with above average fare")
df[df.Fare < 32.204208].Survived.value_counts().plot(kind='barh', ax=axs[1], title="Survivors below average fare")


# In[ ]:



