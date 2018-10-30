
# coding: utf-8

# In[ ]:


#https://www.digitalocean.com/community/tutorials/how-to-build-a-machine-learning-classifier-in-python-with-scikit-learn


# In[3]:


import sklearn
from sklearn.datasets import load_breast_cancer


# In[4]:


data = load_breast_cancer()


# In[5]:


data


# In[6]:


feature_names = data['feature_names']
features = data['data']

label_names = data['target_names']
labels = data['target']


# In[7]:


labels


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.10,
                                                          random_state=42)


# In[16]:


from sklearn.naive_bayes import GaussianNB


# In[17]:


gnb = GaussianNB()


# In[18]:


train_model = gnb.fit(train, train_labels)


# In[19]:


test_model = gnb.predict(test)


# In[20]:


from sklearn.metrics import accuracy_score


# In[21]:


accuracy = accuracy_score(test_model, test_labels)

