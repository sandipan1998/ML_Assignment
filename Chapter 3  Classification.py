#!/usr/bin/env python
# coding: utf-8

# ## MNIST Dataset Classification Demo

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.datasets import fetch_openml


# In[3]:


mnist = fetch_openml('mnist_784', version=1)


# In[4]:


mnist.keys()


# In[5]:


X=mnist.data
y=mnist.target


# In[6]:


X.shape,y.shape


# In[7]:


y=y.astype('int')


# In[8]:


X_train,X_test,y_train,y_test=X[:60000,:],X[60000:,:],y[:60000],y[60000:]


# In[9]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[10]:


some_digit=X[0]


# In[11]:


some_digit_img=some_digit.reshape(28,-1)


# In[12]:


some_digit_img.shape


# In[13]:


plt.imshow(some_digit_img,cmap="binary")
plt.axis("off")
plt.show()


# In[14]:


y_train_5=(y_train==5)


# In[15]:


y_test_5=(y_test==5)


# In[16]:


from sklearn.linear_model import SGDClassifier


# In[17]:


sgd_clf=SGDClassifier(random_state=42)


# In[18]:


sgd_clf.fit(X_train,y_train_5)


# In[19]:


yhat=sgd_clf.predict(X_train[:1])


# In[20]:


yhat


# In[21]:


y_train_5[:1]


# In[22]:


from sklearn.model_selection import StratifiedKFold


# In[23]:


from sklearn.base import clone


# In[24]:


split=StratifiedKFold(n_splits=3,random_state=42)


# In[25]:


accuracy=[]
for train_index,test_index in split.split(X_train,y_train_5):
    clone_clf=clone(sgd_clf)
    X_training_fold=X_train[train_index]
    y_training_fold=y_train_5[train_index]
    X_testing_fold=X_train[test_index]
    y_testing_fold=y_train_5[test_index]
    clone_clf.fit(X_training_fold,y_training_fold)
    yhat=clone_clf.predict(X_testing_fold)
    n_correct=np.sum(yhat==y_testing_fold)
    accuracy.append(n_correct/len(y_testing_fold))
    


# In[26]:


accuracy


# In[27]:


from sklearn.model_selection import cross_val_score


# In[28]:


accuracy=cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring="accuracy")


# In[29]:


accuracy


# In[30]:


from sklearn.base import BaseEstimator


# In[31]:


class Never5Classifier(BaseEstimator):
    def fit(self,X,y):
        pass
    
    
    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)


# In[32]:


never_5_clf=Never5Classifier()


# In[33]:


never_5_clf.fit(X_train,y_train_5)


# In[34]:


never_5_clf.predict(X_train[:5])


# In[35]:


never_5_accuracy=cross_val_score(never_5_clf,X_train,y_train_5,cv=3,scoring="accuracy")


# In[36]:


never_5_accuracy


# In[37]:


from sklearn.model_selection import cross_val_predict


# In[38]:


y_pred_sgd=cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)


# In[39]:


from sklearn.metrics import confusion_matrix


# In[40]:


conf_mat=confusion_matrix(y_train_5,y_pred_sgd)


# In[41]:


from sklearn.metrics import plot_confusion_matrix


# In[42]:


plot_confusion_matrix(sgd_clf,X_train,y_train_5,values_format='g')


# In[43]:


from sklearn.metrics import f1_score,precision_score,recall_score


# In[44]:


print(f"F1 Score:{f1_score(y_train_5,y_pred_sgd)}")
print(f"Precision Score:{precision_score(y_train_5,y_pred_sgd)}")
print(f"Recall Score:{recall_score(y_train_5,y_pred_sgd)}")


# In[45]:


y_pred_score=cross_val_predict(sgd_clf,X_train,y_train_5,cv=3,method="decision_function")


# In[46]:


from sklearn.metrics import precision_recall_curve


# In[47]:


precision, recall, threshold=precision_recall_curve(y_train_5,y_pred_score)


# In[48]:


threshold.shape,precision.shape


# In[49]:



plt.plot(threshold,precision[:-1],label="precision")
plt.plot(threshold,recall[:-1],label="recall")
plt.legend()
plt.show()


# In[50]:


plt.plot(recall[:-1],precision[:-1])

plt.xlabel("recall")
plt.ylabel("precision")


# In[59]:


threshold_85precision=threshold[np.argmax(precision>=0.65)]


# In[60]:


y_pred_score_class=(y_pred_score>=threshold_85precision)


# In[61]:


print(f"F1 Score:{f1_score(y_train_5,y_pred_score_class)}")
print(f"Precision Score:{precision_score(y_train_5,y_pred_score_class)}")
print(f"Recall Score:{recall_score(y_train_5,y_pred_score_class)}")


# In[66]:


from sklearn.metrics import roc_curve


# In[67]:


fpr,tpr,threshold=roc_curve(y_train_5,y_pred_score)


# In[69]:


plt.plot(fpr,tpr)
plt.xlabel("False +ve rate (1-specificity)")
plt.ylabel("True +ve rate sensitivity")


# In[72]:


from sklearn.metrics import roc_auc_score


# In[73]:


roc_auc_score(y_train_5,y_pred_score)


# In[ ]:




