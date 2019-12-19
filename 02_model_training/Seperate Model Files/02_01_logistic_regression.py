#!/usr/bin/env python
# coding: utf-8

# # Pet Adoption Rating Prediction
#
# **Course :** CS596 - Machine Learning
#
# **Name:** Amol Kolhe **Red ID:** 822549722
#
# **Name:** Saumil Shah **Red ID:** 82319571
#
# **Name:** Vaibhav Wadikar **Red ID:** 822035741
#
# ## Model Training Notebook

# ## 1. Imports

# In[1]:


import sys, os, re, random
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import itertools
from itertools import chain

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

join_path = os.path.join
ls = os.listdir
exists = os.path.exists
bname = os.path.basename
dname = os.path.dirname
find = re.findall
mapFn = lambda x, y: list(map(x, y))

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[2]:


base_dataset_dir = '../../00_dataset/split_dataset'
ls(base_dataset_dir)


# In[3]:


# dataset paths
X_train_csv, y_train_csv, y_test_csv, X_test_csv = ['X_train.csv', 'y_train.csv', 'y_test.csv', 'X_test.csv']
X_train_path, y_train_path, y_test_path, X_test_path =  mapFn(lambda x: join_path(base_dataset_dir, x),
                                          [X_train_csv, y_train_csv, y_test_csv, X_test_csv])
X_train_path, y_train_path, y_test_path, X_test_path


# In[4]:


["{} exists...".format(path) for path in [X_train_path, y_train_path, y_test_path, X_test_path] if exists(path)]


#

# ## 2. Load Data

# In[5]:


X_train, y_train, y_test, X_test = mapFn(pd.read_csv,
                                                     [X_train_path, y_train_path, y_test_path, X_test_path])


# In[6]:


print("Training Rows : {}, Features: {} \nTesting Rows : {}, Features: {}".format(X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1]))


# In[7]:


X_train, y_train, y_test, X_test = mapFn(lambda df: df.values,
                                         [X_train, y_train, y_test, X_test])


#

# ### 2.1 Feature Scaling

# In[8]:


def scalingFunction(choice="std"):

    std_scaler = preprocessing.StandardScaler()
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    normalizer = preprocessing.Normalizer()

    if choice == "std": return std_scaler
    elif choice == "minmax": return min_max_scaler
    elif choice == "normalize": return normalizer
    else: return


# In[9]:


def scaleData(train, test, scaling_fn_choice=None):

    if choice != None and choice != "None":
        scaling_fn = scalingFunction(choice=choice)
        train, test = mapFn(scaling_fn.fit_transform, [train, test])
    else:
        print("No Feature Scaling")

    return train, test


# In[10]:


# choice = None
# choice = "minmax"
# choice = "normalize"
choice = "std"

X_train, X_test = scaleData(train=X_train, test=X_test, scaling_fn_choice=choice)


#

# ### 2.2 Dimensionality Reduction - PCA

# In[11]:


entireDataX = np.vstack([X_train, X_test])
entireDataX.shape


# In[12]:


pca_model = PCA(n_components=0.95, svd_solver='full')
# pca_model = PCA(n_components=5)
pca_model.fit(entireDataX)


# In[13]:


variance_captured = sum(pca_model.explained_variance_ratio_)
"{:.3f}%".format(variance_captured)


#

# ### 2.3 Feature Extraction

# In[14]:


X_train, X_test = mapFn(pca_model.transform, [X_train, X_test])


# In[15]:


print("After Dimensionality Reduction: \n\nTraining Rows : {}, Features: {} \nTesting Rows : {}, Features: {}".format(X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1]))


# In[16]:


if X_train.shape[1] <= 5:
    plt.figure()
    sns.pairplot(pd.DataFrame(X_train))
    plt.savefig("x_train_scalefn_{}".format(choice))
    plt.show()


#

# ## 3. Training

# **Model of Interests:**
#
# 1. Logistic Regression
#
# 2. Random Forests
#
# 3. Feed Forward Network
#
# **Overall Training Process**
#
#  - Apply Grid Search on the hyperparameterss of each model
#
#  - Calculate accuracy, precision, recall, and Confusion Matrix for each of them
#
#  - Select model with top accuracy as the candidate model for the type of model
#
#  - Compare Results across the above different three model types

# ### 3.1 Logistic Regression

# #### 3.1.1 Hyperparameters for Logistic Regression

# In[17]:


# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = np.logspace(0, 5, 10)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)


# #### 3.1.2 Grid Search on Hyper Parameters of Logistic Regression

# In[18]:


logistic = linear_model.LogisticRegression(random_state=40)

clf = GridSearchCV(logistic, hyperparameters, cv=3, verbose=0)

log_reg_model = clf.fit(X_train, y_train)


# In[19]:


# View best hyperparameters
print('Best Penalty:', log_reg_model.best_estimator_.get_params()['penalty'])
print('Best C:', log_reg_model.best_estimator_.get_params()['C'])


# In[20]:


y_pred = log_reg_model.predict(X_test)
best_acc = log_reg_model.score(X_test, y_test)
best_acc


# #### 3.1.3 Top Model Performance

# In[21]:


logreg_top_accuracy = best_acc
logreg_top_precision = precision_score(y_test, y_pred, average=None)
logreg_top_recall = recall_score(y_test, y_pred, average=None)
logreg_top_conf_mat = confusion_matrix(y_pred, y_test)

print("For Logistic Regression:")
print("\nTop Accuracy: {}\n".format( logreg_top_accuracy ))
print("\nTop Precision: {}\n".format( logreg_top_precision ))
print("\nTop Recall: {}\n".format( logreg_top_recall ))
print("\nTop Confusion Matrix: \n{}\n".format( logreg_top_conf_mat ))
