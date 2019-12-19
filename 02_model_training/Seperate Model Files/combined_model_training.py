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


# ---

# ### 3.2 Random Forests

# #### 3.2.1 Hyperparameters for Random Forests

# In[22]:


random_forest_classifier = RandomForestClassifier()

# Define the grid
rand_forest_grid = {
    'bootstrap': [True],
    'max_depth': [75, 80, 85, 90],
    'max_features': ['auto'],
    'min_samples_leaf': [5, 10, 15],
    'min_samples_split': [5, 10, 15],
    'n_estimators': [150, 175, 200, 225]
}


# Search parameter space
rand_forest_gridsearch = GridSearchCV(estimator = random_forest_classifier,
                           param_grid = rand_forest_grid,
                           cv = 2,
                           verbose = 1,
                           n_jobs = -1)


# #### 3.2.2 Grid Search on Hyper Parameters of Random Forests

# In[23]:


rand_forest_gridsearch.fit(X_train, y_train)


# In[24]:


rand_forest_gridsearch.best_params_


# #### 3.2.3 Top Model Performance

# In[25]:


y_pred = rand_forest_gridsearch.predict(X_test)
y_pred.shape


# In[26]:


rf_top_accuracy = cohen_kappa_score(y_pred, y_test, weights='quadratic')
rf_top_precision = precision_score(y_test, y_pred, average=None)
rf_top_recall = recall_score(y_test, y_pred, average=None)
rf_top_conf_mat = confusion_matrix(y_pred, y_test)

print("For Random Forest Model:")
print("\nTop Accuracy: {}\n".format( rf_top_accuracy ))
print("\nTop Precision: {}\n".format( rf_top_precision ))
print("\nTop Recall: {}\n".format( rf_top_recall ))
print("\nTop Confusion Matrix: \n{}\n".format( rf_top_conf_mat ))


# ### 3.3 Feed Forward Network

# In[27]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn import metrics

def buildModel(num_classes=5,
                num_layers=2, num_neurons=512, learning_rate=0.001,
                activation_fn="relu", output_fn="softmax", num_features=5):

    model = Sequential()

    for layer_no in range(num_layers):

        if layer_no == 0:
            model.add(Dense(num_neurons, activation=activation_fn, input_shape=(num_features,)))

        else:
            model.add(Dense(num_neurons, activation=activation_fn))

        model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation=output_fn))

    return model

def createFNNModel(x_train, y_train, num_classes=5, num_features=5,
                    num_layers=2, num_neurons=128, learning_rate=0.001, batch_size=128, epochs=10,
                    activation_fn="relu", output_fn="softmax", verbose=1, ):

    fnn_model = buildModel(num_classes=num_classes, num_layers=num_layers, num_neurons=num_neurons, learning_rate=learning_rate,
                                    activation_fn=activation_fn, output_fn=output_fn, num_features=num_features)

    fnn_model.summary()

    fnn_model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    fnn_model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose)

    return fnn_model


# In[28]:


def func_confusion_matrix(y_test, y_pred, num_classes=5):
    """ this function is used to calculate the confusion matrix and a set of metrics.
    INPUT:
        y_test, ground-truth lables;
        y_pred, predicted labels;
    OUTPUT:
        CM, confuction matrix
        acc, accuracy
        arrR[], per-class recall rate,
        arrP[], per-class prediction rate.
    """

    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    unique_values = set(y_pred)
    sorted(unique_values)
#     num_classes = len(unique_values)
    unique_values = np.array(list(unique_values)) # change to array so can use indexes
    possible_string_dict = {}
    # make sure all values are 0 based, so can use built-in "zip" function
    if(issubclass(type(y_test[0]), np.integer)): # if values are integers
        y_test_min = y_test.min()
        if(y_test_min != 0):# if does not contain 0, reduce both test and pred by min value to get 0 based for both
            y_test = y_test - y_test_min;
            y_pred = y_pred - y_test_min;
    else:
        # assume values are strings, change to integers
        # TODO, change to convert list from string to int
        y_test_int = np.empty(len(y_test), dtype=int)
        y_pred_int = np.empty(len(y_pred), dtype=int)
        for index in range(0, num_classes):
            current_value = unique_values[index]
            possible_string_dict[index] = current_value
            y_test_int[y_test == current_value] = index
            y_pred_int[y_pred == current_value] = index
        y_test = y_test_int
        y_pred = y_pred_int

    ## your code for creating confusion matrix;
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int)
    for a, p in zip(y_test, y_pred):
        conf_matrix[a][p] += 1


    ## your code for calcuating acc;
    accuracy = conf_matrix.diagonal().sum() / conf_matrix.sum()

    ## your code for calcualting arrR and arrP;
    recall_array = np.empty(num_classes, dtype=float)
    precision_array = np.empty(num_classes, dtype=float)
    for index in range(0, num_classes):
        value = conf_matrix[index,index]
        recall_sum = conf_matrix[index,:].sum()
        precision_sum = conf_matrix[:, index].sum()
        recall_array[index] = value / recall_sum
        precision_array[index] = value / precision_sum

    return conf_matrix, accuracy, recall_array, precision_array


# In[29]:


num_classes = 5
y_train_catg = keras.utils.to_categorical(y_train, num_classes)
y_test_catg = keras.utils.to_categorical(y_test, num_classes)

y_train_catg.shape, y_test_catg.shape


# #### 3.3.1 Hyperparameters for FNN

# In[30]:


num_features = X_train.shape[1]
epochs = 10


# In[31]:


num_layers_choices = [2, 3]
num_neurons_choices = [128, 256]
learning_rate_choices = [0.01, 0.001]
activation_fn_choices = ["relu", "sigmoid"]
output_function = "softmax"


# #### 3.3.2 Grid Search on Hyper Parameters of FNN

# In[32]:


model_names, models = [], []
model_conf_matrices, model_accuracies, model_recall_array, model_precision_array = [], [], [], []

# Perform Grid Search To Find Best Model
for no_layers in num_layers_choices:

    for no_neurons in num_neurons_choices:

        for lr in learning_rate_choices:

            for act_fn in activation_fn_choices:

                model_name = "_".join([str(no_layers), str(no_neurons), str(lr), act_fn])

                print("\n\nTraining Model {}...".format(model_name))

                trained_model = createFNNModel(X_train, y_train_catg, num_features=num_features,
                                num_layers=no_layers, num_neurons=no_neurons, learning_rate=lr,
                                activation_fn=act_fn, output_fn=output_function, epochs=epochs, verbose=0)

                val_loss, val_acc = trained_model.evaluate(X_test, y_test_catg, verbose=1)
                print("\nVal loss: {} \tVal accuracy: {}".format(val_loss, val_acc))
                models.append(trained_model)
                model_accuracies.append(val_acc)

                y_pred_test = trained_model.predict(X_test)
                print(y_pred_test.shape)
                conf_matrix, _, recall_array, precision_array = func_confusion_matrix(y_test_catg, y_pred_test, num_classes=5)

                model_names.append(model_name)
                model_conf_matrices.append(conf_matrix)
                model_recall_array.append(recall_array)
                model_precision_array.append(precision_array)

                print("\nConfusion Matrix: \n{}, \nPer-Class Precision: \n{} \nPer-Class Recall: \n{}"
                .format( conf_matrix, recall_array, precision_array ))


# #### 3.3.3 Top Model Performance

# In[33]:


top_acc_model_no = np.argmax(model_accuracies)

ffn_top_accuracy = model_accuracies[top_acc_model_no]
ffn_top_precision = model_recall_array[top_acc_model_no]
ffn_top_recall = model_precision_array[top_acc_model_no]
ffn_top_conf_mat = model_conf_matrices[top_acc_model_no]

print("For Feed Forward Network:")
print("\n{} has the top accuracy.".format(model_names[top_acc_model_no]))
print("\nTop Accuracy: {}\n".format( ffn_top_accuracy ))
print("\nTop Precision: {}\n".format( ffn_top_precision ))
print("\nTop Recall: {}\n".format( ffn_top_recall ))
print("\nTop Confusion Matrix: \n{}\n".format( ffn_top_conf_mat ))


# ---

# ## 4. Comparision of Model Evaluation Metrics

# In[34]:


logreg_top_accuracy, rf_top_accuracy, ffn_top_accuracy


# In[35]:


logreg_top_precision, rf_top_precision, ffn_top_precision


# In[36]:


logreg_top_recall, rf_top_recall, ffn_top_recall


#

# In[37]:


ml_model_names = ["Logistic Regression", "Random Forests", "Feed Forward Networks"]
ml_model_names


# In[38]:


df_recalls = pd.DataFrame( [logreg_top_recall,  rf_top_recall, ffn_top_recall ] ).T
df_recalls.columns = ml_model_names
df_recalls


# In[39]:


df_precisions = pd.DataFrame( [logreg_top_precision,  rf_top_precision, ffn_top_precision ] ).T
df_precisions.columns = ml_model_names
df_precisions


#

# In[40]:


def showPlot(dataframe, title="Title", xlabel="X-axis", ylabel="Y-axis"):

    plt_path = "eval_plots"

    if not os.path.exists(plt_path): os.makedirs(plt_path)

    plt.figure()
    for col in dataframe.columns:
        plt.scatter(range(5), dataframe[col])
    plt.legend(dataframe.columns)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig("{}/{}_scatter".format(plt_path, title.lower()))
    plt.show()

    plt.figure()
    for col in dataframe.columns:
        plt.bar(range(5), dataframe[col])
    plt.legend(dataframe.columns)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig("{}/{}_bar".format(plt_path, title.lower()))
    plt.show()

    plt.figure()
    dataframe.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(5))

    plt.savefig("{}/{}_bar_pandas".format(plt_path, title.lower()))
    plt.show()


#

# ### Precision Comparison

# In[41]:


showPlot(df_precisions, title="Precisions",
         xlabel="Adoption Rating (Range 0-5)",
         ylabel="Precision Score")


# ### Recall Comparison

# In[42]:


showPlot(df_recalls, title="Recalls",
         xlabel="Adoption Rating (Range 0-5)",
         ylabel="Recall Score")


# ### Confusion Matrix Comparison

# In[43]:


def plotConfMat(conf_matrices, model_names):

    plt_path = "conf_mats"
    if not os.path.exists(plt_path): os.makedirs(plt_path)


    for (i, mname) in enumerate(model_names):

        plt.figure()
        plot_confusion_matrix(conf_matrices[i], classes=[x for x in range(5)])

        plt.savefig("{}/{}_conf_matrix".format(plt_path, mname.lower()))
        plt.show()


# In[44]:


ml_model_names


# In[45]:


plotConfMat(conf_matrices=[logreg_top_conf_mat, rf_top_conf_mat, ffn_top_conf_mat], model_names=ml_model_names)


# ---
