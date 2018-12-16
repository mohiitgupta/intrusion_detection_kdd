#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.svm import SVC
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import collections
from sklearn import tree
from sklearn.neural_network import MLPClassifier
import time
import os

def _get_int_feature(dictionary, key, counter):
    if key in dictionary:
        return dictionary[key], counter
    else:           # key not in dictionary
        dictionary[key] = counter
    return dictionary[key], counter+1


# In[2]:


def calculate_macro_f1_score(predictions, true_labels):
    true_positives = [0 for i in range(11)]
    false_positives = [0 for i in range(11)]
    false_negatives = [0 for i in range(11)]

    if len(predictions) != len(true_labels):
        print("bug in code, length of predictions should match length of true_labels")
        return None
    for i in range(len(predictions)):
        if predictions[i] == true_labels[i]:
            true_positives[predictions[i]] += 1
        else:
            false_positives[predictions[i]] += 1
            false_negatives[true_labels[i]] += 1

    total_classes = 0
    total_f1 = 0
    for i in range(11):
        if true_positives[i]==0 and false_positives[i]==0:
            continue
        elif true_positives[i]==0 and false_negatives[i]==0:
            continue
        prec = true_positives[i]*1.0/(true_positives[i] + false_positives[i])
        recall = true_positives[i]*1.0/(true_positives[i]+false_negatives[i])
        f1=0
        if prec+recall != 0:
            f1 = 2*prec*recall/(prec+recall)
            total_classes += 1
            total_f1 += f1
    return total_f1*100.0/total_classes

def calculate_micro_f1_score(predictions, true_labels):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    if len(predictions) != len(true_labels):
        print("bug in code, length of predictions should match length of true_labels")
        return None
    for i in range(len(predictions)):
        if predictions[i] == true_labels[i]:
            true_positives += 1
        else:
            false_positives += 1
            false_negatives += 1
    prec = true_positives*1.0/(true_positives + false_positives)
    recall = true_positives*1.0/(true_positives+false_negatives)
    return 2*prec*recall*100.0/(prec+recall)


# In[3]:


dos = ['back','land','neptune','pod','smurf','teardrop']
u2r = ['buffer_overflow','loadmodule','perl','rootkit']
r2l = ['ftp_write','guess_passwd','imap','multihop','phf','spy','warezclient','warezmaster']
probing = ['ipsweep','nmap','portsweep','satan']
normal = ['normal']

ifile = open('../kddcup.data','r')             # loading data
raw_data = ifile.readlines()
ifile.close()

## cleaning ##
cleanedData = []
dict_tcp,tcpCount = {},0
dict_http,httpCount = {},0
dict_sf,sfCount = {},0

nDOS,nU2R,nR2L,nProb,nNormal,nOthers = 0,0,0,0,0,0
for info in raw_data:
    info = info.replace('\n','').replace('.','').split(',')
    info[1], tcpCount = _get_int_feature(dict_tcp, info[1], tcpCount)
    info[2], httpCount = _get_int_feature(dict_http, info[2], httpCount)
    info[3], sfCount = _get_int_feature(dict_sf, info[3], sfCount)
    # print("info is ", info)
    if info[-1] in dos:
        info[-1] = 1 #'DOS' label
        nDOS += 1
#         cleanedData.append(info)
    elif info[-1] in u2r:
        info[-1] = 2 #'U2R'
        nU2R += 1
    elif info[-1] in r2l:
        info[-1] = 3 #'R2L'
        nR2L += 1
    elif info[-1] in probing:
        info[-1] = 4 #'PROBING'
        nProb += 1
    elif info[-1] in normal:           # label is normal
        nNormal += 1
        info[-1] = 0 #'NORMAL' label
        
    else:                               # unspecified label
        nOthers += 1
        continue
    cleanedData.append(info)
# with open('cleaned_data', 'wb') as fp:
#     pickle.dump(cleanedData, fp)


# with open ('cleaned_data', 'rb') as fp:
#     cleanedData = pickle.load(fp)
examples_matrix = np.array(cleanedData)
np.random.shuffle(examples_matrix)


# In[4]:


print(nDOS,nU2R,nR2L,nNormal,nOthers)


# In[5]:


def _run_svm(train_feature_matrix, train_label_matrix, test_feature_matrix):
    clf = SVC(gamma='auto')
    clf.fit(train_feature_matrix, train_label_matrix)
    predicted_labels = clf.predict(test_feature_matrix)
    return predicted_labels


# In[6]:


def _run_dtree(train_feature_matrix, train_label_matrix, test_feature_matrix):
    dt_clf = tree.DecisionTreeClassifier()
    dt_clf = dt_clf.fit(train_feature_matrix, train_label_matrix)
    dt_predictions = dt_clf.predict(test_feature_matrix)
    return dt_predictions


# In[7]:


def _run_nn(train_feature_matrix, train_label_matrix, test_feature_matrix):
    nn_clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(50, 30), random_state=1)
    nn_train_feature_matrix = train_feature_matrix.astype(np.float64)
    nn_test_feature_matrix = test_feature_matrix.astype(np.float64)
    nn_clf.fit(nn_train_feature_matrix, train_label_matrix)
    nn_predictions = nn_clf.predict(nn_test_feature_matrix)
    return nn_predictions


# In[ ]:


# print("example is ", examples_matrix[1])
result_file = open('results.txt','w')
result_file.write("dataset size, svm macroF1Score, svm accuracy, svm time, dtree macroF1Score, dtree accuracy, dtree time, nn macroF1Score, nn accuracy, nn time\n")
for data_size in range(250000,len(cleanedData)+1,5000):
    train_size = int(data_size * 0.7)
    test_size = data_size - train_size
#     train_size = 70000
#     test_size = 30000
    train_feature_matrix = examples_matrix[:train_size,:-1]
    train_label_matrix = examples_matrix[:train_size,-1]
    test_feature_matrix = examples_matrix[train_size+1:train_size+test_size,:-1]
    test_label_matrix = examples_matrix[train_size+1:train_size+test_size,-1]

    #print(collections.Counter(train_label_matrix))
    #print(collections.Counter(test_label_matrix))
    print(data_size)
    #run svm
    
    print('SVM')
    start_time = time.time()
    predicted_labels = _run_svm(train_feature_matrix, train_label_matrix, test_feature_matrix)
    end_time = time.time() - start_time
    macro_f1_score = f1_score(test_label_matrix, predicted_labels, average='macro') 
    accuracy = accuracy_score(test_label_matrix, predicted_labels)
    result_file.write(str(data_size) + ", ")
    result_file.write(str(macro_f1_score) + ", " + str(accuracy) + ", " + str(end_time) + ", ")
    result_file.flush()
    os.fsync(result_file.fileno())
    
    #run decision tree
    print('DT')
    start_time = time.time()
    predicted_labels = _run_dtree(train_feature_matrix, train_label_matrix, test_feature_matrix)
    end_time = time.time() - start_time

    macro_f1_score = f1_score(test_label_matrix, predicted_labels, average='macro') 
    accuracy = accuracy_score(test_label_matrix, predicted_labels)
    result_file.write(str(data_size) + ", ")
    result_file.write(str(macro_f1_score) + ", " + str(accuracy) + ", " + str(end_time) + ", ")
    result_file.flush()
    os.fsync(result_file.fileno())

    #run neural network
    print('ANN')
    start_time = time.time()
    predicted_labels = _run_nn(train_feature_matrix, train_label_matrix, test_feature_matrix)
    end_time = time.time() - start_time

    macro_f1_score = f1_score(test_label_matrix, predicted_labels, average='macro') 
    accuracy = accuracy_score(test_label_matrix, predicted_labels)
    result_file.write(str(macro_f1_score) + ", " + str(accuracy) + ", " + str(end_time) + "\n")
    result_file.flush()
    os.fsync(result_file.fileno())
result_file.close()


# In[ ]:




