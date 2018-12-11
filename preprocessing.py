import numpy as np
from sklearn.svm import SVC
import pickle

def _get_int_feature(dictionary, key, counter):
    if key in dictionary:
        return dictionary[key], counter
    else:           # key not in dictionary
        dictionary[key] = counter
    return dictionary[key], counter+1

def main():
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
    for info in raw_data[:100000]:
        info = info.replace('\n','').replace('.','').split(',')
        info[1], tcpCount = _get_int_feature(dict_tcp, info[1], tcpCount)
        info[2], httpCount = _get_int_feature(dict_http, info[2], httpCount)
        info[3], sfCount = _get_int_feature(dict_sf, info[3], sfCount)
        # print("info is ", info)
        if info[-1] in dos:
            info[-1] = 'DOS'
            nDOS += 1
            cleanedData.append(info)
        # elif info[-1] in u2r:
        #     info[-1] = 'U2R'
        #     nU2R += 1
        # elif info[-1] in r2l:
        #     info[-1] = 'R2L'
        #     nR2L += 1
        # elif info[-1] in probing:
        #     info[-1] = 'PROBING'
        #     nProb += 1
        elif info[-1] in normal:           # label is normal
            nNormal += 1
            cleanedData.append(info)
        else:                               # unspecified label
            nOthers += 1
    # with open('cleaned_data', 'wb') as fp:
    #     pickle.dump(cleanedData, fp)


    # with open ('cleaned_data', 'rb') as fp:
    #     cleanedData = pickle.load(fp)
    examples_matrix = np.array(cleanedData)
    # print("example is ", examples_matrix[1])
    feature_matrix = examples_matrix[:,:-1]
    label_matrix = examples_matrix[:,-1]
    print(feature_matrix[1])
    print('labels are ', label_matrix[1])
    clf = SVC(gamma='auto')
    clf.fit(feature_matrix, label_matrix)
    print(nDOS,nU2R,nR2L,nNormal,nOthers)

if __name__ == '__main__':
    main()