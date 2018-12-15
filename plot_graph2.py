import re

import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.rcParams['figure.figsize'] = (20,20)

def plot_learning_curves(x_axis_label, y_axis_label, x_axis, y_axis_1, y_axis_2, y_axis_3, image_name):
    print (y_axis_1)
    print (x_axis)
    plt.bar(x_axis, y_axis_1, color='blue', align='center', label='SVM')
    plt.bar(x_axis, y_axis_2, color='green', align='center',label='DecisionTree')
    plt.bar(x_axis, y_axis_3, color='pink', align='center', label='Deep Neural Network')
    plt.legend(['SVM', 'DecisionTree', 'Deep Neural Network'], loc='best')
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    # plt.xticks(x_axis)
    plt.title(y_axis_label + ' v/s ' + x_axis_label)
    plt.savefig(image_name,dpi=300)
    plt.show()

def main():
    lists = []
    with open("allResults.txt","rb") as fp:
        results = fp.readlines()
        for i in range(10):
            axis = []
            lists.append(axis)
        
        for result in results[1:]:
            result = str(result)
    #         print (result.split(','))
            
            result = re.sub('[^0-9.,]*', '', result)
            result = result.split(',')
    #         print(result)
            for i, score in enumerate(result):
                if i == 0:
                    lists[i].append(int(score))
                else:
                    
                    score = (float(score)*100)
                    lists[i].append(score)

    print (len(lists[0]))
    # print (lists[1])
    # print (lists[4])
    # print (lists[7])
    plot_learning_curves('Dataset Size', 'F1 Score', lists[0], lists[1], lists[4], lists[7], 'F1Scores.png')
    # plot_learning_curves('Dataset Size', 'Accuracy', lists[0], lists[2], lists[5], lists[8], 'Accuracy.png')
    # plot_learning_curves('Dataset Size', 'Time', lists[0], lists[3], lists[6], lists[9], 'Time.png')



    # plot_learning_curves('Dataset Size', 'F1 Score', lists[0], lists[3], lists[4], lists[7], '')

if __name__ == '__main__':
    main()
