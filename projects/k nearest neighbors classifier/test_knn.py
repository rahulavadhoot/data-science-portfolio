#Importing libraries
import knn
import pandas as pd
import math

#Initializing variables
training_df = pd.read_csv('atomsradii.csv')
testing_df = pd.read_csv('testing.csv')
columns = [0,1]
class_column = 3

#Defining test_euclidean_dist function
def test_euclidean_dist():
    """
    This function tests the euclidean_dist function in knn.py
    """
    #Input points for assert 1
    input_data1 = [0,1]
    input_data2 = [0,0]
    #Assert1
    assert knn.euclidean_dist(input_data1,input_data2)==1,'Euclidean distance incorrect'
    #Input points for assert 2
    input_data1 = [1,2]
    input_data2 = [3,4]
    #Assert2
    assert math.isclose(knn.euclidean_dist(input_data1,input_data2),math.sqrt(8)),'Euclidean distance incorrect'
    return

#Defining test_sorted_euclidean_dist function
def test_sorted_euclidean_dist():
    """
    This function tests the sorted_euclidean_dist function in knn.py
    """
    #Input sorted euclidean distances for assert 1
    sed1 = knn.sorted_euclidean_dist([0.78,0.5],training_df,columns,class_column)
    #Testing if the list is sorted
    #Assert1
    for i in range(0,len(sed1)):
        if i < (len(sed1)-1):
            assert sed1[i][0] <= sed1[i][0], 'Wrong sorting'
        else:
            pass
    #Input sorted euclidean distances for asserts 2 & 3
    sed2=knn.sorted_euclidean_dist([0.45,0.68],training_df,columns,class_column)
    #Testing sorted_euclidean_dist over an example in training dataframe itself
    #Assert2
    assert sed2[0][0] == 0, 'Wrong sorting'
    #Assert3
    assert sed2[0][1] == 'TM', 'Wrong class'
    return

#Defining test_class_prediction function
def test_class_prediction():
    """
    This function tests the class_prediction function in knn.py
    """
    #Input sorted euclidean distances for asserts 1,2, & 3
    sed1 = knn.sorted_euclidean_dist([0.78,0.5],training_df,columns,class_column)
    sed2 = knn.sorted_euclidean_dist([0.45,0.68],training_df,columns,class_column)
    sed3 = knn.sorted_euclidean_dist([0.28,1.01],training_df,columns,class_column)
    #Testing class_prediction over 3 different examples in training dataframe
    #Assert1
    assert knn.class_prediction(sed1,1) == 'PT', 'Wrong prediction'
    #Assert2
    assert knn.class_prediction(sed2,1) == 'TM', 'Wrong prediction'
    #Assert3
    assert knn.class_prediction(sed3,1) == 'Alk', 'Wrong prediction'
    return

#Defining test_knn_classifier function
def test_knn_classifier():
    """
    This function tests the knn_classifier function in knn.py
    """
    #Testing knn_classifier over 3 different examples in training dataframe
    #Assert1
    assert knn.knn_classifier(training_df,[0.9,0.67],1,columns,class_column) == 'PT', 'Wrong classification'
    #Assert2
    assert knn.knn_classifier(training_df,[0.32,0.62],1,columns,class_column) == 'TM', 'Wrong classification'
    #Assert3
    assert knn.knn_classifier(training_df,[0.74,1.45],1,columns,class_column) == 'Alk', 'Wrong classification'
    return

#Defining test_knn_accuracy function
def test_knn_accuracy():
    """
    This function tests the knn_accuracy function in knn.py
    """
    #Assert1
    #Testing knn_accuracy over the training dataframe itself
    assert knn.knn_accuracy(training_df,training_df,[1],3,[0,1]) == {1:100}, 'Wrong accuracy'
    return
