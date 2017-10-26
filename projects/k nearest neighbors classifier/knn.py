#Importing libraries
import numpy as np
import pandas as pd
import operator
import math

#Defining euclidean_dist function
def euclidean_dist(input_data1,input_data2):
    """
    This function calculates the euclidean distance between input and a row
    Arguments:
        input_data1 (row array): A row from the testing data frame
        input_data2 (row array): A row from the training data frame
        Returns:
        ed (float): The euclidean distance between input data and a row
    """
    #Initializing temp variable
    temp=0
    #Calculating euclidean distance
    for i in range(0,len(input_data1)):
        if isinstance(input_data1[i],(int,float)) and isinstance(input_data2[i],(int,float)):
            temp=temp+(input_data1[i]-input_data2[i])**2
        else:
            raise ValueError('Input data not integer or float. Check column indices of the feature vector')
    ed=math.sqrt(temp)
    #Returning euclidean diatance
    return ed

#Defining sorted_euclidean_dist function
def sorted_euclidean_dist(input_data1,data,columns,class_column):
    """
    This function sorts the euclidean distances between the input and all the rows of a data frame
    Arguments:
        input_data (row array): A row from the testing data frame
        data (data frame): Training data frame
        columns (list): The column indices which define the feature vectors
        class_column(integer): The column index which defines the class
    Returns:
        sed (list): The sorted euclidean distances between input and all rows of training data frame
    """
    #Initializing sed as a list
    sed=[]
    #Calculating euclidean distances between input and all rows of training data frame
    for index,row in data.iterrows():
        input_data2=[]
        for i in columns:
            input_data2.append(row[i])
        temp=euclidean_dist(input_data1,input_data2)
        sed.append((temp,data.loc[index][class_column]))
    #Sorting the euclidean distances
    sed.sort()
    #Returning the sorted euclidean distances
    return sed

#Defining class_prediction function
def class_prediction(sorted_euclidean_dist,k):
    """
    This function predicts the class of the input data
    Arguments:
        sorted_euclidean_dist (list): The sorted euclidean distances between input and all rows of training data frame
        k (int): Parameter k in the KNN classifier
    Returns:
        prediction (str): The predicted class of the input data
    """
    #Finding out unique classes of the data
    a=[]
    for i in range(0,len(sorted_euclidean_dist)):
        a.append(sorted_euclidean_dist[i][1])
    class_names=list(set(a))
    #Setting count of classes to zero
    count={}
    for i in class_names:
        count[i]=0
    if k>len(sorted_euclidean_dist):
        raise ValueError('k is greater than number of examples in training data. Use smaller values of k')
    #Calculating the counts of the classes of the k nearest neighbors
    for i in range(0,k):
        for x in class_names:
            if sorted_euclidean_dist[i][1]==x:
                count[x]=count[x]+1
    #Comparing the counts to predict the class of the input data
    prediction=list(count.keys())[0]
    for key in count:
        if count[key] > count[prediction]:
            prediction = key
    #Returning the predicted class of the input data
    return prediction

#Defining knn_classifier function
def knn_classifier(data,input_data,k,columns,class_column):
    """
    This function returns the class of the input data
    Arguments:
        data (data frame): Training data frame
        input_data (row array): A row from the testing data frame
        k (int): Parameter k in the KNN classifier
        columns (list): The column indices which define the feature vectors
    Returns:
        prediction (str): The predicted class of the input data
    """
    #Calculating the sorted euclidean distances
    sorted_ed=sorted_euclidean_dist(input_data,data,columns,class_column)
    #Calculating the prediction of the input data
    prediction=class_prediction(sorted_ed,k)
    #Returning the predicted class of the input data
    return prediction

#Defining knn_accuracy function
def knn_accuracy(training_df,testing_df,k_list,class_column,columns):
    """
    This function tests the accuracy of the KNN for the test data frame for different values of k
    Arguments:
        training_df (data frame): Training data frame
        testing_df (data frame): Testing data frame
        k_list (list): List of k values to be tested for accuracy
        columns (list): The column indices which define the feature vectors
        class_column(integer): The column index which defines the class
    Returns:
        accuracy (dictionary): The accuracy of the KNN for the test data frame for different values of k
    """
    #Initializing accuracy as a dictionary
    accuracy={}
    #Calculating the accuracy
    for i in k_list:
        #Calculating the accuracy for a particular value of k
        count=0
        for index,row in testing_df.iterrows():
            input_data=[]
            for j in range(0,len(columns)):
                input_data.append(row[columns[j]])
            prediction=knn_classifier(training_df,input_data,i,columns,class_column)
            if prediction == row[class_column]:
                count = count+1
            #Adding k as key and accuracy percentage as value
        accuracy[i]=(count*100)/(index+1)
    #Returning the accuracy dictionary
    return accuracy
