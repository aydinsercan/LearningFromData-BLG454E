import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import math
from scipy.stats import pearsonr

#This will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



#reading the dataset from csv files
train_t0 = pd.read_csv("train_t0.csv")
train_t1 = pd.read_csv("train_t1.csv")
test_t0 = pd.read_csv("test_t0.csv")

#this function is for selecting k best feature from given features
#it gets features, labels and k as input
#returns feature selection model and newly transformed data
def KBest(X,y,k):
    FSmodel = SelectKBest(mutual_info_regression, k=k).fit(X, y)
    new_X = FSmodel.transform(X)
    return FSmodel,new_X

#this function is for extracting k features from given data
#PCA, unsupervised dimensionality reduction technique is used
#It gets all features and desired number of features, returns feature extraction model and newly transformed data
def FS_PCA(X,k):
    pca = PCA(n_components=k)
    pca.fit(X)
    return pca, pca.transform(X)

#this function is for using AdaBoost Regression learning model
#When the features and labels are given It returns the learned model
def Adaboost(X,y):
    return AdaBoostRegressor(random_state=0, n_estimators=100).fit(X,y)

#this function is for using Random Forest learning model
#When the features and labels are given It returns the learned model
def RandomForest(X,y):
    return RandomForestRegressor(max_depth=2, random_state=0).fit(X,y)

#this function is for using Regular Linear Regression learning model
#When the features and labels are given
#It returns the learned model
def regularLinearRegression(X,y):
    return LinearRegression().fit(X,y)

#this function is for using Decision Tree Regression learning model
#When the features and labels are given it returns the learned model
def DecisionTree(X,y):
    return DecisionTreeRegressor().fit(X,y)

#Multioutput Learner Class including features, labels, learned models and dimensionality reduction models
class MultioutputLearner:
    def __init__(self,X,Y):
        epsilon = 1e-100
        self.X = np.array(X).copy() #X is the t0 values of dataset
        self.Y = np.array(Y).copy() #Y is the t1 values of dataset, they are the future values (labels)
        self.listofLR = [] #array for saving learning models, includes 595 learned model, one for every dimension of y(label)
        self.listofFS = [] #array for saving dimensionality reduction techniques, includes 595 learned model, one for every dimension of y(label)
    
    #function for training the dataset
    #It takes the multioutput class and desired number of features as input
    def train(self,n_features):
		for i in range(self.Y.shape[1]): #iterating through 595 brain connectivity t1 values 
			y = np.array(self.Y[:,i]).copy()
			epsilon = 1e-100
			for i in range(y.shape[0]):
				if(y[i] > 1):
					y[i] = 1
				if(y[i] < 0):
		            y[i] = 0
		        if((1-y[i]) < epsilon): #to avoid overflow in model
		            y[i] = 1e+10
		        elif(y[i] < epsilon): #to avoid overflow in model
		            y[i] = -1e+10
		        else:
		            temp = y[i]
		            y[i] = np.log(temp/(1-temp)) #inverse sigmoid function

            temp_y = y.copy() #preprocessing the t1 value
            FSmodel,new_X = KBest(self.X, temp_y, n_features) #selecting 'n_features' features in order to decrease the dimesionality 
            temp_model = svm.SVR().fit(new_X,temp_y) #creating the model
            self.listofLR.append(temp_model) #adding learned model to list
            self.listofFS.append(FSmodel) #adding dimensionality reduction technique to list

    #function for testing the dataset
    #It takes the multioutput class and testset of the dataset as input and returns the predicted t1 values
    def test(self,testset):
        testoutput = [] #array for saving predicted t1 values
        testset = np.array(testset).copy() #changing the testset's type to numpy array
        for i in range(len(self.listofLR)): #traversing through learned models
            new_test = self.listofFS[i].transform(testset) #applying dimensionality reduction to testset
            prediction = self.listofLR[i].predict(new_test).copy() #predicting the t1 values
            testoutput.append(prediction) #adding predictions to array
        # returning the transpose of predicted values, since appending prediction adds predictions one after other
        # but we should add predictions side by sides, hence transpose will provide us the desired format
        y = np.array(testoutput)
    	for i in range(y.shape[0]):
        	for j in range(y.shape[1]):
            	y[i][j] = 1/(1 + np.exp(-y[i][j])) #sigmoid function
        return np.transpose(y) 

#this is a function for testing and training the dataset for kaggle project
#It learns from train data and test on test data
#It gets whole dataset, number for folds and number of desired features as input and returns the predicted values
def KFold_Train_and_Test(train_t0, train_t1, test, k,n_features):
    #adding train datasets of t0 and t1 values side by side
    All_train = np.append( np.delete(np.array(train_t0).copy(), [0], axis=1), np.delete(np.array(train_t1).copy(), [0], axis=1) ,axis=1)
    #creating k folds from train datasets
    K_fold = KFold(n_splits=k, shuffle = True, random_state = np.random)
    # getting testset without ids
    test = np.delete(np.array(test).copy(), [0], axis=1)
    #creating an empty Test_result array in order to save predictions
    Test_result = np.zeros((test.shape[0],int(All_train.shape[1]/2)))
    #k-fold cross validation
    for train_index,test_index in K_fold.split(All_train):
        train_fold = np.array((All_train[train_index])) #getting train fold
        fold_t0 = train_fold[:,0:int(train_fold.shape[1]/2)] #getting features of train fold
        fold_t1 = train_fold[:,int(train_fold.shape[1]/2):train_fold.shape[1]] #getting labels of train fold
        model = MultioutputLearner(fold_t0,fold_t1) #creating the multioutputlearner class from train fold
        model.train(n_features) #training the fold
        prediction = model.test(test) #testing the fold on testset
        Test_result += prediction #adding prediction to test results
    return Test_result/k #taking the averages of prediction to return

#this is a function for testing and training the dataset for 5-fold cross validation
#It learns from train folds and test on the remainings
#It gets train dataset, number for folds and number of desired features as input and returns the mean squared error
def KFold2_Train_and_Test(train_t0, train_t1, k, n_features):
    # adding train datasets of t0 and t1 values side by side
    All_train = np.append( np.delete(np.array(train_t0).copy(), [0], axis=1), np.delete(np.array(train_t1).copy(), [0], axis=1) ,axis=1)
    #creating k folds from train datasets
    K_fold = KFold(n_splits=k, shuffle = True, random_state = np.random)
    #for saving all predictions
    All_prediction = np.delete(np.array(train_t0).copy(), [0], axis=1)
    #k-fold cross validation
    for train_index,test_index in K_fold.split(All_train):
        train_fold = np.array((All_train[train_index]))
        fold_t0 = train_fold[:,0:int(train_fold.shape[1]/2)] #features of train fold
        fold_t1 = train_fold[:,int(train_fold.shape[1]/2):train_fold.shape[1]] #labels of train fold

        test_fold = np.array((All_train[test_index]))
        test_fold_t0 = test_fold[:,0:int(test_fold.shape[1]/2)]#getting features of test fold
        test_fold_t1 = test_fold[:,int(test_fold.shape[1]/2):test_fold.shape[1]] #getting labels of test fold

        model = MultioutputLearner(fold_t0,fold_t1) 
        model.train(n_features) #training the train fold
        prediction = model.test(test_fold_t0) #testing the test fold
        All_prediction[test_index] = prediction #saving prediction
    #the mean squared error between predicted train_t1 values and real train_t1 values
    total_error = mean_squared_error(np.delete(np.array(train_t1).copy(), [0], axis=1),All_prediction)
    #the pearson correlation between predicted train_t1 values and real train_t1 values
    pears_err,p = pearsonr(All_prediction.flatten(),np.delete(np.array(train_t1).copy(), [0], axis=1).flatten())
    print("Mean Squared Error: ", total_error)
    print("Pearson Correlation: ",pears_err," ~ p-value: ",p)
    return total_error #mean squared error

#for controlling whether we are testing for kaggle or 5-fold cross validation
kaggle = False
if kaggle == True: #for kaggle, write predictions to submission csv file
    Test_result = KFold_Train_and_Test(train_t0, train_t1, test_t0 , 5, 130).flatten()
    samplesubmission = pd.read_csv('sampleSubmission.csv')
    output = pd.DataFrame({'ID': samplesubmission.index, 'Predicted': Test_result})
    output.to_csv('kaggle_submission.csv', index=False)
else: #for 5-fold cross validation 
    KFold2_Train_and_Test(train_t0, train_t1 , 5, 130)



