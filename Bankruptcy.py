#-------------------------------#
#---- Bankruptcy AI Project ----#
#-------------------------------#

# Importing the Modules :
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


#-------------------------------------------------# DATA PREPROSSING #-------------------------------------------------#

# Import the Dataset File in csv Format :
dataset = pd.read_csv("D:/College/AI/PythonML/Bankruptcy.csv", na_values='?')


# Import Sikit Learn to Take Care of Missing Data :
# Calling the SimpleImputer to Fit and Replace Missing Values :
imputer = SimpleImputer(strategy='median', fill_value='median')

# Fitting Imputer to the Data by the Dependent Matrix
# in Slicing Upper Bound is Excluded :
imputer.fit(dataset.iloc[:, 0:65])
dataset.iloc[:, 0:65] = imputer.transform(dataset.iloc[:, 0:65])

# Divide the Dataset into Dependent Variables x and Independent Varaible y :
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 65].values

print(Counter(y))

# define feature selection
fvalue_Best = SelectKBest(f_classif, k=20)
 
# apply feature selection
X_selected = fvalue_Best.fit_transform(X, y)

print(X_selected.shape)

# Imbalance Data :
oversample = SMOTE()
X_resampled, y_resampled = oversample.fit_resample(X_selected, y)
dataset = resample(dataset, replace=True, n_samples=9773)

print(Counter(y_resampled))

# Splitting Dataset into Training and Test set :
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.4)



# Data Normalization :
norm = MinMaxScaler().fit(X_train)

X_train = norm.transform(X_train)

X_test = norm.transform(X_test)


#-------------------------------------------------# END OF DATA PREPROSSING #-------------------------------------------------#

#-------------------------------------------------# CLASSIFICATIONS #-------------------------------------------------#

# Fitting Logistic Regression to the Training Set :
LR_classifier = LogisticRegression(max_iter=1000)
LR_classifier.fit(X_train, y_train)
    
# Predicting the Test set Resulat in LR :
y_predict_LR = LR_classifier.predict(X_test)
    
# Making the Confusion Matrix :
LR_cm = confusion_matrix(y_test, y_predict_LR)
    
# Accuracy in LR Model :
accuracy_Score_LR = accuracy_score(y_test, y_predict_LR)
    
#Precision in LR Model
Precision_Score_LR = precision_score(y_test, y_predict_LR)
    
ax = sns.heatmap(LR_cm, annot=True, cmap='Blues')
ax.set_title('Logistic Regression Confusion Matrix\n\n');
ax.set_xlabel('\nModel\'s \'Predicted Values')
ax.set_ylabel('Actual Values');
    
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
    
## Display the visualization of the Confusion Matrix.
plt.show()
#---------------------------------------------------------#

# Fitting SVM to the Training Set :
SVM_classifier = SVC(kernel="linear", random_state=0)
SVM_classifier.fit(X_train, y_train)
    
# Predicting the Test set Resulat in SVM :
y_predict_SVM = SVM_classifier.predict(X_test)
    
# Making Confusion Mشtrix in SVM :
SVM_cm = confusion_matrix(y_test, y_predict_SVM)
    
# Accuracy Model in SVM :
accuracy_Score_SVM = accuracy_score(y_test, y_predict_SVM)
    
#Precision in SVM Model
Precision_Score_SVM = precision_score(y_test, y_predict_SVM)
    
ax = sns.heatmap(SVM_cm, annot=True, cmap='Blues')
ax.set_title('SVM Confusion Matrix\n\n');
ax.set_xlabel('\nModel\'s \'Predicted Values')
ax.set_ylabel('Actual Values');
    
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
    
## Display the visualization of the Confusion Matrix.
plt.show()
#---------------------------------------------------------#

# Fitting Decision Tree to the Training Set :
Dtree_Classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
Dtree_Classifier.fit(X_train, y_train)
    
# Predicting the Test set Resulat in Decision Tree :
y_predict_Dtree = Dtree_Classifier.predict(X_test)
    
# Making Confusion Matrix in DT :
Dtree_cm = confusion_matrix(y_test, y_predict_Dtree)
    
# Accuracy Model in DT :
accuracy_Score_Dtree = accuracy_score(y_test, y_predict_Dtree)
    
#Precision in DT Model
Precision_Score_Dtree = precision_score(y_test, y_predict_Dtree)
    
ax = sns.heatmap(Dtree_cm, annot=True, cmap='Blues')
ax.set_title('Decision Tree Confusion Matrix\n\n')
ax.set_xlabel('\nModel\'s \'Predicted Values')
ax.set_ylabel('Actual Values')
    
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
    
## Display the visualization of the Confusion Matrix.
plt.show()
#---------------------------------------------------------#

# Training the Random Forest Classification model on the Training set :
RF_classifier = RandomForestClassifier(
        n_estimators=10, criterion='entropy', random_state=0)
RF_classifier.fit(X_train, y_train)
    
# Predicting the Test set results :
y_predict_RF = RF_classifier.predict(X_test)
    
# Making the Confusion Matrix :
RF_cm = confusion_matrix(y_test, y_predict_RF)
    
# Accuracy Model in Random Forest :
accuracy_Score_RF = accuracy_score(y_test, y_predict_RF)
    
#Precision in RF Model
Precision_Score_RF = precision_score(y_test, y_predict_RF)
    
ax = sns.heatmap(RF_cm, annot=True, cmap='Blues')
ax.set_title('Random Forrest Confusion Matrix\n\n')
ax.set_xlabel('\nModel\'s \'Predicted Values')
ax.set_ylabel('Actual Values')
    
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
    
## Display the visualization of the Confusion Matrix.
plt.show()
#---------------------------------------------------------#

# Training the K-NN model on the Training set
KNN_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
KNN_classifier.fit(X_train, y_train)
    
# Predicting the Test set results
y_predict_KNN = KNN_classifier.predict(X_test)
    
# Making the Confusion Matrix
KNN_cm = confusion_matrix(y_test, y_predict_KNN)
    
# Accuracy Model in Random Forest :
accuracy_Score_KNN = accuracy_score(y_test, y_predict_KNN)
    
#Precision in KNN Model
Precision_Score_KNN = precision_score(y_test, y_predict_KNN)
    
ax = sns.heatmap(KNN_cm, annot=True, cmap='Blues')
ax.set_title('KNeighbors Confusion Matrix\n\n')
ax.set_xlabel('\nModel\'s \'Predicted Values')
ax.set_ylabel('Actual Values')
    
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
    
## Display the visualization of the Confusion Matrix.
plt.show()
#-------------------------------------------------# END OF CLASSIFICATIONS #-------------------------------------------------#


#-------------------------------------------------# GUI #-------------------------------------------------#

# Create The Main GUI Window:
Bankurptcy_GUI = tk.Tk()

# Change GUI Text :
Bankurptcy_GUI.title(" Bankurptcy Project ")

# Set Dimensions :
Bankurptcy_GUI.geometry("650x730+450+50")

# Write Models Labels :
the_label = tk.Label(Bankurptcy_GUI, text="Select The Model", font=("Arial", 30))
the_label.pack()  # Place The Text Into The Main Window

#-------------------------------------------------#


# Models Functions :


def Calc_LRM_odel():
    line_one = f" Accuracy_Score For Linear Regression : {accuracy_Score_LR}"
    line_two = f" Precision For Linear Regression : {Precision_Score_LR}"
    ListOne = [line_one, line_two]
    messagebox.showinfo("Linear Regression Model", "\n".join(ListOne))

#-------------------------------------------------#


def Calc_SVM_Model():
    line_one = f" The Accuracy_Score For Support Vector Machine  : {accuracy_Score_SVM}"
    line_two = f" Precision For Support Vector Machine  : {Precision_Score_SVM}"
    listTwo = [line_one, line_two]
    messagebox.showinfo("Support Vector Machine Model", "\n".join(listTwo))

#-------------------------------------------------#


def Calc_Dtree_Model():
    line_one = f" The Accuracy_Score For Decision Tree : {accuracy_Score_Dtree}"
    line_two = f" Precision For Decision Tree : {Precision_Score_Dtree}"
    listThree = [line_one, line_two]
    messagebox.showinfo("Decision Tree Model", "\n".join(listThree))

#-------------------------------------------------#


def Calc_RF_Model():
    line_one = f" The Accuracy_Score For Random Forest : {accuracy_Score_RF}"
    line_two = f" Precision For Random Forest: {Precision_Score_RF}"
    listFive = [line_one, line_two]
    messagebox.showinfo("Random Forest Model", "\n".join(listFive))

#-------------------------------------------------#


def Calc_KNN_Model():
    line_one = f" The Accuracy_Score For KNN : {accuracy_Score_KNN}"
    line_two = f" Precision For Linear Regression : {Precision_Score_KNN}"
    listSix = [line_one, line_two]
    messagebox.showinfo("KNN Model", "\n".join(listSix))

#-------------------------------------------------#
# Create The Models Button :

# Create main frame
Main_frame = ttk.Frame(Bankurptcy_GUI)
Main_frame.config(width=400, height=400)
Main_frame.config(relief=tk.RIDGE)
Main_frame.config(padding=(100,50))
Main_frame.pack()

#-------------------------------------------------#
LR_Boutton = tk.Button(Main_frame, text="Calculate Accuracy And CM in LR", relief=tk.FLAT,
                       highlightthickness=20, bg="#e91e63", fg="white", borderwidth=0, command=Calc_LRM_odel)
LR_Boutton.pack(pady=20)

#-------------------------------------------------#

SVM_Button = tk.Button(Main_frame, text="Calculate Accuracy And CM in SVM", relief=tk.FLAT,
                       highlightthickness=20, bg="#e91e63", fg="white", borderwidth=0, command=Calc_SVM_Model)
SVM_Button.pack(pady=20)

#-------------------------------------------------#

DT_Button = tk.Button(Main_frame, text="Calculate Accuracy And CM in DT", relief=tk.FLAT,
                      highlightthickness=20, bg="#e91e63", fg="white", borderwidth=0, command=Calc_Dtree_Model)
DT_Button.pack(pady=20)

#-------------------------------------------------#

RF_Button = tk.Button(Main_frame, text="Calculate Accuracy And CM in RF", relief=tk.FLAT,
                     highlightthickness=20, bg="#e91e63", fg="white", borderwidth=0, command=Calc_RF_Model)
RF_Button.pack(pady=20)

#-------------------------------------------------#

KNN_Button = tk.Button(Main_frame, text="Calculate Accuracy And CM in KNN", relief=tk.FLAT,
                      highlightthickness=20, bg="#e91e63", fg="white", borderwidth=0, command=Calc_KNN_Model)
KNN_Button.pack(pady=20)

#-------------------------------------------------#
# Run GUI Infinitely :
Bankurptcy_GUI.mainloop()

#-------------------------------------------------# END OF GUI #-------------------------------------------------#
