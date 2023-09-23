#importing libraries
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import warnings

warnings.filterwarnings(action = 'ignore') 
import sys

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")


import sys

# total arguments
n = len(sys.argv)


print("\nName of Python script:", sys.argv[0])


try:
    #Reading train data
    training_file = str(input('\nInput You Training File\n'))
    train_peptide_data = pd.read_csv(training_file)

except IOError:
    print("Training File Not Found")
    exit(1)

try:
    #Reading test data
    testing_file = str(input('Input You Testing File\n'))
    test_peptide_data = pd.read_csv(testing_file)
except IOError:
    print("Testing File Not Found")
    exit(1)
    


#Extract Label columns from the training data
train_taregt_peptide = train_peptide_data[' Label']
# train_peptide_data = train_peptide_data.sample(frac = 1)


#-----------------------------------x-----------------

training_two_df = []
for text in train_peptide_data.iloc[:,0]:
    if all(char.isalpha() for char in text):
        training_two_df.append(text)





training_two_df = pd.DataFrame(training_two_df)
train_peptide_data = training_two_df

# train_taregt_peptide = train_peptide_data[' Label']


training_peptide_feature_20 = []
testing_peptide_feature_20=[]



  
#   training_peptide_feature_20.append(x)

from collections import Counter

training_peptide_feature_20 = []

for text in train_peptide_data.iloc[:,0]:
    # Convert text to lowercase
    text = text.lower()
    
    # Count the occurrence of each letter in the text
    letter_count = Counter(text)
    
    # Calculate the relative frequency of each letter
    letter_freq = [letter_count.get(chr(i), 0) / len(text) for i in range(ord('a'), ord('z')+1)]
    
    training_peptide_feature_20.append(letter_freq)






testing_peptide_feature_20 = []

for text in test_peptide_data.iloc[:,1]:
    # Convert text to uppercase
    text = text.upper()
    
    # Count the occurrence of each letter in the text
    letter_count = Counter(text)
    
    # Calculate the relative frequency of each letter
    letter_freq = [letter_count.get(chr(i), 0) / len(text) for i in range(ord('A'), ord('Z')+1)]
    
    testing_peptide_feature_20.append(letter_freq)




testing_peptide_feature_20 = pd.DataFrame(testing_peptide_feature_20)
training_peptide_feature_20 = pd.DataFrame(training_peptide_feature_20)


testing_peptide_feature_22 = []
testing_peptide_feature_22 = pd.DataFrame(testing_peptide_feature_22)
training_peptide_feature_22 = []
training_peptide_feature_22 = pd.DataFrame(training_peptide_feature_22)

testing_peptide_feature_22 = testing_peptide_feature_20[0]
training_peptide_feature_22 = training_peptide_feature_20[0]

for i in range(1,26):
  if(i!=1 and i!=23 and i!=25 and i!=(ord('J')-65) and i!=(ord('O')-65) and i!=(ord('U')-65)):
    testing_peptide_feature_22 = pd.concat([testing_peptide_feature_22,testing_peptide_feature_20[i]],axis=1)
    training_peptide_feature_22 = pd.concat([training_peptide_feature_22,training_peptide_feature_20[i]],axis=1)



arr = []
arr_not = ['B','X','Z','O','J','U']

addr_not = []

posa =[]
for i in range(26):
  for j in range(6):
    addr_not.append(i*26+(ord(arr_not[j])-65))
    addr_not.append((ord(arr_not[j])-65)*26+i)





training_featured = []
testing_featured=[]


for i in range(train_peptide_data.shape[0]):
  x = []
  for j in range(26*26):
    x.append(0)
  z = len(train_peptide_data.iloc[i][0])
  for j in range(z-1):
    y = (ord(train_peptide_data.iloc[i][0][j])-65)
    p = (ord(train_peptide_data.iloc[i][0][j+1])-65)
    pos = ((y)*26) + p

    x[pos]=x[pos]+1

  for j in range(26*26):
    x[j]=x[j]/z
  
  training_featured.append(x)


  

for i in range(test_peptide_data.shape[0]):
  x = []
  for j in range(26*26):
    x.append(0)
  z = len(test_peptide_data.iloc[i][1])
  for j in range(z-1):
    y = (ord(test_peptide_data.iloc[i][1][j])-65)
    p = (ord(test_peptide_data.iloc[i][1][j+1])-65)
    pos = ((y)*26)+p

    x[pos]=x[pos]+1
  for j in range(26*26):
    x[j]=x[j]/z
  
  testing_featured.append(x)


  
testing_featured = pd.DataFrame(testing_featured)
training_featured = pd.DataFrame(training_featured)


testing_featured_11 = []
testing_featured_11 = pd.DataFrame(testing_featured_11)
training_featured_11 = []
training_featured_11 = pd.DataFrame(training_featured_11)


testing_featured_11 = testing_featured[0]
training_featured_11 = training_featured[0]

for i in range(1,26*26):
  if((i in addr_not) == False):
    testing_featured_11 = pd.concat([testing_featured_11,testing_featured[i]],axis=1)
    training_featured_11 = pd.concat([training_featured_11,training_featured[i]],axis=1)




testing_featured_11 = pd.concat([testing_featured_11,testing_peptide_feature_22],axis=1)
training_featured_11 = pd.concat([training_featured_11,training_peptide_feature_22],axis=1)



xx_train, xx_test, yy_train, yy_test = train_test_split(
    training_featured_11, train_taregt_peptide, test_size=0.2, random_state=42)


xx_train.reset_index(inplace=True, drop=True)
yy_train = yy_train.reset_index(drop=True)

xx_test.reset_index(inplace=True, drop=True)
yy_test = yy_test.reset_index(drop=True)





# Create the ExtraTreesClassifier model
model = ExtraTreesClassifier(n_estimators=1000, random_state=42)

#model1 = RandomForestClassifier(max_depth=40, random_state=83, max_features='sqrt', n_estimators=1500)


# Fit the model to the training data
model.fit(xx_train, yy_train)
#model1.fit(xx_train, yy_train)
# Make predictions on the testing data
y_pred = model.predict(xx_test)

# Calculate the accuracy and MCC of the model
accuracy = accuracy_score(yy_test, y_pred)
auc = roc_auc_score(yy_test, y_pred)
auc = roc_auc_score(yy_test, y_pred)
# Print the accuracy and MCC of the model
print("Accuracy:-", accuracy)
print('Roc Auc score:-', auc)




#predicting testing data
y_pred = model.predict_proba(testing_featured_11)



valid_ans=[]
final_output = []
final_output.append(y_pred)
f = 1598
for i in range(f):
  ct_value =0
  for j in range(1):
    ct_value+=final_output[j][i][1]
  final_o = ct_value/1
  valid_ans.append(final_o)


df = pd.DataFrame(valid_ans)
df.columns = ['Label']
ID = np.arange(f) + 10001
ID = pd.DataFrame(ID)
ID.columns = ['ID']
final_data = pd.concat([ID, df], axis=1)
print('Output file with predicted classes generated with name -> run_15.csv')
final_data.to_csv('run_15.csv', index=False)
