import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_two = []
cur =0
for i in range(train_data.shape[0]):
  z  = len(train_data.iloc[i][0])
  x =0
  for j in range(z):
    y = (ord(train_data.iloc[i][0][j])-65)
    if((y<0 or y>25)==True):
      x=1
      break
  if(x==0):
    train_two.append(train_data.iloc[i])
    cur+=1


train_two = pd.DataFrame(train_two)
train_data = train_two

train_label = train_data['Label']
train_feature20 = []
test_feature20=[]


for i in range(train_data.shape[0]):
  x = []
  for j in range(26):
    x.append(0)
  z = len(train_data.iloc[i][0])
  for j in range(z):
    y = (ord(train_data.iloc[i][0][j])-65)
    x[y]=x[y]+1

  for j in range(26):
    x[j]=x[j]/z


  train_feature20.append(x)


for i in range(test_data.shape[0]):
  x = []
  for j in range(26):
    x.append(0)
  z = len(test_data.iloc[i][1])
  for j in range(z):
    y = (ord(test_data.iloc[i][1][j])-65)
    x[y]=x[y]+1
  for j in range(26):
    x[j]=x[j]/z
  test_feature20.append(x)


test_feature20 = pd.DataFrame(test_feature20)
train_feature20 = pd.DataFrame(train_feature20)

test_feature2020 = []
test_feature2020 = pd.DataFrame(test_feature2020)
train_feature2020 = []
train_feature2020 = pd.DataFrame(train_feature2020)

test_feature2020 = test_feature20[0]
train_feature2020 = train_feature20[0]

for i in range(1,26):
  if(i!=1 and i!=23 and i!=25 and i!=(ord('J')-65) and i!=(ord('O')-65) and i!=(ord('U')-65)):
    test_feature2020 = pd.concat([test_feature2020,test_feature20[i]],axis=1)
    train_feature2020 = pd.concat([train_feature2020,train_feature20[i]],axis=1)

arr = []
notar = ['B','X','Z','O','J','U']

notadd = []

posa =[]
for i in range(26):
  for j in range(6):
    notadd.append(i*26+(ord(notar[j])-65))
    notadd.append((ord(notar[j])-65)*26+i)

train_feature = []
test_feature=[]


for i in range(train_data.shape[0]):
  x = []
  for j in range(26*26):
    x.append(0)
  z = len(train_data.iloc[i][0])
  for j in range(z-1):
    y = (ord(train_data.iloc[i][0][j])-65)
    p = (ord(train_data.iloc[i][0][j+1])-65)
    pos = ((y)*26) + p

    x[pos]=x[pos]+1

  for j in range(26*26):
    x[j]=x[j]/z

  train_feature.append(x)

for i in range(test_data.shape[0]):
  x = []
  for j in range(26*26):
    x.append(0)
  z = len(test_data.iloc[i][1])
  for j in range(z-1):
    y = (ord(test_data.iloc[i][1][j])-65)
    p = (ord(test_data.iloc[i][1][j+1])-65)
    pos = ((y)*26)+p

    x[pos]=x[pos]+1
  for j in range(26*26):
    x[j]=x[j]/z

  test_feature.append(x)

test_feature = pd.DataFrame(test_feature)
train_feature = pd.DataFrame(train_feature)


import pandas as pd
test_feature1 = test_feature[0]
train_feature1 = train_feature[0]

for i in range(1,26*26):
  if((i in notadd) == False):
    test_feature1 = pd.concat([test_feature1,test_feature[i]],axis=1)
    train_feature1 = pd.concat([train_feature1,train_feature[i]],axis=1)


from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
# Generate some example data 
X, y = make_classification(n_samples=100, n_features=20, random_state=42)
k = 10  # Replace with your desired value
i=0
y_final=[]
# Feature selection
selector = SelectKBest(f_classif, k=k)
train_feature1_selected = selector.fit_transform(train_feature1, train_label)
test_feature1_selected = selector.transform(test_feature1)
while(i!=1):
    # Create the base classifiers
    rf_classifier = RandomForestClassifier(max_depth=60, random_state=83, max_features='sqrt', n_estimators=4000)
    # Define the stacking classifier
    estimators = [('Random Forest', rf_classifier)]
    stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier())
    # Fit the stacking classifier to the selected training features
    stacking_classifier.fit(train_feature1_selected, train_label)
    y_pred = stacking_classifier.predict(test_feature1_selected)
    # You can also calculate the probability estimates if needed
    y_pred_proba = stacking_classifier.predict_proba(test_feature1_selected)
    y_final.append(y_pred)
    i+=1


ans=[]
for i in range(310):
  ct =0
  for j in range(1):
    ct+=y_final[j][i][1]
  avg = ct/1
  ans.append(avg)


import pandas as pd
df = pd.DataFrame(ans)
df.columns = ['Label']
ID = np.arange(310) + 501
ID = pd.DataFrame(ID)
ID.columns = ['ID']
final_data = pd.concat([ID, df], axis=1)
final_data.to_csv('result4.csv', index=False)