# Author: Akram Sheriff
# This is an ML Classification Problem based  data  analysis implementation for a  github based Workflow.
#!usr/bin/python
#
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics

read_file = pd.read_excel (r'C:\Users\isheriff\PycharmProjects\pythonProject1\WIFI_DCF_MAC_Simulation\IOT_Hackathon\git_log_dataset_v5.xlsx')
read_file.to_csv (r'C:\Users\isheriff\PycharmProjects\pythonProject1\WIFI_DCF_MAC_Simulation\IOT_Hackathon\Converted_git_log_dataset_v5.csv', index = None, header=True)

col_names = ['No.','patch hash', 'changed files number', 'modified files number',
                   'added files number', 'deleted files number', 'folder [/src/chip]',
                   'folder [/src/lib]', 'folder [/src/net]', 'folder [/src/platform]',
                   'folder [/src/system]', 'folder [/test]', 'folder [others]', 'file type [*.nc]',
                   'file type [*.target]', 'file type [*.bat]', 'file type [*.py]',
                   'file type [others]', 'tags']

data = pd.read_csv(r'C:\Users\isheriff\PycharmProjects\pythonProject1\WIFI_DCF_MAC_Simulation\IOT_Hackathon\Converted_git_log_dataset_v5.csv', header = 0, na_values= "NaN")
## Tags column in the input_CSV dataset is used for mapping to the sub-component dependency and ML Classsification is used for
## classification problem as per Light weight Design Document.

print("Input Data Statistics")
print(data.shape)
print(data.head())
print(data.info())
#sns.pairplot(data[1:])

print(data.isna().sum(axis=0),"\n The value is: \n")
print(data.isnull().sum() , "\n Count is: \n")
print(data.dropna())
print(data.sample(frac=1))

## As per Categorical Encoding , the Enumeration '0' corresponds to 'a' which is "PAN Join" and so on in below list
Classified_outputs = ['PAN Join','Frequency Hopping','Network Configuration','IPv6 ND','DHCPv6','Traffic','MPL','RPL','Error Handling','Security','LFD','Firmware Upgrading','etc']
Final_Target_component = {}

#Since the output Features are  String by nature by default, we need to Convert them into values by doing some Encoding
for count,value in enumerate(Classified_outputs):
    Final_Target_component[value] = count
print(Final_Target_component)

print("Input Dataset Features: ", col_names[2:])
print(f"\nThe output CG-Mesh Classified sub-components of features analyzed as per Datasets: {Classified_outputs}")
print(f"\nThe Encoded Target output is: {Final_Target_component}")

### Deleting the Original Manually Tagged Column "Tags" as it has String datatype  a,b,c,d,e...to be removed before Training"
del data['tags']

## Tagging 3 Random Values for the  different  git commit_ID's with  Different  random CG-Mesh component Tags fpr  now  so that  any Bugfix commit will have 3 naximum Sub-component CG-Mesh depencies for runing test cases.
random.seed(124)
##  12  Sub-Components of CG-Mesh Stack considered for mapping the  BugId Commit to the Test cases to be selected for execution and hence 12 used in Random value MAX

for i in range(845):
   data.loc[i,'tags_1'] = random.randint(0,12)
   data.loc[i,'tags_2'] = random.randint(0,12)
   data.loc[i,'tags_3'] = random.randint(0,12)
   data.loc[i,'tags'] = random.randint(0,12)

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
data = clean_dataset(data.iloc[:,2:])

## Ignoring the No: and the Commit ID hash Value: first 2 columns in the Dataset, splitting for 3 Targets
train_data, temp_test_data = train_test_split(data ,test_size=0.4, random_state=109)
print(train_data.shape)
print(temp_test_data.shape)

test_data, valid_data = train_test_split(temp_test_data,test_size= 0.5, random_state=109)
## Either Equal Weightage Mean Encoding  technique for sub-component mapping a,b,c,d,e... or Summation Techniques can be used to build the Statisical dataset for lassiifation by SVM

print(test_data.shape)
print(valid_data.shape)

del data['tags_1']
del data['tags_2']
del data['tags_3']

train_stats = train_data.describe()
train_stats.pop("tags")
sns.pairplot(train_stats[train_stats.columns], diag_kind = "kde")
plt.show()

train_stats = train_data.describe()
train_stats.pop("tags")
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_data.pop("tags")
test_labels = test_data.pop("tags")
valid_labels = valid_data.pop("tags")

def norm(x):
    return (x - train_stats['mean'])/train_stats['std']

normed_train_data = norm(train_data)
normed_test_data = norm(test_data)
normed_valid_data = norm(valid_data)
normed_train_data.head(10)

model = svm.SVC(C=1,kernel = "linear")
model.fit(normed_train_data,train_labels)
y_pred = model.predict(normed_test_data)

example_batch = normed_test_data[:10]
example_result = model.predict(example_batch)
print("Predicted Values: ")
print(example_result)

y_pred = model.predict(normed_train_data)
print("Accuracy: ", metrics.accuracy_score(train_labels,y_pred))

y_pred = model.predict(normed_valid_data)
print("Accuracy: ", metrics.accuracy_score(valid_labels, y_pred))

y_pred = model.predict(normed_test_data)
print("Accuracy: ", metrics.accuracy_score(test_labels,y_pred))

ax = plt.subplot()
predict_results = model.predict(normed_test_data)
cm = confusion_matrix(predict_results , predict_results)
sns.heatmap(cm, annot=True , ax = ax)
plt.show()

ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Confusion_Matrix')











