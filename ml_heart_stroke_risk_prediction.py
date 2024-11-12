import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def get_drive_csv_url(share_link):
    file_id = share_link.split('/d/')[1].split('/view')[0]
    return f'https://drive.google.com/uc?id={file_id}'

drive_link = "https://drive.google.com/file/d/1BECHp3vJICRjQ2vjSV85_7AGsQgVmRUm/view?usp=drive_link"
csv_url = get_drive_csv_url(drive_link)

df = pd.read_csv(csv_url)

scaler = StandardScaler()
X = df.drop(columns='target', axis=1)
Y = df['target']
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

model = LogisticRegression(max_iter=1000, solver='saga') 
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(f'This is the accuracy score on training data {training_data_accuracy}')

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f'This is the accuracy score on testing data {testing_data_accuracy}')


target = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2) #Input

target_df = pd.DataFrame([target], columns=X.columns)

target_scaled = scaler.transform(target_df)

prediction = model.predict(target_scaled)

if prediction[0] == 0:
    print("Good News! Patient doesn't have heart disease")
else:
    print("Oh! Patient should visit the doctor")
