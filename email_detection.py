import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


email_data=pd.read_csv("N:\Machine learning\Algorithms\spam_ham_dataset.csv")
email_data=email_data.drop(['Unnamed: 0','label'],axis=1)

# print(email_data['label_num'].value_counts())


                                           #-----------check for null values----------

# print(email_data.isnull().any())

                                         #---------data preprocessing---------

stopset = set(stopwords.words("english"))
corpus = []
for i in range(0, len(email_data)):
  e_mail = re.sub('[^a-zA-Z]', ' ', email_data['text'][i])
  e_mail = e_mail.split()
  ps = PorterStemmer()
  e_mail = [ps.stem(word) for word in e_mail if not word in set(stopwords.words('english'))]
  e_mail = ' '.join(e_mail)
  corpus.append(e_mail)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x=cv.fit_transform(corpus)



y=email_data['label_num']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)

model=KNeighborsClassifier(n_neighbors=1)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

confusion_mat=confusion_matrix(y_test,y_pred,labels=None)
print("confusion_mat = ",confusion_mat)
print("Accuracy Score:",accuracy_score(y_test,y_pred))              
print("precision score = ",precision_score(y_test, y_pred))         
print("recall score = ",recall_score(y_test, y_pred))               
print("F1 score = ",f1_score(y_test, y_pred))         