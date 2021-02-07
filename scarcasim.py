from sklearn import datasets
import numpy as np
from sklearn import svm

x = np.array([[0.3,0.1,0.2,0.1,0.1,0.2],[0.1,0.2,0.2,0.1,0.1,0.2],[0.1,0.15,0.25,0.1,0.1,0.2]])
y = np.array([0,1,1])
data = np.array([[0.1,0.3,0.2,0.1,0.1,0.2],])
clf=svm.SVC()
clf.fit(x,y)
dic={0:'sarcasm',1:'not sarcasm'}
print(dic[clf.predict(data)[0]])