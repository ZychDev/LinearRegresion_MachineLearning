import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

#read data from file
data = pd.read_csv('student-mat.csv', sep=";")

#cleaning attributes, 33 is too much
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#create sonme training data
predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
#split data to train and test
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#model choice
linear = linear_model.LinearRegression()
#send train data to model thats gonna fit the best linear model line to this data
linear.fit(x_train, y_train)

#now we gonna check the accuracy of the model
acc = linear.score(x_test, y_test)
print(acc)

#how we use this model?
#we gonna check the (y = nx + b) for this line
print('Coefficient: ', linear.coef_) #+z+c+v+b...
print('Intercept: ', linear.intercept_) #a*x

#test my linear with data without "final" one (G3) and predicion it
predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])
