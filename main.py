import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

#read data from file
data = pd.read_csv('student-mat.csv', sep=";")

#cleaning attributes, 33 is too much
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#create sonme training data
predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#training model
#for loop, the best model
'''
best = 0
for _ in range(100):
    #split data to train and test
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    #model choice
    linear = linear_model.LinearRegression()
    #send train data to model thats gonna fit the best linear model line to this data
    linear.fit(x_train, y_train)

    #now we gonna check the accuracy of the model
    acc = linear.score(x_test, y_test)
    print(acc)
    if acc > best:
        best = acc
        #safe model in picke directory
        with open("studentmodel.pickle", "wb") as f:
            #source destination
            pickle.dump(linear, f)
print("Best model is: ", best)
'''

#open pickle file
pickle_in = open("studentmodel.pickle", "rb")
#load pickle data
linear = pickle.load(pickle_in)

#how we use this model?
#we gonna check the (y = nx + b) for this line
#print('Coefficient: ', linear.coef_) #+z+c+v+b...
#print('Intercept: ', linear.intercept_) #a*x

#test my linear with data without "final" one (G3) and predicion it
predictions = linear.predict(x_test)
accuracy = linear.score(x_test,y_test)
print(accuracy)

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

#plotting data
p = 'G2'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
