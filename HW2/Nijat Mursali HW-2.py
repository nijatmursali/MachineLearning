import pandas as pd
import numpy as np
import matplotlib.pyplot as graph
import math
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

""" LOADING DATA """
def openCSV():
    data = pd.read_csv("exams.csv")
    exam_1 = data['exam_1']
    exam_2 = data['exam_2']
    passorNOT = data['admitted']

    exam_1 = (exam_1 - exam_1.mean()) / exam_1.std()
    exam_1 = np.c_[np.ones(exam_1.shape[0]),exam_1]
    exam_2 = (exam_2 - exam_2.mean()) / exam_2.std()
    passorNOT = (passorNOT - passorNOT.mean()) / passorNOT.std()
    return exam_1,exam_2,passorNOT;

""" Visualization part 1. """
def firstplot():
    exam_1, exam_2, passorNOT = openCSV();
    firstvar = exam_1[:,1]
    secondvar = exam_2
    graph.xlabel('First Exam Score')
    #graph.scatter(firstvar, secondvar, c = 'b')
    graph.scatter(exam_1[:,1],exam_2, color="red")
    #graph.scatter(exam_1[:,1],exam_2, color = "blue")
    graph.ylabel('Second Exam Score')
    graph.title('First vs Second')
    graph.show()

""" SIGMA FUNCTION """
def sigmofunc(exam_1):
    exam_1 = 4;
    func = float(1 / (1 +math.exp(-exam_1)))
    return func

""" COST FUNCTION """
def costfunc(exam_1,exam_2):
    exam_1, exam_2, passorNOT = openCSV();
    m = len(exam_1)
    for i in  range(0,2):
        cost = -1/m * (exam_2[i]*np.log(sigmofunc(exam_1[i]))+(1-exam_2[i])*np.log(1-sigmofunc(exam_1[i])))
        #print("The cost function in iteration %d is %.3f" % (i,cost))

    return cost

""" GRADIENT FUNCTION """
def graddescent(exam_1, exam_2,theta, learningrate, numofiter):
    exam_1, exam_2, passorNOT = openCSV();

    m = len(exam_1)
    for i in range(1, m):
        theta[0] = theta[0] - learningrate*(sigmofunc(exam_1[i]) - exam_2[i]) * exam_1[i]
        theta[1] = theta[1] - learningrate*(sigmofunc(exam_1[i]) - exam_2[i]) * exam_1[i]

    return theta

""" PLOTTING EXAM 1 EXAM 2 AND ADMIITTED """
def plottingall(exam_1, exam_2,passorNOT):
    exam_1, exam_2, passorNOT = openCSV();

    graph.scatter(exam_1[:,1],exam_2, color="red")
    graph.plot(passorNOT)
    graph.show()

""" TESTING PART """
def testing():
    exam_1, exam_2, passorNOT = openCSV();
    #data1 = [55,70,1]
    #data2 = [40,60,0]
    data1 = np.random.rand(100)
    data2 = np.random.rand(100)
    exam_1 = 50
    exam_2 = 60

    priceprediction = data1 * exam_2 + data1 * exam_1 + data1

    pred =  priceprediction * exam_2 + exam_2
    actdata = exam_2 * exam_2 + exam_2
    print("This is for test number 1.")
    print("Predicted values I got is:")
    print(pred)
    print("Actual data is:",(actdata))

    print("\nThis is for test number 2.\n")
    test1 = [55,70,1]
    test2 = [40, 60, 0]
    print(accuracy_score(test2, test1))
    print(accuracy_score(test2, test1, normalize=False))

""" LINEAR REGRESSION FUNCTION """
def LinearRegro():
    linearreg = LinearRegression()
    exam_1, exam_2, passorNOT = openCSV();
    exam_1Train, exam_1Test, exam_2Train, exam_2Test = train_test_split(exam_1, exam_2, test_size = 1/3, random_state = 0)
    linearreg.fit(exam_1Test, exam_2Test)
    exam_2Predict = linearreg.predict(exam_1Train)

    graph.scatter(exam_1Test[:,1], exam_2Test, color = 'black')
    graph.plot(exam_1Test, linearreg.predict(exam_1Test), color = 'red')
    graph.title("Linear Regression Test ")
    graph.xlabel("Exam #1")
    graph.ylabel("Exam #2")
    graph.show()

""" EVERYTHING PRINTED HERE """
def forPrinting():
    theta = [0,0]
    numofiter = 100
    learningrate = 0.01
    exam_1, exam_2, passorNOT = openCSV(); #GETTING DATA HERE
    m = len(exam_1)


    firstplot() #Visualization 1

    print("The value we get from sigma function is",sigmofunc(5))
    for num in range(0, numofiter):
        costfinal = costfunc(exam_1,exam_2) #COST FUNCTION PRINTED HERE
        theta = graddescent(exam_1, exam_2, theta, learningrate, numofiter) #GRADIENT PRINTED HERE
        #print("Cost function in iteration %d is %s" % (num, costfinal))
        #print("First theta is equal to %s and the second theta equal to %s" %(theta[0],theta[1]))

    plottingall(exam_1,exam_2, passorNOT) #Visualization PART 2
    testing() #TESTING PART
    LinearRegro() #LINEAR REGRESSION



forPrinting() #PRINT FUNCTION CALLED
