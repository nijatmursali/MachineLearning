import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as graph  #matlab versiyasi pythonun
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd #csv faylini read etmek ucun
import csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

#import datamodify as dat

def datatobeTaken():
    data = pd.read_csv("turboazmodified.csv")

    dataframe = pd.DataFrame(data, columns= ['Yurush','Qiymet','Buraxilis ili'])
    yurush = data['Yurush']
    qiymet = data['Qiymet']
    buraxilishili = data['Buraxilish ili']
    yurush = (yurush - yurush.mean()) / yurush.std()
    yurush = np.c_[np.ones(yurush.shape[0]),yurush]
    qiymet = (qiymet - qiymet.mean()) / qiymet.std()
    buraxilishili = (buraxilishili - buraxilishili.mean()) / buraxilishili.std()
    yurush.astype(float)
    m = len(qiymet)
    return yurush, qiymet, buraxilishili;


data = pd.read_csv("turboazmodified.csv")

def firstplot():
    yurush, qiymet, buraxilishili = datatobeTaken();
    m = len(yurush)
    for i in range(0, m):
        if '+08' in yurush[i]:
            yurush[i] = float(yurush[i].replace('+08',''))
        if 'e' in yurush[i]:
            yurush[i] = yurush[i].replace('e','')
            yurush[i] = yurush[i] * 2.7
    graph.xlabel('Yurush')
    graph.scatter(yurush[:,1], qiymet, edgecolors='red')
    graph.ylabel('Qiymet')
    graph.title('Yurush vs Qiymet')
    graph.show()

def secondplot():
    yurush, qiymet, buraxilishili = datatobeTaken();
    graph.scatter(buraxilishili, qiymet, edgecolor = 'b')
    graph.xlabel('Buraxilis')
    graph.ylabel('Qiymet')
    graph.title('Buxaltir')
    graph.show()

def thirdplot():
    yurush, qiymet, buraxilishili = datatobeTaken();
    fig = graph.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(yurush[:,1], qiymet, buraxilishili)
    graph.show()


def heuristicFunct(yurush, theta):
    return np.dot(yurush, theta)

def costFunction(yurush, qiymet, theta):
    m = 1328
    sumofvariables = 0
    for i in range(1, m):
        sumofvariables +=(heuristicFunct(yurush[i], theta) - qiymet[i])**2
    sumofvariables = sumofvariables * (1.0/(2*m))

    return sumofvariables


def updateruletobeComputed(yurush, qiymet, theta, learningrate, numberofiterations):

    theta[0] = theta[0] - learningrate * costFunction(yurush, qiymet, theta) * 2
    theta[1] = theta[1] - learningrate * costFunction(yurush, qiymet, theta) * 2
    return theta

def plottingCostFunction(sumofvariables):

    graph.title("Cost Function is plotted")
    graph.xlabel("Number of iterations")
    graph.ylabel("Cost")
    graph.plot(sumofvariables)
    graph.show()


def test1(yurush, qiymet, buraxilishili):
    #yurush, qiymet, buraxilishili = datatobeTaken();

    yurush = 240000
    buraxilishili = 2000
    qiymet = 11500

    yurush = (yurush - yurush.mean()) / yurush.std()
    qiymet = (qiymet - qiymet.mean()) / qiymet.std()
    buraxilishili = (buraxilishili - buraxilishili.mean()) / buraxilishili.std()

    ntheta, costh = updateruletobeComputed(yurush, qiymet, theta, learningrate, numberofiterations)

    predprice = ntheta[2] * buraxilishili + ntheta[1] * yurush + ntheta[0]

    normqiymet =  predprice * qiymet.std() + qiymet.mean()
    actqiymet = qiymet * qiymet.std() + qiymet.mean()

    print(normqiymet)
    print(actqiymet)

def test2(yurush, qiymet, buraxilishili):

    yurush = 415558
    buraxilishili = 1996
    qiymet = 8800

    yurush = (yurush - yurush.mean()) / yurush.std()
    #yurush = np.c_[np.ones(yurush.shape[0]),yurush]

    qiymet = (qiymet - qiymet.mean()) / qiymet.std()
    #qiymet = np.c_[np.ones(qiymet.shape[0]),qiymet]

    buraxilishili = (buraxilishili - buraxilishili.mean()) / buraxilishili.std()
    #buraxilishili = np.c_[np.ones(buraxilishili.shape[0]),buraxilishili]

    ntheta, costh = updateruletobeComputed(yurush, qiymet, theta, learningrate, numberofiterations)

    predprice = ntheta[2] * buraxilishili + ntheta[1] * yurush + ntheta[0]

    normqiymet =  predprice * qiymet.std() + qiymet.mean()
    actqiymet = qiymet * qiymet.std() + qiymet.mean()

    print(normqiymet)
    print(actqiymet)

def linearRegrTrain():
    linearreg = LinearRegression()
    yurush, qiymet, buraxilishili = datatobeTaken();
    yurushTrain, yurushTest, buraxilishiliTrain, buraxilishiliTest = train_test_split(yurush, buraxilishili, test_size = 1/3, random_state = 0)

    linearreg.fit(yurushTrain, buraxilishiliTrain)
    buraxilishiliPredict = linearreg.predict(yurushTest)

    graph.scatter(yurushTrain, buraxilishiliTrain, color = 'black')
    graph.plot(yurushTrain, linearreg.predict(yurushTrain), color = 'red')
    graph.title("Hello")
    graph.xlabel("Yurush")
    graph.ylabel("Buraxilish ili")
    graph.show()

def linearRegrTest():
    linearreg = LinearRegression()
    yurush, qiymet, buraxilishili = datatobeTaken();
    yurushTrain, yurushTest, buraxilishiliTrain, buraxilishiliTest = train_test_split(yurush, buraxilishili, test_size = 1/3, random_state = 0)

    linearreg.fit(yurushTest, buraxilishiliTest)
    buraxilishiliPredict = linearreg.predict(yurushTrain)

    graph.scatter(yurushTest, buraxilishiliTest, color = 'black')
    graph.plot(yurushTest, linearreg.predict(yurushTest), color = 'red')
    graph.title("Hello")
    graph.xlabel("Yurush")
    graph.ylabel("Buraxilish ili")
    graph.show()

def normequation(yurush, qiymet):
    yurush, qiymet, buraxilishili = datatobeTaken();
    yurushTranspose = yurush.T
    normeq = inv(yurushTranspose.dot(yurush)).dot(yurushTranspose).dot(qiymet)

    print("The value we get from Normal Equation is %s" % (normeq))
    return normeq


def PolynomialModel(degree, yurush, qiymet):
    yurush, qiymet, buraxilishili = datatobeTaken();

    poly = PolynomialFeatures(degree=degree)

    polyyurush = poly.fit_transform(yurush)
    regs = LinearRegression()
    regs.fit(polyyurush, qiymet)


    actval = (yurush - polyyurush.mean()) / yurush.std()
    print(actval)


    #print(yurush.sh)
    graph.scatter(yurush[:,0], qiymet, color = "red")
    graph.plot(yurush, regs.predict(poly.fit_transform(yurush)), color = 'blue')
    graph.show()

def tobePrinted():


    #theta = [1,1,1]
    theta = [0,0]
    numberofiterations = 5 #no. of interations to learn
    learningrate = 0.01 #learning rate is 0.01
    m = 1328
    yurush, qiymet, buraxilishili = datatobeTaken();

    for i in range(numberofiterations):
        costfinished = costFunction(yurush, qiymet, theta) #getting cost from cost function
        theta = (updateruletobeComputed(yurush, qiymet, theta, learningrate, numberofiterations))
        print("Cost function in iteration %d is %s" % (i, costfinished))

        print(theta[0],theta[1])

    graph.scatter(buraxilishili, qiymet, edgecolors='b')

    #graph.plot(buraxilishili, qiymet)
    #graph.show(block = True)
    #graph.close()

    #PolynomialModel(2, yurush, qiymet)
    #normequation(yurush, qiymet)
    #test1(yurush, qiymet, buraxilishili)
    #plottingCostFunction()
    #firstplot()
    #linearRegrTrain()
    #linearRegrTest()
    #secondplot()
    #thirdplot()
    test1(yurush, qiymet, buraxilishili)


tobePrinted()
