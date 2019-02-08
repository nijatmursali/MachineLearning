import numpy as nump
from numpy import *
import matplotlib.pyplot as graph  #matlab versiyasi pythonun
import pandas #csv faylini read etmek ucun
import csv


def plot(yurush, qiymet):
    graph.plot(yurush, qiymet, '*', markersize=7, color = "black")
    graph.xlabel("Yürüş kilometri")
    graph.ylabel("Maşının qiyməti")
    graph.show(black = True)
    graph.close()


def turboazData():
    data = pandas.read_csv("turboaz.csv")

    #with open("turboaz.csv", "r", encoding="utf-8") as turboaz:
    #    reader = csv.reader(turboaz)
    #    for row in reader:
    #        print(row[9],row[3], row[13])
    #yurush = data[:,0].values
    #qiymet = data[:,1].values

    yurush = data['Yurush']
    qiymet = data['Qiymet']
    buraxilishili = data['Buraxilish ili']

    graph.scatter(yurush,qiymet, edgecolors = 'r')
    graph.xlabel('Yurush')
    graph.ylabel('Qiymeti')
    graph.title('Yurush ve Qiymet')
    graph.show()

    #yurush = row[9]
    #qiymet = row[13]
    #buraxilishili = row[3]

    m = len(qiymet)

    #qiymet = qiymet.reshape(m,1)
    return yurush, qiymet;

    #turboaz.close()
def datatoPlot():
    yurush, qiymet,buraxilishili = turboazData();
    plot(yurush,qiymet)
    plot(buraxilishili, qiymet)
    graph.show(block = True)

def heuristicFunct():
    return yurush.dot(theta)

def costFunction():
    m = len(qiymet)
    sumofvariables = 0
    for i in range(1, m):
        sumofvariables +=(hfunction(yurush[i], theta) - qiymet[i])**2
    sumofvariables = sumofvariables * (1.0/(2*m))

    return sumofvariables

def printedFiles():
    learningrate = 0.001
    numofiteration = 10000


    yurush, qiymet = turboazData();
    m = len(qiymet)
    #yurush = c_[ones((m,1)),qiymet]

    theta = [0,0]

    for i in range(0, numofiteration):
        costfinished = costFunction(yurush,qiymet,theta)
        print("Cost function in iteration '%d' '%d': ",i, costfinished)

    #plot(yurush[:,1], qiymet[:,0]).values
    graph.plot(yurush[:,1].values,yurush.dot(theta))
    graph.show(block = True)
    graph.close()

    plot()

printedFiles()
