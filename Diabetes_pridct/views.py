from django.shortcuts import render
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def home(request):
    return render(request,'home.html')

def predict(request):
    return render(request,'predict.html')

def result(request):
    data = pd.read_csv("C:/Users/Vijay Bikkad/Desktop/diabetes.csv")
    #data=pd.read_csv("home\\vijay\\Videos\\diabetes.csv")
    X=data.drop("Outcome",axis=1)
    Y=data['Outcome']
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

    model=LogisticRegression(solver='liblinear')
    model.fit(X_train,Y_train)
    #prediction=model.predict(X_test)

    val1=float(request.GET.get('n1'))
    val2=float(request.GET.get('n2'))
    val3=float(request.GET.get('n3'))
    val4=float(request.GET.get('n4'))
    val5=float(request.GET.get('n5'))
    val6=float(request.GET.get('n6'))
    val7=float(request.GET.get('n7'))
    val8=float(request.GET.get('n8'))

    pred=model.predict([[val1,val2,val3,val4,val5,val6,val7,val8]])

    result2=""
    if pred==[1]:
         result1="Positive"
    else:
         result1="Negative"



    return render(request,'predict.html',{"result2":result1})