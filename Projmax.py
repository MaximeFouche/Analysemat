#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#Question 5

def LU2(A):         #Renvoie la décomposition LU de A
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n, n))
    for j in range(n):
        for k in range(j,n):
            U[j][k] = A[j][k] - L[j,:j] @ U[:j,k]
        
        for k in range(j+1, n):
            L[k][j] = 1/U[j][j] * (A[k][j] -L[k,:j] @ U[:j,j])
    return L,U

def remontee(A,b):          #Résout Ax=b avec A triangulaire supérieure
    x = np.copy(b)
    n = len(b)
    for i in range(n-1, -1, -1):
        x[i] = x[i] - A[i][i+1:n] @ x[i+1:n]
        x[i] = x[i] / A[i][i]
    return x

def descente(A,b):       #Résout Ax=b avec A triangulaire inférieure
    x = np.copy(b)
    n = len(b)
    for i in range(n):
        x[i] = x[i] - A[i][:i] @ x[:i]
        x[i] = x[i] / A[i][i]
    return x

def MatriceA(n,tau):            #Crée la matrice A
    t = np.arange(n)
    V = 1 / (1 + t**2 * tau**2)
    
    A = np.eye(n)
    for i in range(1,n):
        d = np.diagflat((n - i)*[V[i]], i)
        
        A += d + d.T
    return A

def MatP(n,I):                     #Crée P selon la taille n et l'ensemble I des antennes en panne
        P=np.eye(n)
        for i in range(n):
            if i in I:
                P[i,i]=0
        return P
    
def MatAnz(A):                       #Retourne une matrice sans les colonnes et lignes nulles de A
    C=[]
    for i in range(len(A)):
        X=[]
        if A[i,0]!=0:
            for k in range(len(A)):
                if A[i,k]!=0:
                    X.append(A[i,k])
            C.append(X)
    C=np.array(C)
    return C

def Vectnz(V):                  #Renvoie un vecteur V privé de ses coefficients nuls
    C=[]
    for i in range(len(V)):
        if V[i]!=0:
            C.append(V[i])
    C=np.array(C)
    return C


def xfinal(x,I):             #Ajoute les 0 aux indices correspondants pour donner le xf solution de (8) ou (10)
    c=0
    n=len(x)+len(I)
    xf=np.zeros(n)
    for i in range(n):
        if i not in I:
            xf[i]=x[c]
            c+=1
    return xf

A=MatriceA(20,1)
pi=np.ones(20)
I=(6,7,14,15,16,17)  #Ensemble des pannes avec un décalage de -1, les matrices commencent en 0 sous Python
P=MatP(20,I)

PAP=P@A@P
PAPr=MatAnz(PAP)   #Version réduite de PAP
Ppi=P@pi
Ppir=Vectnz(Ppi) #Version réduite de Ppi
L,U=LU2(PAPr)

y=descente(L,Ppir)
x=remontee(U,y)
xf=xfinal(x,I)

print(xf) 
print(np.allclose(PAP@xf, Ppi))       #Vérifie que PAPx=Ppi
print(np.allclose(P@xf, xf))            #Vérifie que Px=x

Ax = A@xf
plt.plot(Ax, label="Ax=π")
plt.step(np.arange(20),xf, label="x")   #Affichage en escalier
plt.title('Question 5')
plt.legend()
plt.show()               #Permet d'afficher les deux graphes sur deux plots différents (on sépare Question 5 et 6)

#Question 6
A=MatriceA(20,1)
I=(6,7,14,15,16,17)  #Ensemble des pannes décalé de -1
P=MatP(20,I)
pi=np.ones(20)

PAAP=P@A.T@A@P
PApi=P@A.T@pi
PAAPr=MatAnz(PAAP)   #Version réduite de PAAP
PApir=Vectnz(PApi)   #Version réduite de PApi

L1,U1=LU2(PAAPr)

y1=descente(L1,PApir)
x1=remontee(U1,y1)
xf1=xfinal(x1,I)

print(xf1) 
print(np.allclose(PAAP@xf1, PApi)) #Vérifie que PA.TAPx=PA.Tpi
print(np.allclose(P@xf, xf))       #Vérifie que Px=x


plt.plot(A@xf1, label="Ax=π")
plt.step(np.arange(20),xf1, label="x")
plt.title('Question 6')
plt.legend()
