# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#Question 5

def LU2(A):
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n, n))
    for j in range(n):
        for k in range(j,n):
            U[j][k] = A[j][k] - L[j,:j] @ U[:j,k]
        
        for k in range(j+1, n):
            L[k][j] = 1/U[j][j] * (A[k][j] -L[k,:j] @ U[:j,j])
    return L,U

def remontee(A,b):
    x = np.copy(b)
    n = len(b)
    for i in range(n-1, -1, -1):
        x[i] = x[i] - A[i][i+1:n] @ x[i+1:n]
        x[i] = x[i] / A[i][i]
    return x

def descente(A,b):
    x = np.copy(b)
    n = len(b)
    for i in range(n):
        x[i] = x[i] - A[i][:i] @ x[:i]
        x[i] = x[i] / A[i][i]
    return x

def MatriceA(n,tau):
    t = np.arange(n)
    V = 1 / (1 + t**2 * tau**2)
    
    A = np.eye(n)
    for i in range(1,n):
        d = np.diagflat((n - i)*[V[i]], i)
        
        A += d + d.T
    return A

def MatP(n,panne):
        P=np.eye(n)
        for i in range(n):
            if i in panne:
                P[i,i]=0
        return P
    
def MatAnz(A):
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

def Vectnz(V):
    C=[]
    for i in range(len(V)):
        if V[i]!=0:
            C.append(V[i])
    C=np.array(C)
    return C


def xfinal(x,panne):
    c=0
    n=len(x)+len(panne)
    xf=np.zeros(n)
    for i in range(n):
        if i not in panne:
            xf[i]=x[c]
            c+=1
    return xf

A=MatriceA(20,1)
panne=(6,7,14,15,16,17)
P=MatP(20,panne)

PAP=P@A@P
PAPr=MatAnz(PAP)
pir=np.ones(14)
pif=np.ones(20)
L,U=LU2(PAPr)

y=descente(L,pir)
x=remontee(U,y)
xf=xfinal(x,panne)

print(xf) 
print(np.allclose(PAP@xf, P@pif)) 
print(np.allclose(P@xf, xf)) 

Ax = A@xf
plt.plot(Ax, label="Ax=π")
plt.step(np.arange(20),xf, label="x")
plt.title('Question 5')
plt.legend()
plt.show()

#Question 6
A=MatriceA(20,1)
panne=(6,7,14,15,16,17)
P=MatP(20,panne)
pif=np.ones(20)

PAAP=P@A.T@A@P
PApi=P@A.T@pif
PAAPr=MatAnz(PAAP)
PApir=Vectnz(PApi)

L1,U1=LU2(PAAPr)

y1=descente(L1,PApir)
x1=remontee(U1,y1)
xf1=xfinal(x1,panne)

print(xf1) 
print(np.allclose(PAAP@xf1, PApi)) 
print(np.allclose(P@xf, xf)) 


plt.plot(A@xf1, label="Ax=π")
plt.step(np.arange(20),xf1, label="x")
plt.title('Question 6')
plt.legend()
