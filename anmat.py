#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 13:02:37 2022

@author: maxfou
"""
# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as sc


"""

    Question 3

"""
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


def remontee2(A,b):
    x = np.copy(b)
    n = len(b)
    for i in range(n-1, -1, -1):
        if P[i,i]==0:
            x[i]=0
        else:
            x[i] = x[i] - A[i][i+1:n] @ x[i+1:n]
            x[i] = x[i] / A[i][i]
    return x

def descente2(A,b):
    x = np.copy(b)
    n = len(b)
    for i in range(n):
        if P[i,i]==0:
            x[i]=0
        else:
            x[i] = x[i] - A[i][:i] @ x[:i]
            x[i] = x[i] / A[i][i]
    return x



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



def MatriceA(n,tau):
    t = np.arange(n)
    V = 1 / (1 + t**2 * tau**2)
    
    A = np.eye(n)
    for i in range(1,n):
        d = np.diagflat((n - i)*[V[i]], i)
        
        A += d + d.T
    return A

A = MatriceA(20,1)
B=MatriceA(4, 1)

pi = np.ones(20)

pi2=np.ones(4)
pi2[2]=0
C=np.eye(4)
C[2,2]=0


P=np.eye(20)
for i in range(20):
    if i==7 or i==8 or i==15 or i==16 or i==17 or i==18:
        P[i,i]=0

L,U = LU2(A)

y = descente2(P@L, P@pi)
x1 = remontee2(U,y)
print(x1)
print(P@A@x1, P@pi)
print(np.allclose(A@x1, pi)) # True
print(np.allclose(P@A@x1, P@pi)) 
print(np.allclose(P@x1, x1)) 

"""

    Question 4    

"""



def Etape1(t):
    F = [np.array([1/t[0]])]
    n = len(t)
    for i in range(1,n):
        delta = t[1:i+1] @ F[-1]

        A = np.zeros((2,2))
        A+=delta
        np.fill_diagonal(A,1)

        b = np.array([0,1])

        
        sol = sc.solve(A,b)
        
        f = sol[0] * np.array([*F[-1][::-1], 0]) + sol[1] * np.array([0, *F[-1]])

        F.append(f)
    
    return F

F = Etape1(A[0])

def Etape2(t,b):
    x = np.array([b[0]/t[0]])
    F = Etape1(t)
    n = len(t)
    
    for i in range(1,n):
        teta = b[i] - ( t[1:i+1] @ x[::-1] )
        x = np.array([*x, 0]) + teta * F[i]
        
    return x

x2 = Etape2(A[0], pi)
    
np.allclose(x1, x2) # True

