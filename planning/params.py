import numpy as np
import pickle

k1= 0.1;k2= 0.2
k1=3.968; k2=2.517; k3=0.1353; k4=-0.5197; k5 = 4.651; k6 = 2.335
A = np.array([[0,0,1,0],[0,0,0,1],[0,0,-k2,k3],[0,0,k4,-k1]])
B = np.array([[0,0],[0,0],[k6,0],[0,k5]])
R = np.array([[5,0],[0,5]])

BRB = B@np.linalg.inv(R)@B.T

file = open('sp_var.pkl','wb')
pickle.dump([k1,k2,A,B,R,BRB],file)