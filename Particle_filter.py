import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
import math

from numpy.random import normal
from numpy.random import multivariate_normal
from scipy.linalg import expm, sinm, cosm
from scipy.stats import multivariate_normal
from numpy.random import choice

pi=math.pi


#(a)

def motion_model(Xt1, phi, para, del_t):
  phi_l, phi_r= phi
  r,w, std_dev_l, std_dev_r,_=para
  mu=0
  t_phi_l= phi_l + normal(mu,std_dev_l)
  t_phi_r= phi_r + normal(mu,std_dev_r)
  omega_dot=np.zeros((3,3))
  omega_dot[0][1]= (r/w)* (t_phi_r - t_phi_l)*(-1)
  omega_dot[1][0]= (r/w)* (t_phi_r - t_phi_l)
  omega_dot[0][2]= (r/2)* (t_phi_r + t_phi_l)
  omega_dot=omega_dot*del_t
  exp_mat= expm(omega_dot)
  result=np.matmul(Xt1, exp_mat)
  return result

#b

def measurement_model(Xt,zt,std_dev_p):
  w=[]
  wsum=0
  for x in Xt:
    m=(x[0][2],x[1][2])
    c= np.array([[1,0],[0,1]])*(std_dev_p**2)
    wi = multivariate_normal.pdf(zt, mean=m, cov=c)
    w.append(wi)
  wt=np.array(w)
  wsum=wt.sum()
  wt=np.multiply(wt, (1/wsum))
  return wt

#c

def PARTICLE_FILTER_PROPAGATE(t1, Xt1, phi, t2, para, N):
  Xt2=[]
  for i in range(0,N):
    xt2=motion_model(Xt1[i], phi, para, t2-t1)
    Xt2.append(xt2)
  Xpred=np.array(Xt2)
  return Xpred

#d

def PARTICLE_FILTER_UPDATE(Xt, zt,std_dev_p):
  wt= measurement_model(Xt,zt,std_dev_p)
  num=[]
  index=0
  Xbar=[]
  for i in range(0,N):
    num.append(i)
  index=choice(num,p=wt,size=N)
  Xt_bar=np.copy(Xt)
  for i in range(0,N):
    xp=Xt[index[i]][0][2]
    yp=Xt[index[i]][1][2]
    Xt_bar[i]=Xt[index[i]]
      # print(xp,yp)
    Xbar.append([xp,yp])
  Xbar=np.transpose(np.array(Xbar))
  return Xbar,Xt_bar

# Given
N= 1000

phi=[1.5,2]
r=0.25
w=0.5
std_dev_l=0.05
std_dev_r=0.05
std_dev_p=0.10
t1=0
para=[r,w, std_dev_l, std_dev_r,std_dev_p]

#e

N=1000
Rt=np.array([[1,0,0],[0,1,0],[0,0,1]])
X0=np.zeros((N,3,3))
for i in range(0,N):
  X0[i]=np.array([[1,0,0],[0,1,0],[0,0,1]])
t1=0
t2=10
plt.scatter(0, 0,c="purple", s=20)
pc=[]
p=PARTICLE_FILTER_PROPAGATE(t1,X0, phi,t2, para, N)
for point in p:
  x=point[0][2]
  y=point[1][2]
  pc.append([x,y])
  plt.scatter(point[0][2], point[1][2],c="Red", s=4)

pc=np.array(pc)
print(np.mean(pc,axis=0,dtype=float))
print(np.cov(np.transpose(pc)))
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# f

N = 1000
Rt = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
X0 = np.zeros((N, 3, 3))
for i in range(0, N):
  X0[i] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
t1 = 0
t = [5, 10, 15, 20]
color = ["black", "red", "green", "blue"]
plt.scatter(0, 0, c="purple", s=40)
i = 0
for t2 in t:
  p = PARTICLE_FILTER_PROPAGATE(t1, X0, phi, t2, para, N)
  x = []
  y = []
  pc = []
  for point in p:
    x = point[0][2]
    y = point[1][2]
    pc.append([x, y])
    plt.scatter(point[0][2], point[1][2], c=color[i], s=4)

  X0 = p
  t1 = t2
  i = i + 1

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# g

N = 1000
Rt = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
X0 = np.zeros((N, 3, 3))
for i in range(0, N):
  X0[i] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
t1 = 0
t = [5, 10, 15, 20]
color_1 = ["black", "red", "green", "blue"]
color_2 = ["Grey", "pink", "#90EE90", "cyan"]
plt.scatter(0, 0, c="purple", s=50)
i = 0
# pt=[]
z5 = np.array([[1.6561, 1.22847]])
z10 = np.array([[1.0505, 3.1059]])
z15 = np.array([[-0.9875, 3.2118]])
z20 = np.array([[-1.6450, 1.1978]])
z = [z5, z10, z15, z20]
zk = 0

for t2 in t:
  p = PARTICLE_FILTER_PROPAGATE(t1, X0, phi, t2, para, N)
  for point in p:
    plt.scatter(point[0][2], point[1][2], c=color_2[i], s=5)
  zt = z[zk]
  Xbar, Xt_bar = PARTICLE_FILTER_UPDATE(p, zt, para[4])
  plt.scatter(Xbar[0], Xbar[1], c=color_1[i], s=5)

  X0 = Xt_bar
  t1 = t2
  i = i + 1
  zk += 1

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()