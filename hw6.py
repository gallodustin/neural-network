#!/usr/bin/env python
# coding: utf-8

# In[78]:


# {f_w:X->Y} with weights w
# x = (x1,x2) in R^2
# y in R
# i-th ReLU unit has weights bi, wi1, wi2 and outputs oi = max(bi + wi1*xi+wi2*x2,0)
# output layer with single sigmoid unit and weights bo,wo1,...,wo20
# output y = sigma(b0+w01)

import numpy as np
import math
import matplotlib.pyplot as plt
import random


# In[170]:


# initialize hidden layer weights to 0
b = np.zeros((20,1))
w = np.zeros((20,2))

# initialize output layer weights to 0
b0 = 0
w0 = np.zeros((1,20))

# initialize input point to 0
x = np.zeros((1,2))


# In[171]:


# sigmoid function
# sigma(x)=1/(1+exp(-x))
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# In[172]:


def ReLU(x, b, w, num=20):
    o = np.zeros((1,num))
    for i in range(num):
        o[0,i] = max(b[i] + w[i,0]*x[0] + w[i,1]*x[1], 0)
    return o


# In[173]:


def output(b, w, o, num=20):
    y = b
    for i in range(num):
        term = w[0,i]
        term *= o[0,i]
        y += term
    return sigmoid(y)


# In[174]:


# Q1.1

# initialize all hidden layer weights to 0.5
b.fill(0.5)
w.fill(0.5)

# initialize all output layer weights to 1
b0 = 1
w0.fill(1)

print("x = (1,1):")
o = ReLU((1,1), b, w)
print("o =",o)
print("y =",output(b0,w0,o))

print("x = (1,-1):")
o = ReLU((1,-1), b, w)
print("o =",o)
print("y =",output(b0,w0,o))

print("x = (-1,-1):")
o = ReLU((-1,-1), b, w)
print("o =",o)
print("y =",output(b0,w0,o))


# In[175]:


# Q1.2

# draw all weights from N(0,1)
for i in range (20):
    b[i] = np.random.normal(0,1)
    w[i,0] = np.random.normal(0,1)
    w[i,1] = np.random.normal(0,1)
    w0[0,i] = np.random.normal(0,1)
b0 = np.random.normal(0,1)

# plot for all x1,x2 in [-5,5]
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
NUM_STEPS = 100
x1 = np.arange(-5, 5, 0.1)
x2 = np.arange(-5, 5, 0.1)
x1, x2 = np.meshgrid(x1, x2)
y = np.empty([NUM_STEPS,NUM_STEPS])
for i in range(NUM_STEPS):
    for j in range(NUM_STEPS):
        o = ReLU((x1[i,j],x2[i,j]), b, w)
        y[i,j] = output(b0,w0,o)
surface = ax.plot_surface(x1, x2, y, cmap='jet', alpha=0.7)


# In[205]:


def ReLU_4(x, b, w, num=4):
    o = np.zeros((1,num))
    for i in range(num):
        o[0,i] = max(b[i] + w[i,0]*x[0,0] + w[i,1]*x[0,1] + w[i,2]*x[0,2] + w[i,3]*x[0,3], 0)
    return o


# In[206]:


def output_4(b, w, o):
    y = b
    for i in range(4):
        term = w[0,i]
        term *= o[0,i]
        y += term
    return sigmoid(y)


# In[210]:


# Q1.3

# initialize all hidden layer weights to 0.5
b = np.zeros((5,1))
b.fill(0.5)
w = np.zeros((5,4))
w.fill(0.5)

# initialize all output layer weights to 1
b0 = 1
w0 = np.zeros((1,4))
w0.fill(1)

print("x = (1,1):")
print("layer 1:")
x = np.zeros((1,4))
x[0,0]=1
x[0,1]=1
o1 = ReLU_4(x, b, w)
print("o1 =", o1)
print("layer2:")
o2 = ReLU_4(o1, b, w)
print("o2 =", o2)
print("layer3:")
o3 = ReLU_4(o2, b, w)
print("o3 =", o3)
print("layer4:")
o4 = ReLU_4(o3, b, w)
print("o4 =", o4)
print("layer4:")
o5 = ReLU_4(o4, b, w)
print("o5 =", o5)
print("y =",output_4(b0,w0,o5))

print("x = (1,-1):")
print("layer 1:")
x = np.zeros((1,4))
x[0,0]=1
x[0,1]=-1
o1 = ReLU_4(x, b, w)
print("o1 =", o1)
print("layer2:")
o2 = ReLU_4(o1, b, w)
print("o2 =", o2)
print("layer3:")
o3 = ReLU_4(o2, b, w)
print("o3 =", o3)
print("layer4:")
o4 = ReLU_4(o3, b, w)
print("o4 =", o4)
print("layer4:")
o5 = ReLU_4(o4, b, w)
print("o5 =", o5)
print("y =",output_4(b0,w0,o5))

print("x = (-1,-1):")
print("layer 1:")
x = np.zeros((1,4))
x[0,0]=-1
x[0,1]=-1
o1 = ReLU_4(x, b, w)
print("o1 =", o1)
print("layer2:")
o2 = ReLU_4(o1, b, w)
print("o2 =", o2)
print("layer3:")
o3 = ReLU_4(o2, b, w)
print("o3 =", o3)
print("layer4:")
o4 = ReLU_4(o3, b, w)
print("o4 =", o4)
print("layer4:")
o5 = ReLU_4(o4, b, w)
print("o5 =", o5)
print("y =",output_4(b0,w0,o5))


# In[230]:


# Q1.2

# draw all weights from N(0,1)
for i in range (5):
    b[i] = np.random.normal(0,1)
    w[i,0] = np.random.normal(0,1)
    w[i,1] = np.random.normal(0,1)
    w[i,2] = np.random.normal(0,1)
    w[i,3] = np.random.normal(0,1)
    if i !=4:
        w0[0,i] = np.random.normal(0,1)
b0 = np.random.normal(0,1)
x = np.zeros((1,4))

# plot for all x1,x2 in [-5,5]
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
NUM_STEPS = 100
x1 = np.arange(-5, 5, .1)
x2 = np.arange(-5, 5, .1)
x1, x2 = np.meshgrid(x1, x2)
y = np.empty([NUM_STEPS,NUM_STEPS])
for i in range(NUM_STEPS):
    for j in range(NUM_STEPS):
        x[0,0]=x1[i,j]
        x[0,1]=x2[i,j]
        o1 = ReLU_4(x, b, w)
        o2 = ReLU_4(o1, b, w)
        o3 = ReLU_4(o2, b, w)
        o4 = ReLU_4(o3, b, w)
        o5 = ReLU_4(o4, b, w)
        y[i,j] = output_4(b0,w0,o5)
surface = ax.plot_surface(x1, x2, y, cmap='jet', alpha=0.7)


# In[9]:


def sig(x):
    return 1 / (1 + math.exp(-x))


# In[23]:


def network(w1,w2,w3,w4,w5,w6,w7,w8,w9,x1,x2):
    print("(weights)",w1,w2,w3,w4,w5,w6,w7,w8,w9,"(input)",x1,x2)
    uA = w1*1 + w2*x1 + w3*x2
    vA = max(uA,0)
    uB = w4*1 + w5*x1 + w6*x2
    vB = max(uB,0)
    uC = w7*1 + w8*vA + w9*vB
    vC = sig(uC)
    print(round(uA,5),round(vA,5),round(uB,5),round(vB,5),round(uC,5),round(vC,5))


# In[24]:


# Q2.1
network(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,-1)
network(1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,-0.2,1.7)
network(4,3,2,1,0,-1,-2,-3,-4,-4,1)
network(0.1,-0.2,0.3,-0.4,0.5,-0.6,0.7,-0.8,0.9,1,-1)


# In[39]:


def gradient(w1,w2,w3,w4,w5,w6,w7,w8,w9,x1,x2,y):
    print("(weights)",w1,w2,w3,w4,w5,w6,w7,w8,w9,"(input)",x1,x2,"(y)",y)
    uA = w1*1 + w2*x1 + w3*x2
    vA = max(uA,0)
    uB = w4*1 + w5*x1 + w6*x2
    vB = max(uB,0)
    uC = w7*1 + w8*vA + w9*vB
    vC = sig(uC)
    E = 0.5*(vC-y)**2
    dEdvC = vC-y
    dEduC = dEdvC*vC*(1-vC)
    print(round(E,5),round(dEdvC,5),round(dEduC,5))


# In[43]:


# Q2.2
gradient(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,-1,1)
gradient(1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,-0.2,1.7,0)
gradient(4,3,2,1,0,-1,-2,-3,-4,-4,1,0)
gradient(0.1,-0.2,0.3,-0.4,0.5,-0.6,0.7,-0.8,0.9,1,-1,1)


# In[46]:


def q23(w1,w2,w3,w4,w5,w6,w7,w8,w9,x1,x2,y):
    print("(weights)",w1,w2,w3,w4,w5,w6,w7,w8,w9,"(input)",x1,x2,"(y)",y)
    uA = w1*1 + w2*x1 + w3*x2
    vA = max(uA,0)
    uB = w4*1 + w5*x1 + w6*x2
    vB = max(uB,0)
    uC = w7*1 + w8*vA + w9*vB
    vC = sig(uC)
    E = 0.5*(vC-y)**2
    dEdvC = vC-y
    dEduC = dEdvC*vC*(1-vC)
    dEdvA = w8*dEduC
    if uA >= 0:
        dvAduA = 1
    else:
        dvAduA = 0
    dEduA = dEdvA*dvAduA
    dEdvB = w9*dEduC
    if uB >= 0:
        dvBduB = 1
    else:
        dvBduB = 0
    dEduB = dEdvB*dvBduB
    print(round(dEdvA,5),round(dEduA,5),round(dEdvB,5),round(dEduB,5))


# In[50]:


# Q2.3
q23(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,-1,1)
q23(1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,-0.2,1.7,0)
q23(4,3,2,1,0,-1,-2,-3,-4,-4,1,0)
q23(0.1,-0.2,0.3,-0.4,0.5,-0.6,0.7,-0.8,0.9,1,-1,1)


# In[55]:


def q24(w1,w2,w3,w4,w5,w6,w7,w8,w9,x1,x2,y):
    print("(weights)",w1,w2,w3,w4,w5,w6,w7,w8,w9,"(input)",x1,x2,"(y)",y)
    uA = w1*1 + w2*x1 + w3*x2
    vA = max(uA,0)
    uB = w4*1 + w5*x1 + w6*x2
    vB = max(uB,0)
    uC = w7*1 + w8*vA + w9*vB
    vC = sig(uC)
    E = 0.5*(vC-y)**2
    dEdvC = vC-y
    dEduC = dEdvC*vC*(1-vC)
    dEdvA = w8*dEduC
    if uA >= 0:
        dvAduA = 1
    else:
        dvAduA = 0
    dEduA = dEdvA*dvAduA
    dEdvB = w9*dEduC
    if uB >= 0:
        dvBduB = 1
    else:
        dvBduB = 0
    dEduB = dEdvB*dvBduB
    dEdw1 = 1*dEduA
    dEdw2 = x1*dEduA
    dEdw3 = x2*dEduA
    dEdw4 = 1*dEduB
    dEdw5 = x1*dEduB
    dEdw6 = x2*dEduB
    dEdw7 = 1*dEduC
    dEdw8 = vA*dEduC
    dEdw9 = vB*dEduC
    
    print(round(dEdw1,5),round(dEdw2,5),round(dEdw3,5),round(dEdw4,5),round(dEdw5,5),round(dEdw6,5),round(dEdw7,5),round(dEdw8,5),round(dEdw9,5))


# In[74]:


# Q2.4
q24(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,-1,1)
q24(1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,-0.2,1.7,0)
q24(4,3,2,1,0,-1,-2,-3,-4,-4,1,0)
q24(0.1,-0.2,0.3,-0.4,0.5,-0.6,0.7,-0.8,0.9,1,-1,1)


# In[75]:


def q25(w1,w2,w3,w4,w5,w6,w7,w8,w9,x1,x2,y,eta):
    print("(weights)",w1,w2,w3,w4,w5,w6,w7,w8,w9,"(input)",x1,x2,"(y)",y,"(eta)",eta)
    uA = w1*1 + w2*x1 + w3*x2
    vA = max(uA,0)
    uB = w4*1 + w5*x1 + w6*x2
    vB = max(uB,0)
    uC = w7*1 + w8*vA + w9*vB
    vC = sig(uC)
    E = 0.5*(vC-y)**2
    dEdvC = vC-y
    dEduC = dEdvC*vC*(1-vC)
    dEdvA = w8*dEduC
    if uA >= 0:
        dvAduA = 1
    else:
        dvAduA = 0
    dEduA = dEdvA*dvAduA
    dEdvB = w9*dEduC
    if uB >= 0:
        dvBduB = 1
    else:
        dvBduB = 0
    dEduB = dEdvB*dvBduB
    dEdw1 = 1*dEduA
    dEdw2 = x1*dEduA
    dEdw3 = x2*dEduA
    dEdw4 = 1*dEduB
    dEdw5 = x1*dEduB
    dEdw6 = x2*dEduB
    dEdw7 = 1*dEduC
    dEdw8 = vA*dEduC
    dEdw9 = vB*dEduC
    print(round(w1,5),round(w2,5),round(w3,5),round(w4,5),round(w5,5),round(w6,5),round(w7,5),round(w8,5),round(w9,5))
    print(round(E,5))
    w1 = w1 - eta*dEdw1
    w2 = w2 - eta*dEdw2
    w3 = w3 - eta*dEdw3
    w4 = w4 - eta*dEdw4
    w5 = w5 - eta*dEdw5
    w6 = w6 - eta*dEdw6
    w7 = w7 - eta*dEdw7
    w8 = w8 - eta*dEdw8
    w9 = w9 - eta*dEdw9
    uA = w1*1 + w2*x1 + w3*x2
    vA = max(uA,0)
    uB = w4*1 + w5*x1 + w6*x2
    vB = max(uB,0)
    uC = w7*1 + w8*vA + w9*vB
    vC = sig(uC)
    E = 0.5*(vC-y)**2
    print(round(w1,5),round(w2,5),round(w3,5),round(w4,5),round(w5,5),round(w6,5),round(w7,5),round(w8,5),round(w9,5))
    print(round(E,5))


# In[76]:


# Q2.5
q25(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,-1,1,0.1)
q25(1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,-0.2,1.7,0,0.1)
q25(4,3,2,1,0,-1,-2,-3,-4,-4,1,0,0.1)
q25(0.1,-0.2,0.3,-0.4,0.5,-0.6,0.7,-0.8,0.9,1,-1,1,0.1)


# In[90]:


def calc_error(w1,w2,w3,w4,w5,w6,w7,w8,w9,data):
    E = 0
    for i in range(760):
        x1 = data[i, 0]
        x2 = data[i, 1]
        y = data[i, 2]
        uA = w1*1 + w2*x1 + w3*x2
        vA = max(uA,0)
        uB = w4*1 + w5*x1 + w6*x2
        vB = max(uB,0)
        uC = w7*1 + w8*vA + w9*vB
        vC = sig(uC)
        E += 0.5*(vC-y)**2
    return E


# In[114]:


# Q2.6

data = np.loadtxt("data.txt")

w1 = 0.1
w2 = -0.2
w3 = 0.3
w4 = -0.4
w5 = 0.5
w6 = -0.6
w7 = 0.7
w8 = -0.8
w9 = 0.9
eta = 0.1

# iteration number zero

rand_index = random.randint(0,759)
x1 = data[rand_index, 0]
x2 = data[rand_index, 1]
y = data[rand_index, 2]
uA = w1*1 + w2*x1 + w3*x2
vA = max(uA,0)
uB = w4*1 + w5*x1 + w6*x2
vB = max(uB,0)
uC = w7*1 + w8*vA + w9*vB
vC = sig(uC)
dEdvC = vC-y
dEduC = dEdvC*vC*(1-vC)
dEdvA = w8*dEduC
if uA >= 0:
    dvAduA = 1
else:
    dvAduA = 0
dEduA = dEdvA*dvAduA
dEdvB = w9*dEduC
if uB >= 0:
    dvBduB = 1
else:
    dvBduB = 0
dEduB = dEdvB*dvBduB
dEdw1 = 1*dEduA
dEdw2 = x1*dEduA
dEdw3 = x2*dEduA
dEdw4 = 1*dEduB
dEdw5 = x1*dEduB
dEdw6 = x2*dEduB
dEdw7 = 1*dEduC
dEdw8 = vA*dEduC
dEdw9 = vB*dEduC

E = calc_error(w1,w2,w3,w4,w5,w6,w7,w8,w9,data)
# print("iteration 0 :","E =",E)
output = np.empty((0,2))
output = np.append(output, [[0,E]], axis=0)

for i in range(10000):
    rand_index = random.randint(0,759)
    x1 = data[rand_index, 0]
    x2 = data[rand_index, 1]
    y = data[rand_index, 2]
    uA = w1*1 + w2*x1 + w3*x2
    vA = max(uA,0)
    uB = w4*1 + w5*x1 + w6*x2
    vB = max(uB,0)
    uC = w7*1 + w8*vA + w9*vB
    vC = sig(uC)
    dEdvC = vC-y
    dEduC = dEdvC*vC*(1-vC)
    dEdvA = w8*dEduC
    if uA >= 0:
        dvAduA = 1
    else:
        dvAduA = 0
    dEduA = dEdvA*dvAduA
    dEdvB = w9*dEduC
    if uB >= 0:
        dvBduB = 1
    else:
        dvBduB = 0
    dEduB = dEdvB*dvBduB
    dEdw1 = 1*dEduA
    dEdw2 = x1*dEduA
    dEdw3 = x2*dEduA
    dEdw4 = 1*dEduB
    dEdw5 = x1*dEduB
    dEdw6 = x2*dEduB
    dEdw7 = 1*dEduC
    dEdw8 = vA*dEduC
    dEdw9 = vB*dEduC
    w1 = w1 - eta*dEdw1
    w2 = w2 - eta*dEdw2
    w3 = w3 - eta*dEdw3
    w4 = w4 - eta*dEdw4
    w5 = w5 - eta*dEdw5
    w6 = w6 - eta*dEdw6
    w7 = w7 - eta*dEdw7
    w8 = w8 - eta*dEdw8
    w9 = w9 - eta*dEdw9
    uA = w1*1 + w2*x1 + w3*x2
    vA = max(uA,0)
    uB = w4*1 + w5*x1 + w6*x2
    vB = max(uB,0)
    uC = w7*1 + w8*vA + w9*vB
    vC = sig(uC)
    if (i+1)%100 == 0:
        E = calc_error(w1,w2,w3,w4,w5,w6,w7,w8,w9,data)
        # print("iteration",i+1,":",E)
        output = np.append(output, [[i+1,E]], axis=0)

plt.plot(output[:,0],output[:,1])
plt.xlabel("round")
plt.ylabel("total error")


# In[ ]:




