
# coding: utf-8

# In[17]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io


# In[13]:


clown=io.imread('clown.gif')


# In[149]:


ch, cw=clown.shape
cexpandN = cv2.resize(clown,(cw*2,ch*2),interpolation=cv2.INTER_NEAREST)
ch2, cw2=cexpandN.shape
cexpandedN = cv2.resize(cexpandN,(cw2*2,ch2*2),interpolation=cv2.INTER_NEAREST)

cexpandC = cv2.resize(clown,(cw*2,ch*2),interpolation=cv2.INTER_CUBIC)
ch2, cw2C=cexpandC.shape
cexpandedC = cv2.resize(cexpandC,(cw2*2,ch2*2),interpolation=cv2.INTER_CUBIC)


# In[150]:


fig = plt.figure(figsize=(2,2))

plt.imshow(clown, cmap='gray')
plt.title("Original Clown",fontsize=20), plt.xticks([]), plt.yticks([])
plt.show()
#blowing up the clown 2x
fig = plt.figure(figsize=(8,8))
plt.subplot(121)
plt.imshow(cexpandN, cmap='gray')
plt.title("2X Clown, nearest",fontsize=20), plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(cexpandC, cmap='gray')
plt.title("2X Clown, cubic",fontsize=20), plt.xticks([]), plt.yticks([])
plt.show()
#blowing up the clown 4x
fig = plt.figure(figsize=(16,16))
plt.subplot(121)
plt.imshow(cexpandedN, cmap='gray')
plt.title("4X Clown, nearest",fontsize=20), plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(cexpandedC, cmap='gray')
plt.title("4X Clown, cubic",fontsize=20), plt.xticks([]), plt.yticks([])
plt.show()
plt.show()


# In[151]:


Paolina=io.imread('Paolina(2).tiff')


# In[152]:


ph, pw=Paolina.shape
pexpandN = cv2.resize(Paolina,(pw*2,ph*2),interpolation=cv2.INTER_NEAREST)
ph2, pw2=cexpandN.shape
pexpandedN = cv2.resize(pexpandN,(cw2*2,ch2*2),interpolation=cv2.INTER_NEAREST)

pexpandC = cv2.resize(Paolina,(pw*2,ph*2),interpolation=cv2.INTER_CUBIC)
ph2, pw2=pexpandC.shape
pexpandedC = cv2.resize(pexpandC,(pw2*2,ph2*2),interpolation=cv2.INTER_CUBIC)


# In[153]:


fig = plt.figure(figsize=(2,2))

plt.imshow(Paolina, cmap='gray')
plt.title("Original Paolina",fontsize=20), plt.xticks([]), plt.yticks([])
plt.show()
#blowing up the clown 2x
fig = plt.figure(figsize=(8,8))
plt.subplot(121)
plt.imshow(pexpandN, cmap='gray')
plt.title("2X Paolina, nearest",fontsize=20), plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(pexpandC, cmap='gray')
plt.title("2X Paolina, cubic",fontsize=20), plt.xticks([]), plt.yticks([])
plt.show()
#blowing up the clown 4x
fig = plt.figure(figsize=(16,16))
plt.subplot(121)
plt.imshow(pexpandedN, cmap='gray')
plt.title("4X Paolina, nearest",fontsize=20), plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(pexpandedC, cmap='gray')
plt.title("4X Paolina, cubic",fontsize=20), plt.xticks([]), plt.yticks([])
plt.show()


# In[4]:


from scipy import ndimage, misc


# In[112]:


sig1 = ndimage.gaussian_laplace(clown, sigma=1)
sig2 = ndimage.gaussian_laplace(clown, sigma=2)
sig3 = ndimage.gaussian_laplace(clown, sigma=3)


# In[141]:


matrix=[]
def detect(a):
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    matrix.append(signchange)
    return matrix


# In[142]:


for p in sig1:
    matrix=detect(p)
matrix=np.vstack(matrix)
A = np.asarray(matrix)
matrix=[]
for p in sig2:
    matrix=detect(p)
matrix=np.vstack(matrix)
B = np.asarray(matrix)
matrix=[]
for p in sig3:
    matrix=detect(p)
matrix=np.vstack(matrix)
C = np.asarray(matrix)


# In[147]:


fig = plt.figure(figsize=(10,20))
plt.subplot(321)
plt.imshow(sig1, cmap='gray')
plt.title("Sigma=1",fontsize=20), plt.xticks([]), plt.yticks([])
plt.subplot(322)
plt.imshow(A, cmap='gray')
plt.title("Zero Crossings Sigma 1",fontsize=20), plt.xticks([]), plt.yticks([])
plt.subplot(323)
plt.imshow(sig2, cmap='gray')
plt.title("Sigma=2",fontsize=20), plt.xticks([]), plt.yticks([])
plt.subplot(324)
plt.imshow(B, cmap='gray')
plt.title("Zero Crossings Sigma 2",fontsize=20), plt.xticks([]), plt.yticks([])
plt.subplot(325)
plt.imshow(sig3, cmap='gray')
plt.title("Sigma=2",fontsize=20), plt.xticks([]), plt.yticks([])
plt.subplot(326)
plt.imshow(C, cmap='gray')
plt.title("Zero Crossings Sigma 3",fontsize=20), plt.xticks([]), plt.yticks([])
plt.show()


# In[131]:




