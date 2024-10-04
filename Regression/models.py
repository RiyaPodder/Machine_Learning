#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


# In[2]:


#Read SNP data
lines = []
with open('SNPs.txt') as f:
    lines = f.readlines()


# In[3]:


snp_matrix = [line.strip().split("   ") for line in lines]


# In[4]:


#Read gene expression data
lines = []
with open('expression.txt') as f:
    lines = f.readlines()


# In[5]:


expression_matrix = [line.strip().split(" ") for line in lines]


# In[6]:


Y = [[float(k[394])] for k in expression_matrix]


# In[7]:


Y_mean = np.mean(Y)


# In[8]:


Y_centered = [[y[0] - Y_mean] for y in Y]


# In[9]:


Y_arr = np.matrix(Y_centered)


# In[10]:


total_snps = len(snp_matrix[0])


# In[11]:


all_beta = []


# In[12]:


for s in range(total_snps):
    X = [[int(k[s])] for k in snp_matrix]
    X_mean = np.mean(X)
    X_centered = [[x[0] - X_mean] for x in X]
    X_arr = np.matrix(X_centered)
    X_t = X_arr.transpose()
    X_a = np.matmul(X_t,X_arr)
    X_b = np.matmul(X_t,Y_arr)
    beta = np.matmul(inv(X_a),X_b)
    all_beta.append(beta.tolist()[0])


# In[13]:


plot_X = [x for x in range(total_snps)]


# In[14]:


plot_Y = [b[0] for b in all_beta]


# In[15]:


print("SNP with greatest influence:")
m = max(np.absolute(plot_Y))
if m in plot_Y:
    print(plot_Y.index(m))
else:
    print(plot_Y.index(-m))


# In[16]:


f = plt.figure()
f.set_figwidth(30)
f.set_figheight(10)
plt.scatter(plot_X, plot_Y)
plt.xlabel("SNP Loci")
plt.ylabel("Beta regression parameter")
plt.show()


# In[17]:


f = plt.figure()
f.set_figwidth(30)
f.set_figheight(10)
plt.plot(plot_X, plot_Y)
plt.xlabel("SNP Loci")
plt.ylabel("Beta regression parameter")
plt.show()


# In[18]:


#Ridge regression 1


# In[19]:


iden = np.identity(1260)


# In[20]:


new_snp_matrix = []
for i in range(len(snp_matrix)):
    sample = snp_matrix[i]
    new_list = [int(sample[j]) for j in range(len(sample))]
    new_snp_matrix.append(new_list)


# In[21]:


for s in range(total_snps):
    X_new = [k[s] for k in new_snp_matrix]
    X_new_mean = np.mean(X_new)
    for sample in new_snp_matrix:
        sample[s] -= X_new_mean


# In[22]:


new_snp_matrix = np.matrix(new_snp_matrix)


# In[23]:


new_snp_matrix_t = new_snp_matrix.transpose()


# In[24]:


partA = np.matmul(new_snp_matrix_t,new_snp_matrix)


# In[25]:


sigma = 1/5


# In[26]:


iden = iden * sigma


# In[27]:


partB = np.add(partA,iden)


# In[28]:


partD = inv(partB)


# In[29]:


partC = np.matmul(new_snp_matrix_t, Y_arr)


# In[30]:


beta_ridge1 = np.matmul(partD, partC)


# In[31]:


plot_X = [x for x in range(total_snps)]
plot_Y = [b.tolist()[0][0] for b in beta_ridge1]


# In[32]:


f = plt.figure()
f.set_figwidth(30)
f.set_figheight(10)
plt.scatter(plot_X, plot_Y)

plt.xlabel("SNP Loci")
plt.ylabel("Beta regression parameter")
plt.show()


# In[33]:


f = plt.figure()
f.set_figwidth(30)
f.set_figheight(10)
plt.plot(plot_X, plot_Y)

plt.xlabel("SNP Loci")
plt.ylabel("Beta regression parameter")
plt.show()


# In[34]:


print("SNP with greatest influence:")
m = max(np.absolute(plot_Y))
if m in plot_Y:
    print(plot_Y.index(m))
else:
    print(plot_Y.index(-m))


# In[35]:


#Ridge Regression 2


# In[36]:


iden = np.identity(1260)
new_snp_matrix = []
for i in range(len(snp_matrix)):
    sample = snp_matrix[i]
    new_list = [int(sample[j]) for j in range(len(sample))]
    new_snp_matrix.append(new_list)

for s in range(total_snps):
    X_new = [k[s] for k in new_snp_matrix]
    X_new_mean = np.mean(X_new)
    for sample in new_snp_matrix:
        sample[s] -= X_new_mean

new_snp_matrix = np.matrix(new_snp_matrix)
new_snp_matrix_t = new_snp_matrix.transpose()
partA = np.matmul(new_snp_matrix_t,new_snp_matrix)
sigma = 1/0.005
iden = iden * sigma
partB = np.add(partA,iden)
partD = inv(partB)
partC = np.matmul(new_snp_matrix_t, Y_arr)
beta_ridge2 = np.matmul(partD, partC)
plot_X = [x for x in range(total_snps)]
plot_Y = [b.tolist()[0][0] for b in beta_ridge2]


# In[37]:


f = plt.figure()
f.set_figwidth(30)
f.set_figheight(10)
plt.scatter(plot_X, plot_Y)

plt.xlabel("SNP Loci")
plt.ylabel("Beta regression parameter")
plt.show()


# In[38]:


f = plt.figure()
f.set_figwidth(30)
f.set_figheight(10)
plt.plot(plot_X, plot_Y)

plt.xlabel("SNP Loci")
plt.ylabel("Beta regression parameter")
plt.show()


# In[39]:


print("SNP with greatest influence:")
m = max(np.absolute(plot_Y))
if m in plot_Y:
    print(plot_Y.index(m))
else:
    print(plot_Y.index(-m))


# In[ ]:




