#!/usr/bin/env python
# coding: utf-8

# In[167]:


#PUNTO 2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io

imagen_original = plt.imread("imagen.jpg")
plt.imshow(imagen_original)
plt.show()


# In[168]:


#PUNTO 3
imagen = color.rgb2gray(imagen_original)
plt.imshow(imagen,cmap="gray")
plt.show()


# In[169]:


#PUNTO 4
print(imagen.shape)


# In[170]:


#PUNTO 5
ancho = imagen.shape[0]
largo = imagen.shape[1]

print(ancho)
print(largo)


# In[171]:


#PUNTO 6
print(imagen[100][290])
print(imagen[200][300])
print(imagen[300][300])


# In[172]:


#PUNTO 6.1
dx=np.zeros(shape=(ancho,largo))

for i in range(0,ancho-1):
    for j in range(0,largo-1):
        dx[i][j]= abs(imagen[i+1][j] - imagen[i][j])

plt.imshow(dx,cmap='gray')
plt.show()


# In[173]:


#PUNTO 7
dy=np.zeros(shape=(ancho,largo))

for i in range(0,ancho-1):
    for j in range(0,largo-1):
        dy[i][j]= abs(imagen[i][j+1] - imagen[i][j])

plt.imshow(dy,cmap='gray')
plt.show()


# In[174]:


#PUNTO 8
gradiente=np.zeros(shape=(ancho,largo))

for i in range(0,ancho):
    for j in range(0,largo):
        gradiente[i][j]= dx[i][j] + dy[i][j]

plt.imshow(gradiente,cmap='gray')
plt.show()


# In[175]:


#PUNTO 9
for i in range(0,ancho):
    for j in range(0,largo):
        gradiente[i][j]= 1 - gradiente[i][j]

plt.imshow(gradiente,cmap='gray')
plt.show()


# In[176]:


#Actividad 2. imagen final con bordes mas resaltados
#Se vuelve a derivar pero ahora en ambas 'x'
dx=np.zeros(shape=(ancho,largo))

for i in range(0,ancho-1):
    for j in range(0,largo-1):
        dx[i][j]= abs(imagen[i+1][j] - imagen[i-1][j])

plt.imshow(dx,cmap='gray')
plt.show()


# In[177]:


#Se vuelve a derivar pero ahora en ambas 'y'
dy=np.zeros(shape=(ancho,largo))

for i in range(0,ancho-1):
    for j in range(0,largo-1):
        dy[i][j]= abs(imagen[i][j+1] - imagen[i][j-1])

plt.imshow(dy,cmap='gray')
plt.show()


# In[178]:


#Se obtiene nuevamente el gradiente
gradiente=np.zeros(shape=(ancho,largo))

for i in range(0,ancho):
    for j in range(0,largo):
        gradiente[i][j]= dx[i][j] + dy[i][j]

plt.imshow(gradiente,cmap='gray')
plt.show()


# In[179]:


#Se invierten los colores
for i in range(0,ancho):
    for j in range(0,largo):
        gradiente[i][j]= 1 - gradiente[i][j]

plt.imshow(gradiente,cmap='gray')
plt.show()


# In[180]:


#Actividad 3. Detección de bordes siguiendo la dirección de la diagonal ↘
#Direccion diagonal ↘
direccion1=np.zeros(shape=(ancho,largo))
for i in range(0,ancho):
    for j in range(0,largo):
            direccion1[i][j] = math.cos(-1*math.pi/4)*(dx[i][j]) + math.sin(-1*math.pi/4)*(dy[i][j])


plt.imshow(direccion1,cmap='gray')
plt.show()


# In[181]:


#Actividad 4. Detección de bordes siguiendo la dirección de la diagonal ↙
#Direccion diagonal ↙
direccion2=np.zeros(shape=(ancho,largo))
for i in range(0,ancho):
    for j in range(0,largo):
            direccion2[i][j] = math.cos(-3*math.pi/4)*(dx[i][j]) + math.sin(-3*math.pi/4)*(dy[i][j])

plt.imshow(direccion2,cmap='gray')
plt.show()


# In[182]:


#Sumar las 2 diagonales
sumDiagonales=np.zeros(shape=(ancho,largo))

for i in range(0,ancho):
    for j in range(0,largo):
        sumDiagonales[i][j]= direccion1[i][j] + direccion2[i][j]

plt.imshow(sumDiagonales,cmap='gray')
plt.show()


# In[183]:


#Invirtiendo los colores
for i in range(0,ancho):
    for j in range(0,largo):
        sumDiagonales[i][j]= 1 - sumDiagonales[i][j]

plt.imshow(sumDiagonales,cmap='gray')
plt.show()


# In[ ]:




