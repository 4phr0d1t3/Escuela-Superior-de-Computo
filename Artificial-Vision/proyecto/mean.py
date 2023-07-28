#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd

path = 'datos/regionprops/Cantaloupe.csv'
canta = pd.read_csv(path)
canta['Fruta'] = 'Cantaloupe'

canta


# In[16]:


path = 'datos/regionprops/Granadilla.csv'
grana = pd.read_csv(path)
grana['Fruta'] = 'Granadilla'

grana


# In[17]:


path = 'datos/regionprops/Mango.csv'
mango = pd.read_csv(path)
mango['Fruta'] = 'Mango'

mango


# In[18]:


path = 'datos/regionprops/Raspberry.csv'
rasp = pd.read_csv(path)
rasp['Fruta'] = 'Raspberry'

rasp


# In[19]:


path = 'datos/regionprops/Strawberry.csv'
straw = pd.read_csv(path)
straw['Fruta'] = 'Strawberry'

straw


# In[20]:


df = pd.concat([canta, grana, mango, rasp, straw], axis=0)
df = df.reset_index()
df.to_csv('fruitsRP.csv', index=False)


# In[21]:


import pandas as pd

path = 'fruitsRP.csv'
df = pd.read_csv(path)

promedio_por_fruta = df.groupby('Fruta')[['Area', 'Perimetro', 'Metrica']].mean()
promedio_por_fruta


# In[22]:


import pandas as pd

path = 'datos/rgb/Cantaloupe.csv'
canta = pd.read_csv(path)
canta['Fruta'] = 'Cantaloupe'

canta


# In[23]:


path = 'datos/rgb/Granadilla.csv'
grana = pd.read_csv(path)
grana['Fruta'] = 'Granadilla'

grana


# In[24]:


path = 'datos/rgb/Mango.csv'
mango = pd.read_csv(path)
mango['Fruta'] = 'Mango'

mango


# In[25]:


path = 'datos/rgb/Raspberry.csv'
rasp = pd.read_csv(path)
rasp['Fruta'] = 'Raspberry'

rasp


# In[26]:


path = 'datos/rgb/Strawberry.csv'
straw = pd.read_csv(path)
straw['Fruta'] = 'Strawberry'

straw


# In[27]:


df = pd.concat([canta, grana, mango, rasp, straw], axis=0)
df = df.reset_index()
df.to_csv('fruitsRGB.csv', index=False)


# In[28]:


import pandas as pd

path = 'fruitsRGB.csv'
df = pd.read_csv(path)

promedio_por_fruta = df.groupby('Fruta')[['R', 'G', 'B']].mean()
promedio_por_fruta

