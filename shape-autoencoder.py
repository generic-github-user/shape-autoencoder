#!/usr/bin/env python
# coding: utf-8

# In[1373]:


import tensorflow as tf
import numpy as np
import nevergrad as ng
import matplotlib.pyplot as plt
import matplotlib
import random
import PIL
from PIL import Image, ImageFont, ImageDraw


# In[871]:


from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


# In[174]:


def U(a, b, c=None):
    if c is None:
        c = len(b)
    a = a + ([None] * (c - len(a)))
    return [ai if ai else b[i] for i, ai in enumerate(a)]


# In[363]:


# plt.style.available


# In[1291]:


def show(im=None, data=[], plot_func='imshow', style='fivethirtyeight', axis='off', figure={}, **kwargs):
    plt.close('all')
    plt.style.use(style)
    fig = plt.figure(**figure)
    ax = fig.add_subplot()
#     ax.imshow(im)
    if im is not None:
        data = [im]+data
    getattr(ax, plot_func)(*data, **kwargs)
    plt.axis(axis)
    return ax


# In[1292]:


def rectangle(dims=[None]*3):
    dims = U(dims, [64, 64, 3])
    bg = 255
    rect = np.full(dims, bg, dtype=int)
    size = 20
    x = random.randint(0, dims[0]-size)
    y = random.randint(0, dims[1]-size)
    w = random.randint(5, size)
    h = random.randint(5, size)
    rect[x:x+w, y:y+h] = np.random.randint(100, 230, [3], dtype=int)
    return rect
