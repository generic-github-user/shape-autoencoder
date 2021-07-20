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

show(rectangle([64,64]), figure={'figsize':[5,5]})


# In[1792]:


def shape(dims=[], res=64):
    dims = U(dims, [res, res, 3])
    canvas = np.full(dims, 255, dtype=int)
    pos = np.random.randint([0, 0], canvas.shape[:2], [2])
    grid = np.stack(np.meshgrid(*[np.arange(0, dim, 1) for dim in canvas.shape[:2]]), axis=2)
    color = np.random.randint(100, 230, [3], dtype=int)
    canvas[np.linalg.norm(pos - grid, axis=2, ord=np.random.randint(1, 5)) < 10] = color
    meta = {
        'pos': pos,
        'color': color
    }
    return canvas, meta

show(shape()[0])


# In[1698]:


def symbol(c=None, res=64, dims=[], font=None):
#     typehandler = {
#         'int'
#     }
    if c is None:
        cn = np.random.randint(0, 1000)
        c = chr(cn)
#         print(c,cn)
    dims = U(dims, [res, res, 3])
    canvas = np.full(dims, 255, dtype=int)
    fonts = ['arial', 'calibri', 'bahnschrift']
    max_size = min(dims[:2])//2
    size = np.random.randint(max_size//4, max_size)
    if not font:
        font = ImageFont.truetype(random.choice(fonts), size)
    color = np.random.randint(100, 230, [3], dtype=int)
    img = Image.fromarray(canvas.astype(np.uint8))
#     np.array(dims)//4
    pos = np.random.randint(0, np.array(dims)[:2]-size)
    ImageDraw.Draw(img).text(tuple(pos), c, tuple(color), font=font)
    meta = {
        'c': c
    }
    return np.array(img), meta

show(symbol()[0])


# In[1489]:


chr(14)


# In[1786]:


def gen_data(F=symbol, N=100):
    return zip(*[F(res=128) for i in range(N)])
data, metadata = gen_data()
data = np.array(data)/255
print(data.dtype, data.shape)
show(data[0])
# todo: record creation parameters


# In[1746]:


data.nbytes


# In[1704]:


metadata[5]


# In[1787]:


tf.keras.backend.clear_session()
def dense_layers(sizes, A):
    return [tf.keras.layers.Dense(s, activation=A) for s in sizes]
Layers = tf.keras.layers
activation = 'sigmoid'
stride = (1, 1)
batchnorm = {'center': 0.5}
resolution = 128
R = resolution
encoder_layers = [
    Layers.InputLayer(input_shape=(R, R, 3)),
    Layers.Lambda(lambda I: tf.image.resize(I, (R//2, R//2), method='bilinear')),
    
    Layers.Conv2D(8, 3, strides=stride, padding='same', activation=activation, data_format='channels_last'),
    Layers.BatchNormalization(**batchnorm),
    
    Layers.Conv2D(16, 3, strides=stride, padding='same', activation=activation, data_format='channels_last'),
    Layers.BatchNormalization(**batchnorm),
    
    Layers.Conv2D(32, 3, strides=stride, padding='same', activation=activation, data_format='channels_last'),
    Layers.MaxPooling2D((2, 2)),
    Layers.Conv2D(32, 3, strides=stride, padding='same', activation=activation, data_format='channels_last'),
#     very helpful at getting past the 'averaging' local minimum (where we get a white or light gray image that essentially averages the training data)
    Layers.BatchNormalization(**batchnorm),
    Layers.MaxPooling2D((2, 2)),
    
    Layers.Conv2D(32, 3, strides=stride, padding='same', activation=activation, data_format='channels_last'),
    Layers.BatchNormalization(**batchnorm),
    Layers.MaxPooling2D((2, 2)),
    
    Layers.Conv2D(64, 3, strides=stride, padding='same', activation=activation, data_format='channels_last'),
    Layers.BatchNormalization(**batchnorm),
    Layers.MaxPooling2D((2, 2)),
    
#     tf.keras.preprocessing.image.smart_resize((20,20)),
#     tf.keras.preprocessing.image.apply_affine_transform()
#     tf.image
    Layers.Flatten(),
    *dense_layers([100, 50, 7], activation),
#     Layers.Dense(400, activation=activation),
#     Layers.Dense(200, activation=activation),
#     Layers.Dense(3, activation=activation),
]
