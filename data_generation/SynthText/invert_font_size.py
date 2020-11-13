import pygame
from pygame import freetype
from text_utils import FontState
import numpy as np
import matplotlib.pyplot as plt
import pickle as cp


pygame.init()


ys = np.arange(8, 200)
A = np.c_[ys, np.ones_like(ys)]

xs = []
models = {}  # linear model

FS = FontState()
# plt.figure()
# plt.hold(True)
for i in range(len(FS.fonts)):
    print("i:" + str(i))
    font = freetype.Font(FS.fonts[i], size=12)
    print("font:" + str(font))
    h = []
    for y in ys:
        print(type(y))
        print("y:" + str(y))
        h.append(font.get_sized_glyph_height(int(y)))
        # h.append(font.get_sized_glyph_height(y))
    h = np.array(h)
    m, _, _, _ = np.linalg.lstsq(A, h)
    models[font.name] = m
    xs.append(h)
    print(font.name)

with open('font_px2pt.cp', 'wb') as f:
    cp.dump(models, f)
# plt.plot(xs,ys[i])
# plt.show()
