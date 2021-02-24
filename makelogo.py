import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from itertools import permutations
from collections import namedtuple

img_shape = (200,100)

State = namedtuple('State', ['image', 'points'])
st8 = State(image=Image.new('RGB', img_shape, color=(255,255,255)), points = init_state())

def init_state():
    step = 10

    pointCoordinates = []
    for a in np.arange(0, img_shape[0], step):
        if a in img_shape: continue
        pointCoordinates.append((a, 0))
        pointCoordinates.append((a, img_shape[1]-1))
    for a in np.arange(0, img_shape[1], step):
        if a in img_shape: continue
        pointCoordinates.append((0, a))
        pointCoordinates.append((img_shape[0]-1, a))

    return pointCoordinates

def actions():
    alpha = 20
    line_width = 3

    pairs = permutations(st8['points'], 2)

    def draw(pair):
        img = st8['image'].copy()
        drw = ImageDraw.Draw(img, 'RGBA')
        drw.line(pair, fill = (0,0,0,alpha), width = line_width)
        return img

    actions = [lambda : draw(pair) for pair in pairs]
    return actions
