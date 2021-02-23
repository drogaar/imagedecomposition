import numpy as np
import matplotlib.pyplot as plt

img_shape = (200,100)

def init():
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

def vectors():
    


img = np.zeros(img_shape + (3,), np.uint8)
cv.line(img,(0,0),(511,511),(255,0,0),5)
