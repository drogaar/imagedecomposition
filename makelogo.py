import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from itertools import permutations
from collections import namedtuple

img_shape = (200,100)

state = {'image' : Image.new('RGB', img_shape, color=(255,255,255)), 'points' : init_state()}
target = Image.open("drawing.png").convert('RGB')

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
    alpha = 50
    line_width = 5

    pairs = permutations(state['points'], 2)
    pairs = [pair for pair in pairs if not (pair[0][0] in pair[1] or pair[0][1] in pair[1]) ]

    def draw(pair):
        img = state['image'].copy()
        drw = ImageDraw.Draw(img, 'RGBA')
        drw.line(pair, fill = (0,0,0,alpha), width = line_width)
        return img

    results = [draw(pair) for pair in pairs]
    return results

def array(image):
    image = np.array(image, dtype=np.float32)
    return image / 255.

def compose_target():
    def score(target, result):
        diff = array(target) - array(result)
        return np.sum(np.where(diff<0, np.abs(diff), 1))

    for i in range(30):
        results = actions()
        differences = np.asarray([score(result, target) for result in results])
        # print(differences, np.argmin(differences), np.min(differences), i)
        print(i)
        state['image'] = results[np.argmin(differences)]
        # state['image'].show()

compose_target()
state['image'].show()

def score(target, result):
    diff = array(target) - array(result)
    return np.sum(np.where(diff<0, np.abs(diff), 1))#np.sum(np.abs(diff[diff<0]))*-1

results = actions()
list(results)[37].show()
list(results)[300].show()

score(target, results[37])
score(target, results[300])

diff = array(target) - array(results[37])
len(diff[diff > 0.1])
len(diff[diff < 0])
x = np.where(diff<0, 1, 1)
plt.imshow(x)

diff2 = array(target) - array(results[300])
len(diff2[diff2 > 0.1])
len(diff2[diff2 < 0])
plt.imshow(np.where(diff2<0, np.abs(diff2), 1))
