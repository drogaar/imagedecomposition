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
    alpha = 25
    line_width = 3

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

target = array(target)
def compose_target():
    def score(target, result):
        diff = array() - array(result)
        # return np.sum(np.where(diff<0, np.abs(diff), 0))
        return np.sum(np.abs(diff[diff<0]))#faster

    for i in range(1000):
        results = actions()

        differences = np.asarray([score(target, result) for result in results])
        # print(differences, np.argmin(differences), np.min(differences), i)
        print(np.argmin(differences), i)
        state['image'] = results[np.argmin(differences)]
        if i % 25 == 0:
            state['image'].show()

compose_target()
state['image'].show()

def score(target, result):
    diff = array(target) - array(result)
    # return np.sum(np.where(diff<0, np.abs(diff), 0))#np.sum(np.abs(diff[diff<0]))*-1
    return np.sum(np.abs(diff[diff<0]))#faster

results = actions()
list(results)[37].show()
list(results)[300].show()
list(results)[295].show()

score(target, results[37])
score(target, results[300])
score(target, results[295])

def phist(arr):
    diff = array(target) - array(arr)
    select = np.where(diff<0, diff, 1)
    plt.hist(np.mean(select, axis=-1))

phist(results[300])
phist(results[295])

def test(arr):
    diff = array(target) - array(arr)
    plt.figure(figsize=(12,6))
    # plt.subplot(1,2,1)
    # plt.hist(diff[diff<0])
    # plt.subplot(1,2,2)
    diff = np.where(diff<0, abs(diff), 0)
    amin, amax = (np.amin(diff), np.amax(diff))
    normal = (diff - amin)/(amax-amin)
    min, max = (np.amin(normal), np.amax(normal))
    sample = plt.imshow(normal, cmap='gray')
    print(amin, amax, min, max)

test(results[300])
test(results[295])
diff = array(target) - array(results[300])
plt.imshow(np.where(diff>0, np.abs(diff), 1))
plt.imshow(np.where(diff<0, np.abs(diff), 1))
plt.imshow(array(results[300]))
plt.imshow(array(target))
plt.imshow(array(target) - array(results[300]))
test(results[300])
test(results[295])

diff = array(target) - array(results[37])
len(diff[diff > 0.1])
len(diff[diff < 0])
x = np.where(diff<0, 1, 1)
plt.imshow(x)

diff2 = array(target) - array(results[300])
len(diff2[diff2 > 0.1])
len(diff2[diff2 < 0])
plt.imshow(np.where(diff2<0, np.abs(diff2), 1))
