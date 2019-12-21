import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from random import shuffle
from random import seed
from bokeh import palettes

plt.style.use('seaborn')
numPoints = 20000
countMax = 300
classes = 5
seed(1000)

path = "data/srVals3.pkl"
f = open(path, "rb")
srMat, states = pickle.load(f)
f.close()


print("Kmeans")
kmeans = KMeans(classes, random_state=2).fit(srMat)
centerSR = kmeans.cluster_centers_
ptClass = kmeans.labels_

print("Getting state closest to SR")
centerCoord = []
#  for cent in centerSR:
    #  minVal = float("inf")
    #  closestInd = 0
    #  for ind, x in enumerate(srMat):
        #  curVal = np.sum(np.square(np.array(cent) - np.array(x)))
        #  if curVal < minVal:
            #  minVal = curVal
            #  closestInd = ind
    #  centerCoord.append(states[closestInd][0:3])
#  print(centerCoord)

print("Getting points")
pts = []
for x in states:
    pts.append(x[0:3])

c = list(zip(pts, ptClass))
pts, ptClass = zip(*c)
pts = np.array(pts)

pts = pts[0:numPoints]
ptClass = ptClass[0:numPoints]

fig = plt.figure()
pal = palettes.Viridis[6]
#  pal = palettes.Spectral[4]
#  pal = palettes.Plasma[4]
ax = fig.add_subplot(111, projection='3d')
for classVal in range(0, classes):
    count = 0
    for i in range(len(pts)):
        if count >= countMax:
            break
        if classVal == ptClass[i]:
            count += 1
            ax.scatter(pts[i, 0], pts[i, 1], pts[i, 2],
                       c=pal[classVal], marker='o')

ax.set_xlabel('X co-ordinate')
ax.set_ylabel('Y co-ordinate')
ax.set_zlabel('Z co-ordinate')
plt.show()


path = "data/srCenter.pkl"
f = open(path, "wb")
pickle.dump(centerSR, f)
f.close()
