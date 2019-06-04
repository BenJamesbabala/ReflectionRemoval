from sklearn.manifold import TSNE
import cv2, os
import numpy as np
from utils import is_image_file
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

SIZE = 256
images = []
Y = []

for root, _, fnames in sorted(os.walk('images')):
    for fname in fnames:
        if is_image_file(fname):
            inputimg = cv2.imread(os.path.join(root, fname), -1)
            input_image = cv2.resize(np.float32(inputimg), (SIZE, SIZE), cv2.INTER_CUBIC) / 255.0
            if root.split('/')[-1] == 'faces':
                Y.append(0)
            else:
                Y.append(1)

            images.append(input_image)

X = np.zeros((len(images), SIZE*SIZE*3))
for i in range(X.shape[0]):
    X[i] = images[i].reshape(-1)

tsne = TSNE(n_components=3, random_state=0)
X_2d = tsne.fit_transform(X)

# from matplotlib import pyplot as plt
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection='3d')
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
labels = 'faces','sky'
for i in range(len(Y)):
    ax.scatter(X_2d[i, 0], X_2d[i, 1], X_2d[i, 2], c=colors[Y[i]], label= labels[Y[i]])
ax.legend()
plt.show()

