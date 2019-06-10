from sklearn.manifold import TSNE
import cv2, os
import numpy as np
from utils import is_image_file
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

D = 3
SIZE = 256
images = []
Y = []

for root, _, fnames in sorted(os.walk('r0000')):
    for fname in fnames:
        if is_image_file(fname):
            inputimg = cv2.imread(os.path.join(root, fname), -1)
            input_image = cv2.resize(np.float32(inputimg), (SIZE, SIZE), cv2.INTER_CUBIC) / 255.0
            Y.append(0)
            images.append(input_image)

for root, _, fnames in sorted(os.walk('r0150')):
    for fname in fnames:
        if is_image_file(fname):
            inputimg = cv2.imread(os.path.join(root, fname), -1)
            input_image = cv2.resize(np.float32(inputimg), (SIZE, SIZE), cv2.INTER_CUBIC) / 255.0
            Y.append(1)
            images.append(input_image)

for root, _, fnames in sorted(os.walk('root_training_real_data/blended')):
    for fname in fnames:
        if is_image_file(fname):
            base = root.split('/')[-2]
            blended = cv2.imread(os.path.join(root, fname), -1)
            trans = cv2.imread(os.path.join(os.path.join(base, 'transmission_layer'), fname), -1)
            reflection = blended - trans
            input_image = cv2.resize(np.float32(reflection), (SIZE, SIZE), cv2.INTER_CUBIC) / 255.0
            name = fname.split('.')[0]
            if name.isdigit() and int(name) <= 103:
                Y.append(2)
            else:
                Y.append(3)
            images.append(input_image)

X = np.zeros((len(images), SIZE*SIZE*3))
for i in range(X.shape[0]):
    X[i] = images[i].reshape(-1)

if D == 3:
    tsne = TSNE(n_components=3, random_state=0)
    X_2d = tsne.fit_transform(X)

    # from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = 'b', 'r', 'y', 'orange', 'g', 'c', 'k', 'w', 'purple', 'm'
    for i in range(len(Y)):
        ax.scatter(X_2d[i, 0], X_2d[i, 1], X_2d[i, 2], c=colors[Y[i]])
else:
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)

    # from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 10))
    colors = 'b', 'r', 'orange', 'y', 'c', 'm', 'purple', 'k', 'w'
    labels = 'real_images_trainning', 'real_images_trainning_addtional', 'real_image_test', 'image_test'

    for i in range(len(Y)):
        plt.scatter(X_2d[i, 0], X_2d[i, 1], c=colors[Y[i]])
plt.show()

