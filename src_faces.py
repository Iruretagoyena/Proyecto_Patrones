from sklearn.decomposition import SparseCoder
import numpy as np
from sklearn.metrics import accuracy_score

def open_image(image_path):
    image = cv2.imread(f'{DATASET_PATH}/{image_path}')
    resized = cv2.resize(image, (40, 40))
    return resized

TRAIN_NAMES = sorted([i for i in images_names if (i.endswith('01.jpg') or i.endswith('02.jpg') or i.endswith('03.jpg'))])
VALIDATION_NAMES = sorted([i for i in images_names if i.endswith('04.jpg')])
TEST_NAMES = sorted([i for i in images_names if (i.endswith('05.jpg') or i.endswith('06.jpg'))])

Y_TRAIN = np.array([int(image_name.split('_')[0][2:]) for image_name in TRAIN_NAMES]).astype(int)
Y_VAL = np.array([int(image_name.split('_')[0][2:]) for image_name in VALIDATION_NAMES]).astype(int)
Y_TEST = np.array([int(image_name.split('_')[0][2:]) for image_name in TEST_NAMES]).astype(int)

TRAIN_IMAGES = [open_image(image_path) for image_path in TRAIN_NAMES]
VAL_IMAGES = [open_image(image_path) for image_path in VALIDATION_NAMES]
TEST_IMAGES = [open_image(image_path) for image_path in TEST_NAMES]


flattened_train_imgs = []

for img in TRAIN_IMAGES:
    img = np.divide(img, 255)
    img = img.flatten()
    flattened_train_imgs.append(img)

np.save('Ytrain.npy', flattened_train_imgs)

flattened_d = []

for img in Y_TRAIN:
    flattened_d.append([img])

np.save('d.npy', flattened_d)

flattened_test_imgs = []

for img in TEST_IMAGES:
    img = np.divide(img, 255)
    img = img.flatten()
    flattened_test_imgs.append(img)
np.save('Ytest.npy', flattened_test_imgs)
dt = []

for img in Y_TEST:
    dt.append([img])

np.save('dt.npy', dt)


# Dataset: ORL 10 faces x 40 subjects (training first 9 images, testing last image)
Ytrain = np.load('Ytrain.npy')   # 9 faces of 40 subjects (size of face image = 16x16)
d      = np.load('d.npy')        # training labels (1, 2, ... 40)
Ytest  = np.load('Ytest.npy')    # 1 face  of 40 subjects (size of face image = 16x16)
dt     = np.load('dt.npy')       # testing labels (1, 2, ... 40)

d      = d-1                     # now labels are 0...39
dt     = dt-1                    # now labels are 0...39

Nt     = Ytest.shape[0]          # number of testing samples
dmax   = int(np.max(d))          # number of classes
T      = 5                       # number of non-zero coefficients

D      = Ytrain  # Dictionary
coder  = SparseCoder(dictionary=D, transform_n_nonzero_coefs=T,transform_algorithm='omp')

ds     = np.zeros((Nt,1))        # predcited labels
for i_t in range(Nt):
    print('... processing sample '+str(i_t))
    ytest = Ytest[i_t,:]
    xt    = coder.transform(ytest.reshape(1, -1))
    e     = np.ones((dmax,1))    # reconstruction error
    for i in range(dmax):
        xi       = xt.copy()
        ii       = np.argwhere(d!=i)
        xi[0,ii] = 0             # remoove coefficients that do not belong to class i
        e[i]     = np.linalg.norm(ytest - D.T.dot(xi.T))
    ds[i_t] = np.argmin(e)

acc = accuracy_score(ds,dt)
print(acc)