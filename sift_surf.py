# Tarda alrededor de 1 minuto en ser ejecutado

import cv2
#pip3 install opencv-python==3.3.0.10 opencv-contrib-python==3.3.0.10
import os
import numpy as np
from operator import itemgetter
from sklearn.metrics import accuracy_score, confusion_matrix

DATASET_PATH = 'FaceMask166'
FEATURES_PATH = 'features'
NUM_OF_CLASSES = 166
VERTICAL_CUT = 45
HORIZONTAL_CUT = 105

def open_image(image_path):
    image = cv2.imread(f'{DATASET_PATH}/{image_path}')
    return image

images_names = sorted([file_name for file_name in os.listdir(DATASET_PATH) if file_name.endswith('jpg')])
images_names = images_names[:(NUM_OF_CLASSES * 6)]

TRAIN_NAMES = sorted([i for i in images_names if (i.endswith('01.jpg') or i.endswith('02.jpg') or i.endswith('03.jpg'))])
VALIDATION_NAMES = sorted([i for i in images_names if i.endswith('04.jpg')])
TEST_NAMES = sorted([i for i in images_names if (i.endswith('05.jpg') or i.endswith('06.jpg'))])

Y_TRAIN = np.array([int(image_name.split('_')[0][2:]) for image_name in TRAIN_NAMES]).astype(int)
Y_VAL = np.array([int(image_name.split('_')[0][2:]) for image_name in VALIDATION_NAMES]).astype(int)
Y_TEST = np.array([int(image_name.split('_')[0][2:]) for image_name in TEST_NAMES]).astype(int)


TRAIN_IMAGES = [open_image(image_path)[:HORIZONTAL_CUT,VERTICAL_CUT:-VERTICAL_CUT] for image_path in TRAIN_NAMES]
VAL_IMAGES = [open_image(image_path)[:HORIZONTAL_CUT,VERTICAL_CUT:-VERTICAL_CUT] for image_path in VALIDATION_NAMES]
TEST_IMAGES = [open_image(image_path)[:HORIZONTAL_CUT,VERTICAL_CUT:-VERTICAL_CUT] for image_path in TEST_NAMES]

def build_dataset(n_clases):
    train_limit = n_clases * 3
    val_limit = n_clases
    test_limit = n_clases * 2

    dataset = {}
    dataset['n_clases'] = n_clases
    dataset['train_limit'] = train_limit
    dataset['val_limit'] = val_limit
    dataset['test_limit'] = test_limit
    dataset['y_train'] = Y_TRAIN[:train_limit]
    dataset['y_val'] = Y_VAL[:val_limit]
    dataset['y_test'] = Y_TEST[:test_limit]
    dataset['train_images'] = TRAIN_IMAGES[:train_limit]
    dataset['val_images'] = VAL_IMAGES[:val_limit]
    dataset['test_images'] = TEST_IMAGES[:test_limit]
    dataset['x_train'] = []
    dataset['x_val'] = []
    dataset['x_test'] = []

    return dataset

dataset = build_dataset(166)



def get_descriptors(images):
    features = []
    # pip3 install opencv-python==3.3.0.10 opencv-contrib-python==3.3.0.10
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create(upright=1)
    for img in images:
        keypoints, descriptors = surf.detectAndCompute(img, None)
        img = cv2.drawKeypoints(img, keypoints, None)
        """ cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() """
        new_feats = {"image": img, "descriptors": descriptors, "keypoints": keypoints }
        features.append(new_feats)
    return features

TRAIN_DESCRIPTORS = get_descriptors(dataset['train_images'])
TEST_DESCRIPTORS = get_descriptors(dataset['test_images'])
VAL_DESCRIPTORS = get_descriptors(dataset['val_images'])


def show_match(IMG_1, IMG_2, matches):
    img3 = cv2.drawMatches(
    IMG_1["image"],
    IMG_1["keypoints"],
    IMG_2["image"],
    IMG_2["keypoints"],
    matches,
    IMG_2["image"],
    singlePointColor=123,
    flags=2
    )
    cv2.imshow("Image", img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_distances(IMG_1):
    distances = []
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    for IMG_2 in TRAIN_DESCRIPTORS:
        try:
            N_MATCHES = 5
            matches = bf.match(IMG_1["descriptors"], IMG_2["descriptors"])
            matches = sorted(matches, key = lambda x:x.distance)
            matches = matches[:N_MATCHES]
            des_sum = sum(list(map(lambda x:x.distance, matches)))
            distances.append(des_sum)
            #show_match(IMG_1, IMG_2, matches)
        except:
            distances.append(float('Inf'))
    return distances

def get_min_distance_index(distances):
    min_distance_index = min(enumerate(distances), key=itemgetter(1))[0]
    return min_distance_index

def get_class_by_matches(y_train, IMG_1):
    distances = get_distances(IMG_1)
    min_dist_index = get_min_distance_index(distances)
    return y_train[min_dist_index]


train_pred = [get_class_by_matches(dataset['y_train'], IMG) for IMG in TRAIN_DESCRIPTORS]
val_pred = [get_class_by_matches(dataset['y_train'], IMG) for IMG in VAL_DESCRIPTORS]
test_pred = [get_class_by_matches(dataset['y_train'], IMG) for IMG in TEST_DESCRIPTORS]


train_accuracy = accuracy_score(dataset['y_train'], train_pred) * 100
val_accuracy = accuracy_score(dataset['y_val'], val_pred) * 100
test_accuracy = accuracy_score(dataset['y_test'], test_pred) * 100

conf_matrix = confusion_matrix(dataset['y_val'], val_pred)

print(f'\nTrain Accuracy: {train_accuracy}')
print(f'Val Accuracy: {val_accuracy}\n')
print(f'Test Accuracy: {test_accuracy}\n')

print(conf_matrix)