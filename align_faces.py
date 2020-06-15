# USAGE
# python align_faces.py

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2
import os

SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
DATASET_PATH = 'dataset'
PROCESS_PATH = 'unaligned'

images = [i for i in os.listdir('dataset/unaligned') if i.endswith('jpg')]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
fa = FaceAligner(predictor, desiredFaceWidth = 256)

for image_path in images:
	print(f'Processing image: {image_path}...')
	image = cv2.imread(f'{DATASET_PATH}/{PROCESS_PATH}/{image_path}')
	image = imutils.resize(image, width = 256)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 2)

	for rect in rects:
		(x, y, w, h) = rect_to_bb(rect)
		faceAligned = fa.align(image, gray, rect)
		# Para comprobar que esté bien:
		# faceAligned[90][165] = [255, 255, 255]
		# faceAligned[90][90] = [255, 255, 255]
		cv2.imwrite(f'{DATASET_PATH}/{image_path}', faceAligned)
