import os
import sys
import time
import json
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from visualize import visualize_nodes
from operator import itemgetter

def process():

	def get_bbox_coords(points): #(x,y, weight,height)

		min_x = points[0] #min(points, key=lambda x: x[0])[0] - 5
		max_x = points[0] + points[2] #max(points, key=lambda x: x[0])[0] + 5
		min_y = points[1] #min(points, key=lambda x: x[1])[1] - 5
		max_y = points[1] + points[3] #max(points, key=lambda x: x[1])[1] + 5
		return min_x, max_x, min_y, max_y
	
	def get_bbox_coords1(points): #(x,y, weight,height)
		min_x = min(points, key=lambda x: x[0])[0] - 5
		max_x = max(points, key=lambda x: x[0])[0] + 5
		min_y = min(points, key=lambda x: x[1])[1] - 5
		max_y = max(points, key=lambda x: x[1])[1] + 5
		return min_x, max_x, min_y, max_y
	def get_area(points):
		min_x, max_x, min_y, max_y = get_bbox_coords(points)
		return (max_x - min_x)*(max_y - min_y)

	def get_bbox(image, points):
		min_x, max_x, min_y, max_y = get_bbox_coords1(points)
		bbox = np.array(image.crop((min_x, min_y, max_x, max_y)))
		return bbox, min_x, min_y

	def get_car_bboxes(image, labels):
		bboxes, polygon_points = [], []
		#print(labels[0]['components'])
		for i in range(len(labels)):
			for item in labels[i]['components']:
				area = get_area(item['bbox'])
				if area > 10000 and labels[i]['label']=='car':
					box, min_x, min_y = get_bbox(image, item['poly'])
					adjusted_points = [[p[0] - min_x, p[1] - min_y] for p in item['poly']]
					bboxes.append(box)
					polygon_points.append(np.array(adjusted_points))
		return bboxes, polygon_points

	# Get train images
	raw_images_dir = './dataset/raw_images/train/aachen/'
	train_image_paths = [img for img in sorted(os.listdir(raw_images_dir))\
		if os.path.isfile(os.path.join(raw_images_dir, img))][:20]
	train_images = [raw_images_dir + path for path in train_image_paths]

	# Get train polygons
	annotated_images_dir = './dataset/annotated_images/train/aachen/'
	train_label_paths = [file for file in sorted(os.listdir(annotated_images_dir))][:20]#\
		#if 'polygons.json' in file][:20]
	train_labels = []
	for label_file in train_label_paths:
		with open(annotated_images_dir + label_file, 'r') as f:
			train_labels.append(json.load(f))
	#print(train_labels)
	# Get bounding boxes and corresponding polygon labels
	bboxes, polygon_labels = [], []
	for i, image_path in enumerate(train_images):
		image = Image.open(image_path)
		
		bboxes_image, points_image = get_car_bboxes(image, train_labels[i])
	
		bboxes.extend(bboxes_image)
		polygon_labels.extend(points_image)
	return bboxes, polygon_labels
