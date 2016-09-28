from __future__ import division
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import sys

NETS = {'vgg16': ('VGG16',
				  'pre-trained-models/map_words_faster_rcnn.caffemodel')}

def vis_detections(im, title, dets, thresh):
	# im = im[:, :, (2, 1, 0)]
	for i in xrange(dets.shape[0]):
		bbox = dets[i, :4]
		score = dets[i, -1]
		if score > thresh:
			cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

		
	cv2.imshow(title, im)
	cv2.waitKey(0)

def save_detections(im, im_name, dets, thresh):
	for i in xrange(dets.shape[0]):
		bbox = dets[i, :4]
		score = dets[i, -1]
		if score > thresh:
			cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 4)
	cv2.imwrite(im_name, im)

def im_detect_sliding_crop(net, im, crop_h, crop_w, step):
	imh, imw, _ = im.shape

	cls_ind = 1

	boxes = np.zeros((0, 4), dtype=np.float32)
	scores = np.zeros((0, 1), dtype=np.float32)

	y1 = 0
	while y1 < imh:
		y2 = min(y1 + crop_h, imh)
		if y2 - y1 < 25:
			y1 += step
			continue

		x1 = 0
		while x1 < imw:			
			x2 = min(x1 + crop_w, imw)
			if x2 - x1 < 25:
				x1 += step
				continue

			crop_im = im[y1:y2, x1:x2, :]

			# # check
			# cv2.imshow("im", crop_im)
			# cv2.waitKey(0)
			# print crop_im.shape
			crop_scores, crop_boxes = im_detect(net, crop_im)
			crop_boxes = crop_boxes[:, 4*cls_ind:4*(cls_ind + 1)]
			crop_scores = crop_scores[:, cls_ind] + (0.01 * np.random.random() - 0.005)

			# vis_detections(crop_im, 'crop image', 
			# 			   np.hstack((crop_boxes, 
			# 			   			  crop_scores[:, np.newaxis])), 0.25)

			crop_boxes[:,0] += x1
			crop_boxes[:,1] += y1
			crop_boxes[:,2] += x1
			crop_boxes[:,3] += y1

			boxes = np.vstack((boxes, crop_boxes))
			scores = np.vstack((scores, crop_scores[:, np.newaxis]))

			# # print crop_boxes.shape, crop_scores.shape, boxes.shape, scores.shape
			# keep_idx = np.where(crop_scores > 0.1)
			# print len(keep_idx[0])

			# keep_idx = np.where(scores > 0.1)
			# print len(keep_idx[0])

			# vis_detections(im, 'entire image',
			# 			   np.hstack((boxes, scores)), 0.25)

			x1 += step

		y1 += step

	return scores, boxes

def parse_args():
	"""Parse input arguments."""
	parser = argparse.ArgumentParser(description='Face Detection using Faster R-CNN')
	parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
						default=0, type=int)
	parser.add_argument('--cpu', dest='cpu_mode',
						help='Use CPU mode (overrides --gpu)',
						action='store_true')
	parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
						choices=NETS.keys(), default='vgg16')

	args = parser.parse_args()

	return args

if __name__ == '__main__':
	cfg.TEST.HAS_RPN = True  # Use RPN for proposals
	# cfg.TEST.BBOX_REG = False

	args = parse_args()

	prototxt = 'models/map/VGG16/faster_rcnn_end2end/test.prototxt'
	caffemodel = NETS[args.demo_net][1]

	if not os.path.isfile(caffemodel):
		raise IOError(('{:s} not found.\nDid you run ./pre-trained-models/'
					   'fetch_pre_trained_model.sh?').format(caffemodel))

	if args.cpu_mode:
		caffe.set_mode_cpu()
	else:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu_id)
		cfg.GPU_ID = args.gpu_id
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)

	print '\n\nLoaded network {:s}'.format(caffemodel)

	CONF_THRESH = 0.65
	NMS_THRESH = 0.15

	crop_w = 500
	crop_h = 500
	step = 400

	im_names = ['images/D0090-5242001.tiff']

	# Warmup on a dummy image
	im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
	for i in xrange(2):
		_, _= im_detect(net, im)

	for im_name in im_names:
		im = cv2.imread(os.path.join(data_dir, im_name))

		# # Detect all object classes and regress object bounds
		timer = Timer()
		timer.tic()
		# scores, boxes = im_detect(net, im)
		scores, boxes = im_detect_sliding_crop(net, im, crop_h, crop_w, step)
		timer.toc()
		print ('Detection took {:.3f}s for '
		       '{:d} object proposals').format(timer.total_time, boxes.shape[0])

		dir_name, mat_name = os.path.split(im_name)
		if not os.path.exists(os.path.join(work_dir, dir_name)):
			os.makedirs(os.path.join(work_dir, dir_name))

		res = {'boxes': boxes, 'scores': scores}
		sio.savemat(os.path.join(work_dir, mat_name), res)

		dets = np.hstack((boxes,
				  scores)).astype(np.float32)
		keep = nms(dets, NMS_THRESH)
		dets = dets[keep, :]

		keep = np.where(dets[:, 4] > CONF_THRESH)
		dets = dets[keep]

		# vis_detections(im, 'words', dets, CONF_THRESH)
		save_detections(im, im_name, dets, CONF_THRESH)