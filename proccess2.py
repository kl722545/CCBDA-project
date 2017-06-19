import cv2
import pickle
import numpy as np
from collections import deque
import random

class LSUN:
	def __init__(self,file_list,img_size):
		self.file_list = file_list
		self.epoch = 0
		self.file_queue = deque()
		self.append_file_buffer()
		self.itemlist = []
		self.img_size = (img_size[1], img_size[0])
	def append_file_buffer(self):
		tmp_list = self.file_list[:]
		random.shuffle(tmp_list)
		self.file_queue += tmp_list
	def append_item_buffer(self):
		if len(self.file_queue) == 0:
			self.epoch += 1
			self.append_file_buffer()
		self.read_file(self.file_queue.popleft())
	def read_file(self,file_path):
		with open (file_path, 'rb') as fp:
			self.itemlist = pickle.load(fp)
		random.shuffle(self.itemlist)
	def get_next_batch(self,batch_size):
		if len(self.itemlist) == 0:
			self.append_item_buffer()
		batch_list = self.itemlist[0:batch_size]
		self.itemlist = self.itemlist[batch_size:]
		imgs = [cv2.imdecode(np.array(item, dtype=np.uint8), 1) for item in batch_list]
		height,width = self.img_size
		imgs = [img[upper:(upper+height),left:(left+width)] for img in imgs for (upper,left) in [(random.randrange(img.shape[0]-height), random.randrange(img.shape[1]-width))]]
		return np.array(imgs)
