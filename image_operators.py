from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import cv2
import numpy as np


class ImageOperator(ABC):
	"""
	Abstract class for image operators
	"""

	def __str__(self):
		return self.__class__.__name__

	@abstractmethod
	def __call__(self, image_input):
		"""
		Abstract method for applying the operator
		:param image_input: either a path to an image or a numpy array.
		:return: np array, the result of the operator.
		"""
		raise NotImplementedError

	@staticmethod
	def read_image(image_input):
		if isinstance(image_input, str):
			image_input = cv2.imread(image_input, cv2.IMREAD_COLOR)
		return image_input

	def plot(self, image_input, result=None):
		image = self.read_image(image_input)
		if result is None:	result = self(image_input)
		plt.figure(figsize=(10, 5))
		plt.subplot(1, 2, 1)
		plt.imshow(image, cmap='gray')
		plt.title('Original Image')
		plt.axis('off')

		plt.subplot(1, 2, 2)
		plt.imshow(result, cmap='gray')
		plt.title(str(self))
		plt.axis('off')
		plt.show()


class Identity(ImageOperator):
	def __call__(self, image_input):
		"""	Identity operator, used mostly for compatibility with other operators """

		return self.read_image(image_input)


class Sobel(ImageOperator):

	def __call__(self, image_input):
		"""	Sobel operator for edge detection """
		image = self.read_image(image_input)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		abs_grad_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3))
		abs_grad_y = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3))
		blur_gray_grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
		return blur_gray_grad


class OtsuThresholding(ImageOperator):

	def __call__(self, image_input):
		""" Otsu's thresholding for image binarization """
		image = self.read_image(image_input)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
		_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		return thresh


class HistEqualize(ImageOperator):
	def __call__(self, image_input):
		""" Histogram equalization for contrast enhancement """
		image = self.read_image(image_input)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
		equalized = cv2.equalizeHist(gray)
		return equalized


class Erode(ImageOperator):
	def __call__(self, image_input):
		""" Erosion operator for noise reduction """
		image = self.read_image(image_input)
		kernel = np.ones((3, 3), np.uint8)
		erosion = cv2.erode(image, kernel, iterations=1)
		return erosion


class Composition(ImageOperator):
	def __init__(self, op_list: list[ImageOperator]):
		"""
		composition of multiple image operators
		:param op_list: list of image operators
		"""
		self.op_list = op_list

	def __call__(self, image_input):
		image = self.read_image(image_input)
		for op_cls in self.op_list:
			image = op_cls(image)
		return image

	def __str__(self):
		return "".join([op.__name__.capitalize() for op in self.op_list])
