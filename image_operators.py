from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import cv2
import numpy as np


class ImageOperator(ABC):
	"""
	Abstract class for image operators
	"""

	def __str__(self) -> str:
		return self.__class__.__name__

	@abstractmethod
	def __call__(self, image_input: str | np.ndarray) -> np.ndarray:
		"""
		Abstract method for applying the operator
		:param image_input: either a path to an image or a numpy array.
		:return: np array, the result of the operator.
		"""
		raise NotImplementedError

	@staticmethod
	def read_image(image_input: str | np.ndarray) -> np.ndarray:
		if isinstance(image_input, str):
			image_input = cv2.imread(image_input, cv2.IMREAD_COLOR)
		return image_input

	def plot(
			self,
			image_input: str | np.ndarray,
			result: np.ndarray | None = None
	) -> None:
		"""
		Plot the original image and the result of the operator
		:param image_input: either a path to an image or a numpy array.
		:param result: the result of the operator. if not provided, the operator will be called.
		:return:
		"""
		image = self.read_image(image_input)
		if result is None: result = self(image_input)
		plt.figure(figsize=(10, 5))
		plt.subplot(1, 2, 1)
		plt.imshow(image, cmap='gray')
		plt.title('Original Image')
		plt.axis('off')

		plt.subplot(1, 2, 2)
		plt.imshow(result, cmap='gray')
		plt.title(str(self))
		plt.axis('off')
		plt.tight_layout()
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
	def __init__(self, kernel_size=(1, 1), iterations=1):
		"""
		Erosion operator for noise reduction
		:param kernel_size: tuple (width, height) of the kernel
		:param iterations: number of iterations for the erosion
		"""
		self.kernel_size = kernel_size
		self.iterations = iterations

	def __call__(self, image_input):
		""" Erosion operator for noise reduction """
		image = self.read_image(image_input)
		kernel = np.ones(self.kernel_size, np.uint8)
		erosion = cv2.erode(image, kernel, iterations=self.iterations)
		return erosion


class Resize(ImageOperator):
	def __init__(self, x_scale_factor: float = 1.2, y_scale_factor: float = 1.2, interpolation=cv2.INTER_CUBIC):
		"""
		Resize the image by specified scaling factors for x and y axes
		:param x_scale_factor: The scaling factor for the x-axis
		:param y_scale_factor: The scaling factor for the y-axis
		:param interpolation: Interpolation method to use
		"""
		self.x_scale_factor = x_scale_factor
		self.y_scale_factor = y_scale_factor
		self.interpolation = interpolation

	def __call__(self, image_input):
		image = self.read_image(image_input)
		resized_image = cv2.resize(image, None, fx=self.x_scale_factor, fy=self.y_scale_factor,
								   interpolation=self.interpolation)
		return resized_image


class ConvertToGray(ImageOperator):
	def __call__(self, image_input):
		image = self.read_image(image_input)
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		return gray_image


class BilateralFilterAndThreshold(ImageOperator):
	def __init__(self, d: int = 5, sigma_color: int = 75, sigma_space: int = 75):
		"""
		Apply bilateral filter and thresholding
		:param d: Diameter of each pixel neighborhood
		:param sigma_color: Filter sigma in the color space
		:param sigma_space: Filter sigma in the coordinate space
		"""
		self.d = d
		self.sigma_color = sigma_color
		self.sigma_space = sigma_space

	def __call__(self, image_input):
		image = self.read_image(image_input)
		bilateral_filtered = cv2.bilateralFilter(image, self.d, self.sigma_color, self.sigma_space)
		_, thresholded = cv2.threshold(bilateral_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		return thresholded


class Dilate(ImageOperator):
	def __init__(self, kernel_size=(1, 1), iterations=1):
		"""
		:param kernel_size: tuple (width, height) of the kernel
		:param iterations: number of iterations for the erosion
		"""
		self.kernel_size = kernel_size
		self.iterations = iterations

	def __call__(self, image_input):
		""" Dilation operator for noise reduction """
		image = self.read_image(image_input)
		kernel = np.ones(self.kernel_size, np.uint8)
		dilation = cv2.dilate(image, kernel, iterations=self.iterations)
		return dilation


class GaussianBlur(ImageOperator):
	def __init__(self, kernel_size=(5, 5), sigma_x=0, sigma_y=0):
		"""
		Gaussian blur operator for noise reduction
		:param kernel_size: tuple (width, height) of the kernel
		:param sigma_x: standard deviation in x-direction
		:param sigma_y: standard deviation in y-direction
		"""
		self.kernel_size = kernel_size
		self.sigma_x = sigma_x
		self.sigma_y = sigma_y

	def __call__(self, image_input):
		""" Gaussian blur operator """
		image = self.read_image(image_input)
		blurred = cv2.GaussianBlur(image, self.kernel_size, self.sigma_x, self.sigma_y)
		return blurred


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
		return "".join([op.__str__() for op in self.op_list])
