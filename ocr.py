from abc import ABC

import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

import numpy as np


class OCREngine(ABC):
	"""	Abstract class for OCR engines """

	def __init__(self, language):
		self.language = language

	def __call__(self, image: np.ndarray):
		raise NotImplementedError


class Tesseract(OCREngine):
	"""	OCR engine using pytesseract """

	def __init__(self, language):
		super().__init__(language)

	def __call__(self, image: np.ndarray) -> str:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # pytesseract works with RGB images
		result = pytesseract.image_to_string(image, lang=self.language)
		return result
