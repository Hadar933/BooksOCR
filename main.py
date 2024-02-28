import os
import image_operators as io

import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def main(
		image_path: str,
		op_list: list[io.ImageOperator],
		verbose: bool = False,
		save_ocr: bool = False
) -> dict[str: dict]:
	"""
	Apply a list of operators to an image and calculate the OCR results
	:param image_path: path to the image
	:param op_list: list of image operators to apply
	:param verbose: plots the results of each operator
	:param save_ocr: saves the OCR results to a txt file
	:return: dictionary of all ocr and image operator results
	"""
	all_results = dict()
	for i, op_instance in enumerate(op_list):
		print(f'[{i + 1}/{len(op_list)}] {op_instance.__str__()}')
		op_result = op_instance(image_path)
		if verbose: op_instance.plot(image_path, op_result)
		op_ocr = pytesseract.image_to_string(op_result, lang='heb')
		if save_ocr:
			ocr_path = os.path.join(os.getcwd(), f'ocr_{op_instance.__str__()}.txt')
			with open(ocr_path, 'w', encoding='utf-8') as file:
				file.write(op_ocr)
		all_results[str(op_instance)] = {'image': op_result, 'ocr': op_ocr}
	return all_results


if __name__ == '__main__':
	ocrs = main(
		image_path='books.jpg',
		op_list=[
			io.Identity(),
			io.Composition([
				io.GaussianBlur(),
				io.Sobel()
			])
		]
	)
	print(ocrs['Identity']['ocr'])
	print('=' * 50)
	print(ocrs['GaussianBlur-->Sobel']['ocr'])
