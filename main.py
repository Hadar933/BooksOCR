import os
import image_operators as io
import utils
import ocr


def main(
		image_path: str,
		image_operators: list[io.ImageOperator],
		ocr_engine: ocr.OCREngine,
		verbose: bool = False,
		save_ocr: bool = False
) -> dict[str: dict]:
	"""
	Apply a list of operators to an image and calculate the OCR results
	:param image_path: path to the image
	:param image_operators: list of image operators to apply
	:param ocr_engine: the OCR engine to use
	:param verbose: plots the results of each operator
	:param save_ocr: saves the OCR results to a txt file
	:return: dictionary of all ocr and image operator results
	"""
	all_results = dict()
	for i, op_instance in enumerate(image_operators):
		print(f'[{i + 1}/{len(image_operators)}] {op_instance.__str__()}')
		op_result = op_instance(image_path)
		op_ocr = ocr_engine(op_result)
		all_results[str(op_instance)] = {'image': op_result, 'ocr': op_ocr}
		if verbose:
			op_instance.plot(image_path, op_result)
		if save_ocr:
			utils.save_ocr(op_instance, op_ocr)
	return all_results


if __name__ == '__main__':
	ocrs = main(
		image_path='resources/images/books.jpg',
		image_operators=[
			io.Identity(),
			io.Composition([
				io.GaussianBlur(),
				io.Sobel()
			])
		],
		ocr_engine=ocr.Tesseract(language='heb'),
		verbose=True,
		save_ocr=True
	)
	print(ocrs['Identity']['ocr'])
	print('=' * 50)
	print(ocrs['GaussianBlurSobel']['ocr'])
