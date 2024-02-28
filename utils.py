import os

import image_operators as io


def save_ocr(
		image_operator: io.ImageOperator,
		ocr_text: str,
		base_dir: str | None = None
) -> None:
	""" saves the ocr results to a txt file """
	if not base_dir:
		base_dir = os.path.join('resources', 'ocr')
	ocr_path = os.path.join(
		os.getcwd(),
		os.path.join(base_dir, f'ocr_{image_operator.__str__()}.txt')
	)
	with open(ocr_path, 'w', encoding='utf-8') as file:
		file.write(ocr_text)
