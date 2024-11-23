import cv2
import numpy as np
import easyocr
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoModelForSeq2SeqLM, T5TokenizerFast
from PIL import Image
import re
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'UrukHan/t5-russian-spell'
tokenizer = T5TokenizerFast.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
reader_easyocr = easyocr.Reader(['ru'], gpu=torch.cuda.is_available())
processor_tr = TrOCRProcessor.from_pretrained('kazars24/trocr-base-handwritten-ru')
model_tr = VisionEncoderDecoderModel.from_pretrained('kazars24/trocr-base-handwritten-ru')




def preprocess_image(image_path):
    """
    Preprocesses the input image by converting it to grayscale and applying binary thresholding.
    
    Args:
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Processed binary image.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return binary


def extract_text_easyocr(image):
    """
    Extracts text from an image using EasyOCR.
    
    Args:
        image (np.ndarray): Preprocessed binary image.

    Returns:
        str: Extracted text.
    """
    result = reader_easyocr.readtext(image, detail=0)
    return ' '.join(result)


def extract_text_tesseract(image):
    """
    Extracts text from an image using Tesseract OCR.
    
    Args:
        image (np.ndarray): Preprocessed binary image.

    Returns:
        str: Extracted text.
    """
    return pytesseract.image_to_string(image, lang='rus')


def extract_text_trocr(image_path):
    """
    Extracts text from an image using Hugging Face TrOCR.

    Args:
        image_path (str): Path to the input image.

    Returns:
        str: Extracted text.
    """
    image = Image.open(image_path).convert("RGB")
    # Process the image to create pixel values
    pixel_values = processor_tr(images=image, return_tensors="pt").pixel_values.to(device)
    
    # Generate text using the model
    generated_ids = model_tr.generate(pixel_values)
    
    # Decode the generated text
    generated_text = processor_tr.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text



def correct_text(text):
    """
    Corrects the extracted text using a T5-based spell correction model.
    
    Args:
        text (str): Input text.

    Returns:
        str: Corrected text.
    """
    task_prefix = "Spell correct: "
    text = [text] if not isinstance(text, list) else text
    encoded = tokenizer(
        [task_prefix + sequence for sequence in text],
        padding="longest",
        max_length=256,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    predicts = model.generate(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
    return tokenizer.batch_decode(predicts, skip_special_tokens=True)[0]


def parse_receipt(text):
    """
    Parses the corrected text to extract structured data like item names and prices.
    
    Args:
        text (str): Corrected text.

    Returns:
        dict: Parsed data containing items and total price.
    """
    items = []
    total = None
    item_pattern = re.compile(r'([А-Яа-яЁё\s]+)\s+(\d+,\d{2})')
    total_pattern = re.compile(r'(ИТОГО|TOTAL)\s+(\d+,\d{2})')
    lines = text.split('\n')
    for line in lines:
        item_match = item_pattern.search(line)
        if item_match:
            name = item_match.group(1).strip()
            price = item_match.group(2).replace(',', '.')
            items.append({'name': name, 'price': float(price)})
        total_match = total_pattern.search(line)
        if total_match:
            total = float(total_match.group(2).replace(',', '.'))
    return {'items': items, 'total': total}


def main():
    """
    Main function to run the pipeline for text extraction and processing.
    """
    image_path = 'check-subtotal-1.jpg'
    processed_image = preprocess_image(image_path)

    extracted_text_easy = extract_text_easyocr(processed_image)
    corrected_text_easy = correct_text(extracted_text_easy)
    print("EasyOCR parsed receipt:", corrected_text_easy)

    extracted_text_tesseract = extract_text_tesseract(processed_image)
    corrected_text_tesseract = correct_text(extracted_text_tesseract)
    print("Tesseract parsed receipt:", corrected_text_tesseract)

    extracted_text_trocr = extract_text_trocr(image_path)
    corrected_text_trocr = correct_text(extracted_text_trocr)
    print("TrOCR parsed receipt:", corrected_text_trocr)


if __name__ == "__main__":
    main()
