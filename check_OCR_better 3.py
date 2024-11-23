import cv2
import numpy as np
import easyocr
from transformers import T5TokenizerFast, AutoModelForSeq2SeqLM
import os
import torch
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

model_name = 'UrukHan/t5-russian-spell'
tokenizer = T5TokenizerFast.from_pretrained(model_name)
model_spell = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

reader_easyocr = easyocr.Reader(['ru'], gpu=torch.cuda.is_available())

images_dir = 'images'
if not os.path.exists(images_dir):
    os.makedirs(images_dir)


def upscale_image(image, scale_factor=4):
    """
    Увеличение разрешения изображения с использованием интерполяции.
    """
    height, width = image.shape[:2]
    new_dimensions = (width * scale_factor, height * scale_factor)
    upscaled = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_CUBIC)
    return upscaled


def preprocess_image(image_path):
    """
    Предобрабатывает входное изображение для улучшения качества OCR.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {image_path}")

    scale_factor = 4
    image = cv2.resize(
        image,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC
    )

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)

    return image


def preprocess_word(word_image):
    """
    Предобработка изображения слова для улучшения качества OCR.
    """
    if word_image is None or word_image.size == 0:
        return word_image

    word_image = upscale_image(word_image, scale_factor=2)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    word_image = clahe.apply(word_image)

    _, word_image = cv2.threshold(word_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    word_image = cv2.medianBlur(word_image, 3)

    return word_image


def extract_and_save_words(image, image_path):
    """
    Детекция, предобработка и распознавание слов с помощью EasyOCR. 
    """
    results = reader_easyocr.readtext(image, detail=1, paragraph=False)
    words = []

    for idx, (bbox, text, conf) in enumerate(results):
        if conf < 0.2:
            print(f"Пропущено слово {idx + 1} из-за низкой уверенности ({conf:.2f}).")
            continue

        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = (int(top_left[0]), int(top_left[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        word_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        if word_image.size == 0:
            print(f"Пропущено слово {idx + 1}: пустое изображение.")
            continue

        processed_word_image = preprocess_word(word_image)

        word_filename = os.path.join(images_dir, f'word_{idx + 1}.png')
        cv2.imwrite(word_filename, processed_word_image)

        word_results = reader_easyocr.readtext(processed_word_image, detail=0, paragraph=False)
        if word_results:
            final_text = word_results[0]
            words.append((final_text, conf))
            print(f"Слово {idx + 1}: '{final_text}' сохранено в файл {word_filename} с уверенностью {conf:.2f}")
        else:
            print(f"Не удалось повторно распознать слово {idx + 1} после предобработки.")

    return words


def correct_text(text):
    """
    Коррекция извлечённого текста с использованием модели T5.
    """
    if not text:
        return text

    task_prefix = "Spell correct: "
    text_input = [task_prefix + text]
    encoded = tokenizer(
        text_input,
        padding="longest",
        max_length=256,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        predicts = model_spell.generate(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )
    corrected = tokenizer.batch_decode(predicts, skip_special_tokens=True)[0]

    # Если исправление сильно отличается, оставить исходный текст
    if len(corrected) > len(text) * 1.5 or len(corrected) < len(text) * 0.5:
        print(f"Слишком большое изменение текста. Оригинал: {text}, Исправлено: {corrected}")
        return text

    return corrected

def parse_receipt(words_text):
    """
    Парсинг скорректированного текста для извлечения структурированных данных.
    """
    items = []
    total = None
    combined_text = ' '.join(words_text)
    lines = combined_text.split('\n')  
    for line in lines:
        total_match = re.search(r'(ИТОГО|TOTAL)\s+(\d+[,\.]\d{2})', line, re.IGNORECASE)
        if total_match:
            try:
                total = float(total_match.group(2).replace(',', '.'))
                print(f"Найдена итоговая сумма: {total}")
            except ValueError:
                print(f"Не удалось преобразовать итоговую сумму: {total_match.group(2)}")
            continue
        
        item_matches = re.findall(r'([А-Яа-яЁё\s]+)\s+(\d+[,\.]\d{2})', line)
        for name, price_str in item_matches:
            try:
                price = float(price_str.replace(',', '.'))
                items.append({'name': name.strip(), 'price': price})
                print(f"Найден товар: {name.strip()}, цена: {price}")
            except ValueError:
                print(f"Не удалось преобразовать цену: {price_str}")
    
    return {'items': items, 'total': total}


def main():
    """
    Основная функция для запуска пайплайна по извлечению и обработке текста.
    """
    image_path = 'check-subtotal-1.jpg'
    print(f"Загружаем изображение: {image_path}")
    
    try:
        preprocessed_image = preprocess_image(image_path)
    except FileNotFoundError as e:
        print(e)
        return
    
    detected_words = extract_and_save_words(preprocessed_image, image_path)
    
    if not detected_words:
        print("Не удалось распознать слова на изображении.")
        return

    corrected_words = []
    for idx, (word, conf) in enumerate(detected_words):
        corrected = correct_text(word)
        corrected_words.append(corrected)
        print(f"Слово {idx + 1} до коррекции: {word}, после коррекции: {corrected}")

    parsed_data = parse_receipt([w for w, _ in detected_words])
    
    print("\nРаспознанные данные:")
    print(parsed_data)


if __name__ == "__main__":
    main()
