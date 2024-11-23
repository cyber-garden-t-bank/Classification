import cv2
import numpy as np
import easyocr
from transformers import T5TokenizerFast, AutoModelForSeq2SeqLM
from PIL import Image
import re
import torch
import os

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


def upscale_image(image, scale_factor=2):
    """
    Увеличение разрешения изображения с использованием интерполяции.
    
    Args:
        image (np.ndarray): Входное изображение.
        scale_factor (int): Коэффициент масштабирования.
    
    Returns:
        np.ndarray: Апскейленное изображение.
    """
    height, width = image.shape[:2]
    new_dimensions = (width * scale_factor, height * scale_factor)
    upscaled = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('upscaled.png', upscaled)  
    return upscaled


def preprocess_image(image_path):
    """
    Предобрабатывает входное изображение для улучшения качества OCR.
    Включает в себя: масштабирование, выравнивание, коррекцию контраста,
    удаление шумов, бинаризацию и морфологические операции.

    Args:
        image_path (str): Путь к входному изображению.

    Returns:
        np.ndarray: Предобработанное изображение.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {image_path}")

    scale_factor = 2
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

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]  
    min_size = 20 

    image_filtered = np.zeros((labels.shape), np.uint8)

    for i in range(1, num_labels):
        if sizes[i - 1] >= min_size:
            image_filtered[labels == i] = 255

    image = image_filtered

    cv2.imwrite('preprocessed.png', image)

    return image

def extract_and_save_words(image, image_path):
    """
    Детекция и распознавание слов с помощью EasyOCR, сохранение каждого слова как отдельного изображения.
    
    Args:
        image (np.ndarray): Предобработанное изображение.
        image_path (str): Путь к исходному изображению.
    
    Returns:
        list of str: Список распознанных слов.
    """
    results = reader_easyocr.readtext(image, detail=1, paragraph=False)
    
    words = []
    for idx, (bbox, text, conf) in enumerate(results):
        if conf < 0.3:
            continue
        
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = (int(top_left[0]), int(top_left[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        
        word_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        
        if word_image.size == 0:
            continue
        
        word_filename = os.path.join(images_dir, f'word_{idx + 1}.png')
        cv2.imwrite(word_filename, word_image)
        
        words.append(text)
        print(f"Сохранено слово {idx + 1}: '{text}' в файл {word_filename} с уверенностью {conf:.2f}")
    
    return words


def correct_text(text):
    """
    Коррекция извлечённого текста с использованием модели на основе T5.
    
    Args:
        text (str): Входной текст.

    Returns:
        str: Исправленный текст.
    """
    if not text:
        return ""
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
    return corrected


def parse_receipt(words_text):
    """
    Парсинг скорректированного текста для извлечения структурированных данных.
    
    Args:
        words_text (list of str): Список скорректированных слов.
    
    Returns:
        dict: Структурированные данные с товарами и итоговой суммой.
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
    

    words = extract_and_save_words(preprocessed_image, image_path)
    
    if not words:
        print("Не удалось распознать слова на изображении.")
        return
    
    corrected_words = []
    for idx, word in enumerate(words):
        corrected = correct_text(word)
        corrected_words.append(corrected)
        print(f"Скорректированное слово {idx + 1}: {corrected}")
    
    parsed_data = parse_receipt(corrected_words)
    
    print("\nРаспознанные данные:")
    print(parsed_data)


if __name__ == "__main__":
    main()
