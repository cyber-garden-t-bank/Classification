import cv2
import numpy as np
import easyocr
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoModelForSeq2SeqLM, T5TokenizerFast
from PIL import Image
import re
import torch
import os

# Отключение предупреждений параллелизма токенизаторов
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Настройка устройства (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Загрузка модели для коррекции орфографии
model_name = 'UrukHan/t5-russian-spell'
tokenizer = T5TokenizerFast.from_pretrained(model_name)
model_spell = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Инициализация OCR-инструментов
reader_easyocr = easyocr.Reader(['ru'], gpu=torch.cuda.is_available())
processor_tr = TrOCRProcessor.from_pretrained('kazars24/trocr-base-handwritten-ru')
model_tr = VisionEncoderDecoderModel.from_pretrained('kazars24/trocr-base-handwritten-ru').to(device)


def preprocess_image(image_path):
    """
    Предобработка входного изображения: конвертация в оттенки серого и адаптивная бинаризация.
    
    Args:
        image_path (str): Путь к входному изображению.

    Returns:
        np.ndarray: Бинаризованное изображение.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Применение адаптивной бинаризации для лучшей обработки различных условий освещения
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 8)
    cv2.imwrite('binary.png', binary)  # Сохранение для отладки
    return binary


def extract_lines_projection(image):
    """
    Сегментация чека на отдельные строки с использованием проекционного профиля.

    Args:
        image (np.ndarray): Бинаризованное изображение чека.

    Returns:
        list of np.ndarray: Список изображений отдельных строк.
    """
    # Инвертируем изображение: текст белый, фон черный
    inverted = cv2.bitwise_not(image)
    cv2.imwrite('inverted_projection.png', inverted)  # Сохранение для отладки

    # Подсчитываем количество белых пикселей по строкам (горизонтальная проекция)
    horizontal_projection = np.sum(inverted, axis=1)
    cv2.imwrite('horizontal_projection.png', (horizontal_projection / np.max(horizontal_projection) * 255).astype(np.uint8))  # Визуализация проекции

    # Определяем порог для разделения строк
    threshold = np.max(horizontal_projection) * 0.2  # Настройка порога может потребоваться
    lines = []
    start = None
    padding = 5  # Отступы между строками

    for i, value in enumerate(horizontal_projection):
        if value > threshold and start is None:
            start = i
        elif value <= threshold and start is not None:
            end = i
            # Добавляем отступы
            y_start = max(start - padding, 0)
            y_end = min(end + padding, image.shape[0])
            line = image[y_start:y_end, :]
            lines.append(line)
            start = None

    # Добавляем последнюю строку, если она существует
    if start is not None:
        y_start = max(start - padding, 0)
        y_end = min(image.shape[0], image.shape[0])
        line = image[y_start:y_end, :]
        lines.append(line)

    print(f"Найдено {len(lines)} строк.")
    return lines


def extract_lines_easyocr(image_path):
    """
    Использование EasyOCR для обнаружения и сегментации строк.

    Args:
        image_path (str): Путь к изображению.

    Returns:
        list of tuple: Список кортежей (bbox, text, conf).
    """
    results = reader_easyocr.readtext(image_path, detail=1, paragraph=False)
    lines = []
    for bbox, text, conf in results:
        lines.append((bbox, text, conf))
        print(f"Распознанная строка: {text} с уверенностью {conf}")
    return lines


def preprocess_line(line_image):
    """
    Предобработка изображения строки.

    Args:
        line_image (np.ndarray): Изображение строки.

    Returns:
        np.ndarray: Предобработанное изображение строки.
    """
    # Увеличение контраста
    processed = cv2.equalizeHist(line_image)
    # Дополнительная бинаризация при необходимости
    _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return processed


def extract_text_easyocr(image):
    """
    Извлечение текста из изображения с использованием EasyOCR.
    
    Args:
        image (np.ndarray): Предобработанное изображение.

    Returns:
        str: Извлеченный текст.
    """
    if image is None:
        return ""
    result = reader_easyocr.readtext(image, detail=0, paragraph=True)
    return ' '.join(result)


def extract_text_tesseract(image):
    """
    Извлечение текста из изображения с использованием Tesseract OCR.
    
    Args:
        image (np.ndarray): Предобработанное изображение.

    Returns:
        str: Извлеченный текст.
    """
    if image is None:
        return ""
    return pytesseract.image_to_string(image, lang='rus')


def extract_text_trocr(image):
    """
    Извлечение текста из изображения с использованием TrOCR.
    
    Args:
        image (np.ndarray): Предобработанное изображение.

    Returns:
        str: Извлеченный текст.
    """
    if image is None:
        return ""
    image_pil = Image.fromarray(image).convert("RGB")
    pixel_values = processor_tr(images=image_pil, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model_tr.generate(pixel_values)
    generated_text = processor_tr.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def correct_text(text):
    """
    Коррекция извлеченного текста с использованием модели на основе T5.
    
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
        predicts = model_spell.generate(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
    corrected = tokenizer.batch_decode(predicts, skip_special_tokens=True)[0]
    return corrected


def parse_receipt(lines_text):
    """
    Парсинг скорректированного текста для извлечения структурированных данных.
    
    Args:
        lines_text (list of str): Список скорректированных строк текста.

    Returns:
        dict: Структурированные данные с товарами и итоговой суммой.
    """
    items = []
    total = None
    item_pattern = re.compile(r'([А-Яа-яЁё\s]+)\s+(\d+[,\.]\d{2})')
    total_pattern = re.compile(r'(ИТОГО|TOTAL)\s+(\d+[,\.]\d{2})')
    
    for line in lines_text:
        # Проверка на итоговую сумму
        total_match = total_pattern.search(line)
        if total_match:
            try:
                total = float(total_match.group(2).replace(',', '.'))
                print(f"Найдена итоговая сумма: {total}")
            except ValueError:
                print(f"Не удалось преобразовать итоговую сумму: {total_match.group(2)}")
            continue
        
        # Проверка на товар и цену
        item_match = item_pattern.search(line)
        if item_match:
            name = item_match.group(1).strip()
            price_str = item_match.group(2).replace(',', '.')
            try:
                price = float(price_str)
                items.append({'name': name, 'price': price})
                print(f"Найден товар: {name}, цена: {price}")
            except ValueError:
                print(f"Не удалось преобразовать цену: {price_str}")
    
    return {'items': items, 'total': total}


def main():
    """
    Основная функция для запуска пайплайна по извлечению и обработке текста.
    """
    image_path = 'check-subtotal-1.jpg'  # Убедитесь, что путь к изображению верен
    print(f"Загружаем изображение: {image_path}")
    try:
        processed_image = preprocess_image(image_path)
    except FileNotFoundError as e:
        print(e)
        return

    # Сегментация чека на строки с использованием проекционного профиля
    lines = extract_lines_projection(processed_image)

    if not lines:
        print("Не удалось сегментировать чек на строки с помощью проекционного профиля.")
        # Попробуем альтернативный метод с использованием EasyOCR
        print("Попробуем сегментировать строки с использованием EasyOCR.")
        ocr_results = extract_lines_easyocr(image_path)
        if not ocr_results:
            print("Не удалось сегментировать чек на строки с помощью EasyOCR.")
            return
        # Допустим, каждая строка это отдельный текст
        corrected_lines = []
        for idx, (bbox, text, conf) in enumerate(ocr_results):
            print(f"\nОбрабатываем строку {idx + 1}/{len(ocr_results)}")
            corrected_text = correct_text(text)
            print(f"Скорректированный текст: {corrected_text}")
            corrected_lines.append(corrected_text)
    else:
        corrected_lines = []

        for idx, line in enumerate(lines):
            print(f"\nОбрабатываем строку {idx + 1}/{len(lines)}")
            preprocessed = preprocess_line(line)
            cv2.imwrite(f'line_{idx + 1}.png', preprocessed)  # Сохранение для отладки

            # Извлечение текста с помощью EasyOCR
            text_easy = extract_text_easyocr(preprocessed)
            corrected_easy = correct_text(text_easy)
            print(f"EasyOCR: {corrected_easy}")

            # Извлечение текста с помощью Tesseract
            text_tesseract = extract_text_tesseract(preprocessed)
            corrected_tesseract = correct_text(text_tesseract)
            print(f"Tesseract: {corrected_tesseract}")

            # Извлечение текста с помощью TrOCR
            text_trocr = extract_text_trocr(preprocessed)
            corrected_trocr = correct_text(text_trocr)
            print(f"TrOCR: {corrected_trocr}")

            # Объединение результатов OCR (можно использовать наиболее частый результат или другой метод)
            # Здесь просто объединяем все три результата для простоты
            combined_text = f"{corrected_easy} {corrected_tesseract} {corrected_trocr}"
            combined_text = combined_text.strip()
            corrected_combined = correct_text(combined_text)
            print(f"Объединенный и скорректированный текст: {corrected_combined}")

            corrected_lines.append(corrected_combined)

    # Парсинг всех строк
    parsed_data = parse_receipt(corrected_lines)

    print("\nРаспознанные данные:")
    print(parsed_data)


if __name__ == "__main__":
    main()
