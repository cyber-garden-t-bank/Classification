import cv2
import numpy as np
import easyocr
from transformers import T5TokenizerFast, AutoModelForSeq2SeqLM
import os
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, pipeline
from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="bartowski/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF",
	filename="Llama-3.1-Nemotron-70B-Instruct-HF-IQ1_M.gguf",
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

model_name = 'UrukHan/t5-russian-spell'
tokenizer_spell = T5TokenizerFast.from_pretrained(model_name)
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

    # Апскейлим для увеличения детализации
    image = upscale_image(image, scale_factor=2)

    # Контрастирование (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    # Бинаризация
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Удаление шума
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return image


def split_text_into_lines(image):
    """
    Делит текст на строки с использованием горизонтальной проекции.
    """
    projection = np.sum(image, axis=1)
    lines = []
    start, end = None, None

    for i, value in enumerate(projection):
        if value > 0 and start is None:
            start = i
        elif value == 0 and start is not None:
            end = i
            lines.append((start, end))
            start = None

    if start is not None:
        lines.append((start, len(projection)))

    line_images = [image[start:end] for start, end in lines]
    return line_images


def split_text_into_words(line_image):
    """
    Делит строку текста на отдельные слова с использованием контурного анализа.
    """
    contours, _ = cv2.findContours(line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    word_images = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10: 
            word_image = line_image[y:y+h, x:x+w]
            word_images.append((x, word_image))

    word_images.sort(key=lambda x: x[0])  
    return [img for _, img in word_images]


def process_line_with_easyocr(line_image):
    """
    Распознаёт текст в строке целиком с помощью EasyOCR.
    """
    return reader_easyocr.readtext(line_image, detail=0, paragraph=False)


def extract_and_save_words(image):
    """
    Обрабатывает изображение текста, деля его на строки и слова.
    """
    line_images = split_text_into_lines(image)
    all_words = []

    for line_idx, line_image in enumerate(line_images):
        # Обрабатываем строку EasyOCR целиком
        line_texts = process_line_with_easyocr(line_image)
        if line_texts:
            all_words.extend(line_texts)
            word_images = split_text_into_words(line_image)
            for word_idx, word_image in enumerate(word_images):
                word_results = reader_easyocr.readtext(word_image, detail=0, paragraph=False)
                if word_results:
                    all_words.extend(word_results)
                    word_filename = os.path.join(images_dir, f'line_{line_idx}_word_{word_idx}.png')
                    cv2.imwrite(word_filename, word_image)
                    print(f"Слово сохранено: {word_results[0]} в файл {word_filename}")

    return all_words


def correct_text(text):
    """
    Коррекция извлечённого текста с использованием модели T5.
    """
    if not text:
        return text

    task_prefix = "Spell correct: "
    text_input = [task_prefix + text]
    encoded = tokenizer_spell(
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
    corrected = tokenizer_spell.batch_decode(predicts, skip_special_tokens=True)[0]

    if len(corrected) > len(text) * 1.5 or len(corrected) < len(text) * 0.5:
        print(f"Слишком большое изменение текста. Оригинал: {text}, Исправлено: {corrected}")
        return text

    return corrected

def extract_receipt_info(text_1, text_2):


    prompt = f"""
    Задача: Выделить из текста ключевую информацию о чеке. 

    Шаблон результата:
    - ИНН: [номер]
    - Товары:
      1. [название товара] - [цена за штуку] x [количество] = [итоговая цена]
      2. ...
    - Итоговая сумма: [сумма]

    Текст чека:
    {text_1}
    Текст чека вариант 2:
    {text_2}

    Если есть ошибки в тексте, учитывайте, что это может быть связано с OCR-распознаванием. Попробуйте интерпретировать данные так, чтобы они соответствовали формату чека.
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    outputs = model.generate(inputs.input_ids, max_new_tokens=500, do_sample=True, temperature=0.7)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("Текст чека:")[-1].strip()


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
    
    detected_words = extract_and_save_words(preprocessed_image)
    
    if not detected_words:
        print("Не удалось распознать слова на изображении.")
        return
    text_1 = ' '.join(detected_words)
    corrected_words = []
    for idx, word in enumerate(detected_words):
        corrected = correct_text(word)
        corrected_words.append(corrected)
        print(f"Слово {idx + 1} до коррекции: {word}, после коррекции: {corrected}")
    text_2 = ' '.join(corrected_words)
    print(text_1,"\n\n\n", text_2)
    print("\nРаспознанные данные:")
    print(extract_receipt_info(text_1, text_2))

if __name__ == "__main__":
    main()
