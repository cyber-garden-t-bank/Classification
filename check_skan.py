from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

def recognize_text_trocr(image_path):
    image = Image.open(image_path).convert("RGB") 
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # Генерация текста
    output_ids = model.generate(pixel_values)
    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return text

# Пример использования
image_path = "/home/dark/Documents/GitHub/Classification/check-subtotal-1.jpg"  
recognized_text = recognize_text_trocr(image_path)

# Вывод результата
print("Распознанный текст:")
print(recognized_text)
