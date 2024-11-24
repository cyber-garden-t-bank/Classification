import cv2
from pyzbar.pyzbar import decode
import json
from datetime import datetime

def parse_qr_to_json(qr_data):
    pairs = qr_data.split("&")
    result = {}
    for pair in pairs:
        key, value = pair.split("=")
        result[key] = value
    
    decoded = {
        "Дата и время покупки": datetime.strptime(result["t"], "%Y%m%dT%H%M%S").strftime("%d.%m.%Y %H:%M:%S"),
        "Сумма": f"{result['s']} рублей",
        "Фискальный накопитель": result["fn"],
        "Номер фискального документа": result["i"],
        "Фискальный признак документа": result["fp"],
        "Тип операции": {
            "1": "Покупка (приход)",
            "2": "Возврат прихода",
            "3": "Расход",
            "4": "Возврат расхода"
        }.get(result["n"], "Неизвестный тип операции")
    }

    # Преобразуем в читаемый JSON
    return json.dumps(decoded, ensure_ascii=False, indent=4)
    json_data = json.dumps(result, ensure_ascii=False, indent=4)
    return json_data

def read_qr_code(image_path):
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    
    
    qr_codes = decode(binary)
    if not qr_codes:
        print("QR-код не найден.")
        return

    for qr_code in qr_codes:
        qr_data = qr_code.data.decode('utf-8')
        return parse_qr_to_json(qr_data)     



image_path = "check-subtotal-1.jpg"  
print(read_qr_code(image_path))
