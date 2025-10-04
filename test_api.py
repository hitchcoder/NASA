# test_api.py (версия для загрузки файлов)
import requests
import json
import os

API_URL = "http://127.0.0.1:8000/predict_file"
# Укажите имя вашего тестового FITS файла
TEST_FILE_PATH = "test_case.fits" 

if not os.path.exists(TEST_FILE_PATH):
    print(f"Ошибка: Тестовый файл '{TEST_FILE_PATH}' не найден!")
    print("Пожалуйста, скачайте любой .fits файл с кривой блеска Kepler и поместите его в папку проекта.")
else:
    print(f"--- Отправка файла '{TEST_FILE_PATH}' на анализ ---")
    
    # Открываем файл в бинарном режиме для чтения ('rb')
    with open(TEST_FILE_PATH, 'rb') as f:
        # 'files' - это специальный параметр для отправки файлов в requests
        files = {'file': (os.path.basename(TEST_FILE_PATH), f, 'application/fits')}
        
        try:
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("Ответ от API получен:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(f"Ошибка! Статус-код: {response.status_code}")
                print("Ответ:", response.text)
                
        except requests.exceptions.ConnectionError:
            print("Ошибка соединения! Убедитесь, что API сервер запущен.")