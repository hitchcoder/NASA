import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi import FastAPI, File, UploadFile, HTTPException
from inference import predict_from_fits_file # Импортируем нашу новую функцию

app = FastAPI(
    title="Exoplanet Hunter API (File Upload)",
    description="Upload FITS file to detect exoplanets.",
    version="2.0.0",
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict_file", summary="Предсказать по FITS файлу")
async def create_upload_file(file: UploadFile = File(...)):
    """
    Принимает FITS файл (`.fits`), анализирует его кривую блеска 
    и возвращает вероятность наличия планеты.
    """
    # Проверяем, что это FITS файл (хотя бы по расширению)
    if not file.filename.endswith('.fits'):
        raise HTTPException(status_code=400, detail="Неверный формат файла. Пожалуйста, загрузите .fits файл.")
    
    print(f"\nПолучен файл для анализа: {file.filename}")
    
    # Читаем содержимое файла в байты
    contents = await file.read()
    
    # Отправляем содержимое на обработку
    result = predict_from_fits_file(contents)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    # Дополняем результат именем файла
    result["filename"] = file.filename
    return result
