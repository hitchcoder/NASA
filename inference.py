# inference.py (версия для работы с файлами)
import tensorflow as tf
import lightkurve as lk
import numpy as np
import tempfile
import os

print("Загрузка обученной модели... Это может занять некоторое время.")
MODEL = tf.keras.models.load_model('best_model.h5')
print("Модель успешно загружена.")

GLOBAL_VIEW_SIZE = 2048
LOCAL_VIEW_SIZE = 256

def predict_from_fits_file(file_content: bytes) -> dict:
    """
    Принимает содержимое .fits файла в виде байтов, обрабатывает его 
    и возвращает предсказание модели.
    """
    # lightkurve может читать файлы из памяти, но надежнее сохранить во временный файл
    # tempfile автоматически создает безопасный временный файл и удаляет его после использования
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=True) as temp_f:
        temp_f.write(file_content)
        temp_f.flush() # Убедимся, что все данные записаны на диск
        
        try:
            # 1. Чтение и предобработка данных из файла
            # lk.read сразу работает с путем к файлу
            lc = lk.read(temp_f.name).remove_nans()
            
            # Если внутри FITS файла несколько кривых, попытаемся найти PDCSAP_FLUX
            if 'PDCSAP_FLUX' in lc.columns:
                lc = lc['PDCSAP_FLUX']

            flat_lc = lc.flatten(window_length=1001)
            
            # 2. Создание представлений (Global и Local View)
            # Global View
            normalized_flux = flat_lc.flux.value / np.median(flat_lc.flux.value)
            global_view = np.interp(
                np.linspace(0, len(normalized_flux) - 1, GLOBAL_VIEW_SIZE),
                np.arange(len(normalized_flux)),
                normalized_flux
            )
            global_view = (global_view - np.mean(global_view)) / np.std(global_view)
            global_view = global_view.reshape(1, GLOBAL_VIEW_SIZE, 1)

            # Local View (с поиском сигнала через BLS)
            periodogram = flat_lc.to_periodogram(method='bls')
            best_fit_period = periodogram.period_at_max_power
            best_fit_t0 = periodogram.transit_time_at_max_power
            
            folded_lc = flat_lc.fold(period=best_fit_period, epoch_time=best_fit_t0)
            folded_flux = folded_lc.flux.value
            local_view = np.interp(
                np.linspace(0, len(folded_flux) - 1, LOCAL_VIEW_SIZE),
                np.arange(len(folded_flux)),
                folded_flux
            )
            local_view = (local_view - np.mean(local_view)) / np.std(local_view)
            local_view = local_view.reshape(1, LOCAL_VIEW_SIZE, 1)

            # 3. Предсказание
            prediction = MODEL.predict({
                'global_input': global_view,
                'local_input': local_view
            }, verbose=0)

            probability = float(prediction[0][0])
            decision = "Кандидат в планеты" if probability > 0.5 else "Планета не обнаружена"
            
            return {
                "filename": "uploaded_file.fits", # В реальном приложении можно передавать имя файла
                "planet_probability": probability,
                "decision": decision,
                "found_period_days": float(best_fit_period.value)
            }

        except Exception as e:
            return {"error": f"Ошибка при обработке FITS файла: {str(e)}"}