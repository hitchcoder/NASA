# inference.py (версия для работы с файлами)
import tensorflow as tf
import lightkurve as lk
import numpy as np
import tempfile
import os

MODEL = tf.keras.models.load_model('best_model.h5')

GLOBAL_VIEW_SIZE = 2048
LOCAL_VIEW_SIZE = 256

def predict_from_fits_file(file_content: bytes) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=True) as temp_f:
        temp_f.write(file_content)
        temp_f.flush() 
        try:
            lc = lk.read(temp_f.name).remove_nans()
            
            if 'PDCSAP_FLUX' in lc.columns:
                lc = lc['PDCSAP_FLUX']

            flat_lc = lc.flatten(window_length=1001)
            
            normalized_flux = flat_lc.flux.value / np.median(flat_lc.flux.value)
            global_view = np.interp(
                np.linspace(0, len(normalized_flux) - 1, GLOBAL_VIEW_SIZE),
                np.arange(len(normalized_flux)),
                normalized_flux
            )
            global_view = (global_view - np.mean(global_view)) / np.std(global_view)
            global_view = global_view.reshape(1, GLOBAL_VIEW_SIZE, 1)

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

            prediction = MODEL.predict({
                'global_input': global_view,
                'local_input': local_view
            }, verbose=0)

            probability = float(prediction[0][0])
            decision = "Candidate" if probability > 0.5 else "Planet is not detected"
            
            return {
                "filename": "uploaded_file.fits", 
                "planet_probability": probability,
                "decision": decision,
                "found_period_days": float(best_fit_period.value)
            }

        except Exception as e:
            return {"error": f"Error while reading FITS file: {str(e)}"}
        

TABULAR_FEATURES_ORDER = [
    'koi_score', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_nt',
    'koi_duration', 'koi_depth', 'koi_model_snr', 'koi_impact'
]
NUM_TABULAR_FEATURES = len(TABULAR_FEATURES_ORDER)

def predict_from_fits_file_hybrid(file_content: bytes) -> dict:
    """
    Принимает FITS файл, извлекает кривую блеска, вычисляет табличные
    признаки и возвращает предсказание трехветвевой модели.
    """
    with tempfile.NamedTemporaryFile(suffix=".fits") as temp_f:
        temp_f.write(file_content)
        temp_f.flush()
        
        try:
            # 1. Чтение и предобработка
            lc_file = lk.read(temp_f.name)
            lc = lc_file.get_lightcurve('PDCSAP_FLUX').remove_nans()
            flat_lc = lc.flatten(window_length=1001)
            
            # 2. Поиск сигнала с помощью BLS
            periodogram = flat_lc.to_periodogram(method='bls')
            
            # --- ШАГ Б: Вычисление аналогов табличных признаков ---
            bls_results = periodogram.compute_stats(period=periodogram.period_at_max_power,
                                                    duration=periodogram.duration_at_max_power,
                                                    transit_time=periodogram.transit_time_at_max_power)
            
            # Создаем словарь с вычисленными признаками
            computed_features = {
                'koi_duration': bls_results['duration'][0].value * 24, # в часах
                'koi_depth': bls_results['depth'][0].value * 1e6, # в ppm
                'koi_model_snr': bls_results['snr'][0],
                'koi_impact': periodogram.impact_parameter_at_max_power, # Не всегда доступен, может быть NaN
                # Признаки, которые мы не можем вычислить, задаем нейтральными
                'koi_score': 0.5,
                'koi_fpflag_ss': 0,
                'koi_fpflag_co': 0,
                'koi_fpflag_nt': 0
            }
            
            # Собираем вектор признаков в правильном порядке
            tabular_vector = np.array([computed_features.get(key, 0) for key in TABULAR_FEATURES_ORDER], dtype=np.float32)
            # Заменяем возможные NaN на 0
            tabular_vector = np.nan_to_num(tabular_vector)
            tabular_vector = tabular_vector.reshape(1, NUM_TABULAR_FEATURES)

            # 3. Создание представлений для CNN-ветвей
            # Global View
            normalized_flux = flat_lc.flux.value / np.median(flat_lc.flux.value)
            global_view = np.interp(np.linspace(0, len(normalized_flux) - 1, GLOBAL_VIEW_SIZE), np.arange(len(normalized_flux)), normalized_flux)
            global_view = (global_view - np.mean(global_view)) / np.std(global_view)
            global_view = global_view.reshape(1, GLOBAL_VIEW_SIZE, 1)

            # Local View
            folded_lc = flat_lc.fold(period=periodogram.period_at_max_power, epoch_time=periodogram.transit_time_at_max_power)
            folded_flux = folded_lc.flux.value
            local_view = np.interp(np.linspace(0, len(folded_flux) - 1, LOCAL_VIEW_SIZE), np.arange(len(folded_flux)), folded_flux)
            local_view = (local_view - np.mean(local_view)) / np.std(local_view)
            local_view = local_view.reshape(1, LOCAL_VIEW_SIZE, 1)

            # 4. Предсказание на трехветвевой модели
            prediction = MODEL.predict({
                'global_input': global_view,
                'local_input': local_view,
                'tabular_input': tabular_vector
            }, verbose=0)

            probability = float(prediction[0][0])
            decision = "Кандидат в планеты" if probability > 0.5 else "Планета не обнаружена"
            
            return {
                "filename": "uploaded_file.fits",
                "planet_probability": probability,
                "decision": decision,
                "computed_features": computed_features,
            }

        except Exception as e:
            return {"error": f"Ошибка при обработке FITS файла: {str(e)}"}