from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from textual import work

from custom_errors import MissingColumnsError, RowCountError, LoadFileError, CancelingFileSelectionError

# import re
# import joblib
# import numpy as np
# import pandas as pd
# from datetime import datetime
# from string import punctuation
# from collections import Counter
# from sklearn.model_selection import train_test_split
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.feature_extraction.text import TfidfVectorizer

# from configuration import STOPWORDS_RU

# # Функция безопасного разделения
# def safe_stratified_split(X, y, test_size, random_state):
#     counts = Counter(y)
#     min_samples = min(counts.values())

#     if min_samples < 2:
#         # Находим классы с 1 элементом
#         rare_classes = [cls for cls, cnt in counts.items() if cnt == 1]

#         # Создаем маски
#         train_mask = ~np.isin(y, rare_classes)
#         test_mask = np.isin(y, rare_classes)

#         # Основные классы (стратифицированно)
#         X_train, X_test, y_train, y_test = train_test_split(
#             X[train_mask],
#             y[train_mask],
#             test_size=test_size,
#             stratify=y[train_mask],
#             random_state=random_state
#         )

#         # Редкие классы (все в тест)
#         X_test = np.concatenate([X_test, X[test_mask]])
#         y_test = np.concatenate([y_test, y[test_mask]])

#         return X_train, X_test, y_train, y_test
#     else:
#         return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


# # Проверка датасета
# def is_sample_suitable(X, y, label_encoder=None, min_samples=1000, min_classes=2, min_samples_per_class=5):
#     """
#     Проверка, подходит ли выборка для обучения.

#     Параметры:
#     - X: признаки (тексты)
#     - y: метки (закодированные числа)
#     - label_encoder: экземпляр LabelEncoder для декодирования меток
#     - min_samples: минимальное общее количество образцов
#     - min_classes: минимальное количество уникальных классов
#     - min_samples_per_class: минимальное количество примеров для каждого класса

#     Возвращает:
#     - True, если выборка подходит, иначе False
#     """
#     # Проверка общего количества образцов
#     if len(X) < min_samples:
#         print(f"⚠️ Выборка слишком мала (n_samples={len(X)} < {min_samples}).")
#         return False

#     # Проверка количества уникальных классов
#     unique_classes, class_counts = np.unique(y, return_counts=True)
#     if len(unique_classes) < min_classes:
#         print(f"⚠️ Слишком мало классов (n_classes={len(unique_classes)} < {min_classes}).")
#         return False

#     # Проверка, что в каждом классе достаточно примеров
#     if np.any(class_counts < min_samples_per_class):
#         # Декодируем числовые метки в оригинальные названия (если передан label_encoder)
#         if label_encoder is not None:
#             problematic_classes = {
#                 label_encoder.inverse_transform([cls])[0]: int(count)
#                 for cls, count in zip(unique_classes, class_counts)
#                 if count < min_samples_per_class
#             }
#         else:
#             problematic_classes = {
#                 int(cls): int(count)
#                 for cls, count in zip(unique_classes, class_counts)
#                 if count < min_samples_per_class
#             }

#         print(f"⚠️ Некоторые классы содержат слишком мало примеров: {problematic_classes} < {min_samples_per_class}.")
#         return False

#     # Проверка, что после векторизации останутся признаки
#     vectorizer = TfidfVectorizer(min_df=1, max_df=1.0)
#     try:
#         X_vec = vectorizer.fit_transform(X)
#         if X_vec.shape[1] == 0:
#             print("⚠️ После векторизации не осталось признаков. Попробуйте изменить `min_df`/`max_df`.")
#             return False
#     except ValueError as e:
#         print(f"⚠️ Ошибка векторизации: {e}")
#         return False

#     return True

# def make_predictions(model_path, encoder_path, input_file, output_file=None):
#     """Полный цикл предсказания на новых данных"""
#     try:
#         # 1. Загрузка компонентов
#         print("\nЗагрузка сохраненных компонентов...")
#         model = joblib.load(model_path)  # Загружаем ВЕСЬ пайплайн
#         encoder = joblib.load(encoder_path)

#         # 2. Чтение данных
#         print("Чтение входного файла...")
#         new_data = pd.read_excel(input_file)
#         print(f"Загружено {len(new_data)} строк")

#         # 3. Проверка данных
#         if 'name' not in new_data.columns:
#             raise ValueError("Файл должен содержать столбец 'name'")

#         new_data = new_data.dropna()

#         # 4. Предсказание (векторизация происходит внутри пайплайна автоматически)
#         print("Выполнение предсказаний...")
#         new_data['predicted_group'] = encoder.inverse_transform(model.predict(new_data['name'].astype(str)))

#         # 5. Сохранение
#         if not output_file:
#             output_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

#         new_data.to_excel(output_file, index=False)
#         print(f"\nПредсказания сохранены в: {output_file}")
#         return new_data

#     except Exception as e:
#         print(f"\nОшибка: {str(e)}")
#         return None

# class TextCleaner(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         return X.astype(str).apply(clean_text)

# @work(thread=True)
# def select_excel_file_threadsafe() -> Optional[Path]:
#     return select_excel_file()  # Tkinter в фоновом потоке

# def select_excel_file() -> Optional[Path]:
#     """Выбор файла Excel без вызова destroy()/quit()."""
#     root = tk.Tk()
#     root.withdraw()
#     root.wm_attributes('-topmost', 1)  # Диалог поверх других окон

#     file_path = filedialog.askopenfilename(
#         title="Выберите файл Excel",
#         filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
#     )
#     return Path(file_path) if file_path else None

def find_cls_model_files() -> list[Path]:
    """
    Ищет файлы .joblib, начинающиеся на 'cls_model' в текущей директории.

    Возвращает:
        list[Path]: Список путей к найденным файлам (объекты Path).
    """
    current_dir = Path.cwd()
    return list(current_dir.glob("cls_model*.joblib"))



def load_and_validate_excel(file_path: Path, min_rows: int) -> pd.DataFrame:
    """
    Загружает Excel-файл и проверяет:
    1) Наличие столбцов 'name' и 'group'.
    2) Достаточное количество строк (>= min_rows).

    Параметры:
        file_path (Path): Путь к файлу Excel.
        min_rows (int): Минимально необходимое количество строк.

    Возвращает:
        pd.DataFrame: Если проверки пройдены.
        Exception: Если проверки не пройдены или возникла ошибка.
    """
    try:
        if not file_path:
            print('ОШИБКА if not file_path')
            raise CancelingFileSelectionError("Отмена выбора файла для обучения")
        df = pd.read_excel(file_path)

        # Проверка столбцов
        required_columns = {'name', 'group'}
        if not required_columns.issubset(df.columns):
            raise MissingColumnsError(f"Ошибка: отсутствуют столбцы {required_columns - set(df.columns)}")

        # Проверка количества строк
        if len(df) < min_rows:
            raise RowCountError(f"Ошибка: в файле только {len(df)} строк (требуется >= {min_rows})")
        return df

    except Exception as e:
        print(f"ОШИБКА в load_and_validate_excel: {e}")
        raise LoadFileError(f"Ошибка загрузки файла: {e}")
