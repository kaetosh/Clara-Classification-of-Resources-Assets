from pathlib import Path
import pandas as pd
import subprocess
import sys
import os
import numpy as np
import ctypes
from ctypes import wintypes


from typing import List


from custom_errors import MissingColumnsError, RowCountError, LoadFileError, CancelingFileSelectionError, NoFilesToDeleteError

# Явно определим недостающий тип PVOID
if not hasattr(wintypes, 'PVOID'):
    wintypes.PVOID = ctypes.c_void_p

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        print(f"Running in PyInstaller temp dir: {base_path}")  # Debug
    except AttributeError:
        base_path = os.path.abspath(".")
        print(f"Running in dev mode, base path: {base_path}")  # Debug

    full_path = os.path.join(base_path, relative_path)
    normalized_path = os.path.normpath(full_path)
    print(f"Resource final path: {normalized_path}")  # Debug
    return normalized_path

def load_font_temp(font_path: str) -> bool:
    """Упрощенная версия загрузки шрифта"""
    try:
        font_path_abs = os.path.abspath(font_path)
        if not os.path.exists(font_path_abs):
            return False

        # Простая версия без Ex-функции
        result = ctypes.windll.gdi32.AddFontResourceW(font_path_abs)
        return result > 0
    except:
        return False

def unload_font_temp(font_path: str) -> bool:
    """Выгружает временный шрифт"""
    try:
        FR_PRIVATE = 0x10
        font_path_unicode = os.path.abspath(font_path)

        RemoveFontResourceEx = ctypes.windll.gdi32.RemoveFontResourceExW
        RemoveFontResourceEx.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.PVOID]
        RemoveFontResourceEx.restype = wintypes.BOOL

        if not os.path.exists(font_path_unicode):
            print(f"Файл шрифта не найден: {font_path_unicode}")
            return False

        res = RemoveFontResourceEx(font_path_unicode, FR_PRIVATE, None)
        if not res:
            raise ctypes.WinError()

        HWND_BROADCAST = 0xFFFF
        WM_FONTCHANGE = 0x001D
        ctypes.windll.user32.SendMessageW(HWND_BROADCAST, WM_FONTCHANGE, 0, 0)
        return True

    except Exception as e:
        print(f"Ошибка выгрузки шрифта: {e}")
        return False


def check_font() -> bool:
        """Упрощенная проверка шрифта без зависимостей"""
        try:
            from matplotlib import font_manager
            return "Cascadia Code" in {f.name for f in font_manager.fontManager.ttflist}
        except:
            # Если проверка невозможна, предполагаем что шрифта нет
            return False



def confusion_matrix_to_markdown(cm: np.ndarray, classes: np.ndarray) -> str:
    """
    Возвращает матрицу ошибок для markdown разметки.
    """
    header = "| Истина \\ Прогноз | " + " | ".join(classes) + " |\n"
    separator = "|---" * (len(classes) + 1) + "|\n"
    rows = ""
    for i, class_name in enumerate(classes):
        row = f"| {class_name} | " + " | ".join(str(x) for x in cm[i]) + " |\n"
        rows += row
    return header + separator + rows


def find_cls_model_files() -> list[Path]:
    """
    Ищет файлы .joblib, начинающиеся на 'cls_model' в текущей директории.

    Возвращает:
        list[Path]: Список путей к найденным файлам (объекты Path).
    """
    current_dir = Path.cwd()
    return list(current_dir.glob("cls_model*.joblib"))


def load_and_validate_excel(
    file_path: Path,
    required_columns: set,
    min_rows: int = None
) -> pd.DataFrame:
    """
    Загружает Excel-файл и проверяет наличие заданных столбцов и (опционально) минимальное количество строк.

    Параметры:
        file_path (Path): Путь к файлу Excel,
        required_columns (set): Множество обязательных столбцов. По умолчанию {'Наименование', 'Группа'},
        min_rows (int или None): Минимальное количество строк. Если None — проверка не выполняется.

    Возвращает:
        DataFrame: Если проверки пройдены.

    Исключения:
        CancelingFileSelectionError: если file_path не задан.
        MissingColumnsError: если отсутствуют обязательные столбцы.
        RowCountError: если количество строк меньше min_rows.
        LoadFileError: при ошибках чтения файла или других исключениях.
    """
    try:
        if not file_path:
            print('ОШИБКА if not file_path')
            raise CancelingFileSelectionError("Отмена выбора файла")

        df = pd.read_excel(file_path)

        # Проверка столбцов
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise MissingColumnsError(f"Ошибка: отсутствуют столбцы {missing}")

        # Проверка количества строк (если задана)
        if min_rows is not None and len(df) < min_rows:
            raise RowCountError(f"Ошибка: в файле только {len(df)} строк (требуется >= {min_rows})")

        if df.empty:
            raise RowCountError('Таблица пустая')

        return df
    except FileNotFoundError:
        raise LoadFileError(f"Ошибка загрузки файла: {file_path.name} не найден")

    except Exception as e:
        raise LoadFileError(f"Ошибка загрузки файла: {e}")

def open_excel_file(path: Path):
    """
    Открывает Excel-файл в приложении Excel по заданному пути.

    Args:
        path (Path): Путь к файлу Excel.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    # Приводим путь к строке с абсолютным путем
    file_path = str(path.resolve())

    if sys.platform == "win32":
        # Windows: используем startfile
        os.startfile(file_path)
    elif sys.platform == "darwin":
        # macOS: открываем через open
        subprocess.run(["open", "-a", "Microsoft Excel", file_path])
    else:
        # Linux и другие: пытаемся открыть через xdg-open
        # (Excel на Linux обычно не установлен, но можно попытаться)
        subprocess.run(["xdg-open", file_path])


def delete_files_by_type(file_types: List[int]) -> None:
    """
    Удаляет файлы в текущей папке по указанным типам.
    Вызывает NoFilesToDeleteError, если файлы не найдены.

    Параметры:
        file_types: Список чисел, где:
            [0] - удалить все .xlsx файлы
            [1] - удалить все .joblib файлы
            [0, 1] - удалить оба типа файлов

    Исключения:
        ValueError: Если передан недопустимый тип файла
        NoFilesToDeleteError: Если не найдены файлы для удаления
        PermissionError: Если нет прав на удаление файлов
        OSError: При других ошибках файловой системы
    """
    valid_inputs = [[0], [1], [0, 1]]
    if file_types not in valid_inputs:
        raise ValueError(f"Недопустимый список типов. Допустимые значения: {valid_inputs}")

    extensions = []
    if 0 in file_types:
        extensions.append('.xlsx')
    if 1 in file_types:
        extensions.append('.joblib')

    files_found = False

    try:
        for filename in os.listdir('.'):
            if any(filename.endswith(ext) for ext in extensions):
                files_found = True
                try:
                    os.remove(filename)
                    print(f"Удалён файл: {filename}")
                except PermissionError as e:
                    raise PermissionError(f"Нет прав на удаление файла {filename}") from e
                except OSError as e:
                    raise OSError(f"Ошибка при удалении файла {filename}") from e

        if not files_found:
            raise NoFilesToDeleteError("Не найдено файлов для удаления с указанными расширениями")

    except OSError as e:
        raise OSError(f"Ошибка при чтении содержимого папки") from e


