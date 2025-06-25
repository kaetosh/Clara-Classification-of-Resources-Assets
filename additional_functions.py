from pathlib import Path
import pandas as pd
import subprocess
import sys
import os
import numpy as np

from custom_errors import MissingColumnsError, RowCountError, LoadFileError, CancelingFileSelectionError



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
