# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 10:37:18 2025

@author: a.karabedyan
"""
from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import datetime

from textual import work
from textual.app import ComposeResult
from textual.containers import Grid
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, LoadingIndicator, Markdown

from additional_functions import load_and_validate_excel, open_excel_file
from widgets import ExcelDirectoryTree, JoblibDirectoryTree
from configuration import MIN_SAMPLES
from complementNB import AssetClassifier

from configuration import REQUIRED_COLUMNS


class LoaderIndicatorCustom(ModalScreen):
    def compose(self) -> ComposeResult:
        yield Grid(Label("Идет обработка данных."),
                   LoadingIndicator(),
                   id='grid-loader_indicator',
                   )

class TrainingWarningModal(ModalScreen):
    """
    Окно с предупреждением о том, что нужен файл с данными
    для обучения определенного формата. Кнопки - Продолжить и Отмена.
    При нажатии Продолжить - окно FileSelectTrainModal с выбором файла для
    обучения модели.
    """
    def compose(self) -> ComposeResult:
        yield Grid(Label("""Убедитесь, что excel файл с обучающей выборкой расположен в папке вместе с приложением. Убедитесь, что его содержимое соотвествует требованиям раздела -Обязательные условия-""",
                         id="label-training-warning-modal"),
                   Button("Продолжить", variant="default", id="button-continue-training-warning-modal"),
                   Button("Отмена", variant="default", id="button-cancel-training-warning-modal"),
                   id='grid-training-warning-modal'
                         )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "button-continue-training-warning-modal":
            self.app.push_screen(FileSelectTrainModal())
        elif event.button.id == "button-cancel-training-warning-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()


class FileSelectTrainModal(ModalScreen[Optional[Path]]):
    """
    Окно с деревом файлов текущей папки для выбора файла с данными
    для обучения. Кнопки Выбрать и Отмена. После выбора файла - запуск обучения
    модели и последующий вывод отчета по обучению (окно PrintReportModal)
    """
    def compose(self) -> ComposeResult:
        yield Grid(Label("Выберите файл Excel для обучения (.xlsx):",
                         id="label-file-select-train-modal"),
                   ExcelDirectoryTree("./", id="tree-file-select-train-modal"),
                   Button("Выбрать", id="button-select-tree-file-select-train-modal", disabled=True),
                   Button("Отмена", variant="default", id="button-cancel-tree-file-select-train-modal"),
                   id='grid-file-select-train-modal'
                   )


    def on_mount(self):
        # Отключаем кнопку пока файл не выбран
        self.query_one("#button-select-tree-file-select-train-modal").disabled = True

    def on_directory_tree_file_selected(self, event: ExcelDirectoryTree.FileSelected):
        """Обработчик выбора файла"""
        if event.path.suffix == ".xlsx":
            self.app.selected_path_file_train = event.path
            self.query_one("#button-select-tree-file-select-train-modal").disabled = False
            self.query_one("#label-file-select-train-modal").update(f"Выбран: {event.path.name}")
        else:
            self.notify("Выберите файл Excel (.xlsx) с данными для обучения")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "button-select-tree-file-select-train-modal":
            self.app.push_screen(LoaderIndicatorCustom())
            self.process_file(self.app.selected_path_file_train)  # Запускаем фоновую задачу
        elif event.button.id == "button-cancel-tree-file-select-train-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()

    @work(thread=True)  # Запускаем в отдельном потоке, чтобы не блокировать UI
    def process_file(self, file) -> None:
        try:
            df = load_and_validate_excel(file, required_columns=set(REQUIRED_COLUMNS), min_rows=MIN_SAMPLES)



            # НЕ ЗАБЫТЬ УБРАТЬ НА ПРОДЕ!!!!!!!!!
            df = df.sample(n=1000, random_state=42)




            # Инициализация и обучение
            self.app.classifier = AssetClassifier(max_features=20000)
            self.app.report = self.app.classifier.train(df, text_column=REQUIRED_COLUMNS[0], target_column=REQUIRED_COLUMNS[1])
            self.app.call_from_thread(self.on_success)  # Возвращаемся в основной поток

        except Exception as e:
            self.app.call_from_thread(self.on_error, str(e))  # Обработка ошибок

    def on_success(self) -> None:
        while len(self.app.screen_stack) > 1:
            self.app.pop_screen()
        self.app.notify("Обучение завершено", title="Статус")
        self.app.push_screen(PrintReportModal(self.app.report))


    def on_error(self, error: str) -> None:
        self.app.pop_screen()  # Закрываем индикатор
        self.app.notify(error, title="Ошибка")


class PrintReportModal(ModalScreen):
    """
    Выводит на экран результат обучения модели (метрики, описание, таблица).
    Кнопки Сохранить модель (выводит окно SetNameModelModal) и Отмена.
    """
    def __init__(self, message: str, **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def compose(self) -> ComposeResult:
        yield Grid(
            Label('Отчет по результатам обучения', id="label-print-report-modal"),
            Markdown(self.message, id='markdown-print-report-modal'),
            Button("Сохранить модель", variant="default", id="button-save-print-report-modal"),
            Button("Отмена", variant="default", id="button-cancel-print-report-modal"),
            id='grid-print-report-modal'
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "button-save-print-report-modal":
            self.app.push_screen(SetNameModelModal())
        elif event.button.id == "button-cancel-print-report-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()


class SetNameModelModal(ModalScreen):
    """
    Окно для ввода имени и сохранения модели. Кнопки Сохранить и Отмена.
    Нажатие любой кнопки закрывает текущие окна.
    """
    def compose(self) -> ComposeResult:
        yield Grid(
            Label("Модель успешно обучена. Введите имя модели и нажмите -Сохранить-",
                  id="label-set-name-model-modal"),
            Input(placeholder="Имя модели", type="text", id="input-set-name-model-modal"),
            Button("Сохранить", variant="default", id="button-save-set-name-model-modal"),
            Button("Отмена", variant="default", id="button-cancel-set-name-model-modal"),
            id="grid-set-name-model-modal",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "button-save-set-name-model-modal":
            # Сохранение модели
            name_model = self.query_one('#input-set-name-model-modal').value
            self.app.classifier.save_model(name_model)
            self.app.notify("Модель сохранена. Можно классифицировать.", title="Статус")
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()
        elif event.button.id == "button-cancel-set-name-model-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()


class PredictWarningModal(ModalScreen):
    """
    Окно с предупреждением о том, что нужен файл с данными
    для прогноза определенного формата. Кнопки - Продолжить и Отмена.
    При нажатии Продолжить - окно FileSelectPredictModal с выбором файла для
    прогноза моделью.
    """
    def compose(self) -> ComposeResult:
        yield Grid(
            Label("""Убедитесь, что excel файл с данными для классификации расположен в папке вместе с приложением. Убедитесь, что его содержимое соотвествует требованиям раздела -Получение предсказаний-. Убедитесь, что предварительно обученная модель сохранена и расположена в папке вместе с приложением.""",
                  id="label-predict-warning-modal"),
            Button("Продолжить", variant="default", id="button-continue-predict-warning-modal"),
            Button("Отмена", variant="default", id="button-cancel-predict-warning-modal"),
            id="grid-predict-warning-modal",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "button-continue-predict-warning-modal":
            self.app.push_screen(FileSelectPredictModal())
        elif event.button.id == "button-cancel-predict-warning-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()

class FileSelectPredictModal(ModalScreen[Optional[Path]]):
    """
    Окно с деревом файлов текущей папки для выбора файла с данными
    для прогноза. Кнопки Выбрать и Отмена. После выбора файла - окно FileSelectModelModal
    для выбора обученной модели для классификации.
    """
    def compose(self) -> ComposeResult:

        yield Grid(Label("Выберите файл Excel для классификации (.xlsx):",
                         id='label-file-select-predict-modal'),
                   ExcelDirectoryTree("./", id="tree-file-select-predict-modal"),
                   Button("Выбрать", id="button-select-file-select-predict-modal", disabled=True),
                   Button("Отмена", id="button-cancel-file-select-predict-modal"),
                   id='grid-file-select-predict-modal'
                   )

    def on_mount(self):
        # Отключаем кнопку пока файл не выбран
        self.query_one("#button-select-file-select-predict-modal").disabled = True

    def on_directory_tree_file_selected(self, event: ExcelDirectoryTree.FileSelected):
        """Обработчик выбора файла"""
        if event.path.suffix == ".xlsx":
            self.app.selected_path_file_predict = event.path
            self.query_one("#button-select-file-select-predict-modal").disabled = False
            self.query_one("#label-file-select-predict-modal").update(f"Выбран: {event.path.name}")
        else:
            self.notify("Выберите файл Excel (.xlsx) с данными для классификации")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "button-select-file-select-predict-modal":
            self.app.push_screen(FileSelectModelModal())
        elif event.button.id == "button-cancel-file-select-predict-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()

class FileSelectModelModal(ModalScreen[Optional[Path]]):
    """
    Окно с деревом файлов текущей папки для выбора файла с обученной моделью
    для прогноза. Кнопки Выбрать и Отмена. После выбора файла - запуск процесса
    классификации, после чего вызов окна PrintPredictModal с результатами классификации.
    """
    def compose(self) -> ComposeResult:
        yield Grid(Label("Выберите файл модели для классификации (.joblib):",
                         id='label-file-select-model-modal'),
                   JoblibDirectoryTree("./", id="tree-file-select-model-modal"),
                   Button("Выбрать", id="button-select-file-select-model-modal", disabled=True),
                   Button("Отмена", id="button-cancel-file-select-model-modal"),
                   id='grid-file-select-model-modal'
                   )

    def on_mount(self):
        # Отключаем кнопку пока файл не выбран
        self.query_one("#button-select-file-select-model-modal").disabled = True

    def on_directory_tree_file_selected(self, event: JoblibDirectoryTree.FileSelected):
        """Обработчик выбора файла"""
        if event.path.suffix == ".joblib":
            self.app.selected_path_file_joblib = event.path
            self.query_one("#button-select-file-select-model-modal").disabled = False
            self.query_one("#label-file-select-model-modal").update(f"Выбран: {event.path.name}")
        else:
            self.notify("Выберите файл модели (.joblib)")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "button-select-file-select-model-modal":
            self.app.push_screen(LoaderIndicatorCustom())
            self.process_file(file_predict_data=self.app.selected_path_file_predict,
                              model_path= self.app.selected_path_file_joblib)  # Запускаем фоновую задачу
        elif event.button.id == "button-cancel-file-select-model-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()

    @work(thread=True)  # Запускаем в отдельном потоке, чтобы не блокировать UI
    def process_file(self, file_predict_data, model_path) -> None:
        try:
            df = load_and_validate_excel(file_predict_data, required_columns={REQUIRED_COLUMNS[0]})
            loaded_classifier = AssetClassifier.load_model(model_path)
            # Предсказание на новых данных
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.app.result = f'Результат_класс_по_{model_path.stem}_{timestamp}.xlsx'
            _, md_table = loaded_classifier.predict(df, return_proba=True, output_file=self.app.result)
            self.app.call_from_thread(self.on_success, df, md_table)  # Возвращаемся в основной поток
        except Exception as e:
            self.app.call_from_thread(self.on_error, str(e))  # Обработка ошибок

    def on_success(self, df: pd.DataFrame, md_table) -> None:

        while len(self.app.screen_stack) > 1:
            self.app.pop_screen()
        self.app.notify(f"Файл {self.app.result} успешно выгружен!", title="Статус")
        self.app.push_screen(PrintPredictModal(md_table, Path(self.app.result)))

    def on_error(self, error: str) -> None:
        while len(self.app.screen_stack) > 1:
            self.app.pop_screen()
        self.app.notify(f"Ошибка: {error}", title="Ошибка")


class PrintPredictModal(ModalScreen):
    """
    Окно с результатами классификации. Кнопки Открыть... и Отмена.
    Открыть запускает сохраненный файл excel с результатами классификации и закрывает
    текущие окна.
    """
    def __init__(self, message: str, path_file_result_classification: Path, **kwargs):
        super().__init__(**kwargs)
        self.message = message
        self.path_file = path_file_result_classification

    def compose(self) -> ComposeResult:
        yield Grid(
            Label('Отчет по результатам классификации (отображено не более 100 первых строк)', id="label-print-predict-modal"),
            Markdown(self.message, id='markdown-print-predict-modal'),
            Button("Открыть результаты классификации", variant="default", id="button-open-print-predict-modal"),
            Button("Отмена", variant="default", id="button-cancel-print-predict-modal"),
            id='grid-print-predict-modal'
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "button-open-print-predict-modal":
            self.app.notify("Открываем файл с результатами классификации...", title="Статус")
            open_excel_file(self.path_file)
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()
        elif event.button.id == "button-cancel-print-predict-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()
