import pandas as pd
import time
from complementNB import AssetClassifier
from configuration import NAME_FILE_DATA, RANDOM_SAMPLING, MIN_SAMPLES
from joblib import parallel_backend
import asyncio
from typing import List,Iterable, Set, Optional
from pathlib import Path

from textual import on, message
from textual.app import App, ComposeResult
from textual import work
from textual.types import NoActiveAppError
from textual.containers import Grid, ScrollableContainer, Container
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Label, Markdown, DirectoryTree, Input, Static, LoadingIndicator
from textual.reactive import reactive
from additional_functions import (
                                  load_and_validate_excel,
                                  find_cls_model_files,

)

from data_text import (NAME_APP,
                       NAME_OUTPUT_FILE,
                       SUB_TITLE_APP,
                       MIN_ROWS,
                       TEXT_INTRODUCTION,
                       TEXT_GENERAL,
                       TEXT_ERR_FILES_EXCEL,
                       TEXT_ERR_NO_PROCESSED_FILES,
                       TEXT_ERR_NOT_ALL_PROCESSED_FILES,
                       TEXT_ERR_PERMISSION,
                       TEXT_ERR_FILE_NOT_FOUND,
                       TEXT_APP_EXCEL_NOT_FIND,
                       TEXT_UNKNOW_ERR,
                       TEXT_ALL_PROCESSED_FILES,
                       TEXT_GENERATING_LIST_SHEETS,
                       TEXT_AGGREGATION_PROCESS,
                       TEXT_NOT_SELECT_DIR,
                       TEXT_ERR_NO_SELECT_SHEETS)


class ModalApp(App):
    """An app with a modal dialog."""

    CSS = """
    Screen {
        layout: vertical;
        }

    #markdown-container {
        height: 6fr;
        overflow-y: auto;
        border: solid $accent;
        margin: 1;
        }

    #dialog {
        layout: grid;
        grid-size: 2 2;
        grid-columns: 2fr 1fr;
        width: 90;
        height: 12;
        border: solid $accent;
        padding: 1 1;
        grid-gutter: 1 1;
        }

    DisplayingFolderTree {
        width: 50%;
        height: 70%;
        border: solid $accent;
        padding: 1 1;
        }
    #dir_tree {
        height: 90%;
        width: 60%;
        }

    #open-btn_dir_tree {
        height: 10%;
        width: 30%;
        padding: 1 1;

        }

    FolderOverview {

        align: center middle;
        }

    DisplayingFolderTree {
        padding: 1 1;
        align: center middle;
        }

    #button-grid {
        grid-size: 2;
        grid-gutter: 1;
        }

    Button {
        width: 100%;
        }

    Container {
        height: 1fr;
        }
    """


    def compose(self) -> ComposeResult:
        yield Header()
        yield ScrollableContainer(
            Markdown(TEXT_INTRODUCTION),
            id="markdown-container"
        )
        yield Container(
            Grid(
                Button("Обучить модель", variant="default", id="train"),
                Button("Классифицировать", variant="default", id="classify"),
                id="button-grid"
            )
        )
        yield Footer()

    def on_mount(self) -> None:
        self.title = NAME_APP
        self.sub_title = SUB_TITLE_APP

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "train":
            self.push_screen(FolderOverview())
        elif event.button.id == "classify":
            print("Нажата кнопка Классифицировать")

class FilteredDirectoryTree(DirectoryTree):
    """DirectoryTree с фильтрацией: только Excel-файлы и скрытые файлы исключены"""

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        filtered_paths = []
        for path in paths:
            # Исключаем скрытые файлы/папки (начинающиеся с точки)
            if path.name.startswith("."):
                continue

            # Включаем все папки (чтобы можно было перемещаться по дереву)
            if path.is_dir():
                filtered_paths.append(path)
            # Включаем только Excel-файлы
            elif path.suffix.lower() in ('.xlsx', '.xls'):
                filtered_paths.append(path)

        return filtered_paths

class LoaderIndicatorCustom(ModalScreen):
    def compose(self) -> ComposeResult:
        yield Static("Идет обработка данных")
        yield LoadingIndicator()


class FileSelectScreen(ModalScreen[Optional[Path]]):
    def compose(self) -> ComposeResult:
        yield Static("Выберите файл Excel (.xlsx):")
        yield FilteredDirectoryTree("./", id="tree-view")
        yield Button("Выбрать", id="select-btn", disabled=True)

    selected_path: reactive[Optional[Path]] = reactive(None)

    def on_mount(self):
        # Отключаем кнопку пока файл не выбран
        self.query_one("#select-btn").disabled = True

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected):
        """Обработчик выбора файла"""
        if event.path.suffix in (".xlsx", ".xls"):
            self.selected_path = event.path
            self.query_one("#select-btn").disabled = False
            self.query_one(Static).update(f"Выбран: {event.path.name}")
        else:
            self.notify("Выберите файл Excel (.xlsx)")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "select-btn" and self.selected_path:
            # self.dismiss(self.selected_path)

            if self.selected_path and self.selected_path.suffix in (".xlsx", ".xls"):
                self.app.push_screen(LoaderIndicatorCustom())
                self.process_file(self.selected_path)  # Запускаем фоновую задачу

    @work(thread=True)  # Запускаем в отдельном потоке, чтобы не блокировать UI
    def process_file(self, file) -> None:
        try:
            print('Запущен process_file!!!!')

            df = load_and_validate_excel(file, min_rows=MIN_ROWS)
            print("Файл загружен, вызываю on_success...")
            self.app.call_from_thread(self.on_success, df)  # Возвращаемся в основной поток
            print("запустили call_from_thread...")
        except Exception as e:
            print("ОШИБКА!!!!...")
            self.app.call_from_thread(self.on_error, str(e))  # Обработка ошибок

    def on_success(self, df: pd.DataFrame) -> None:
        print("on_success запущен...")
        self.app.pop_screen()  # Закрываем индикатор
        while len(self.app.screen_stack) > 1:
            self.app.pop_screen()
        self.app.notify("Файл успешно загружен!", title="Успех")

    def on_error(self, error: str) -> None:
        self.app.pop_screen()  # Закрываем индикатор
        self.app.notify(f"Ошибка: {error}", title="Ошибка")

class FolderOverview(ModalScreen):
    def compose(self) -> ComposeResult:
        yield Grid(
            Static("Убедитесь, что excel файл расположен в папке вместе с приложением", id="overview_window"),
            Button("ОК", variant="default", id="review"),
            id="dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "review":
            self.app.push_screen(FileSelectScreen())









if __name__ == "__main__":
    app = ModalApp()
    app.run()

# print('Загрузка данных...', end='\r')
# df_all = pd.read_excel(NAME_FILE_DATA).dropna(subset=['name', 'group'])
# print("Данные выгружены!")

# if RANDOM_SAMPLING:
#     df = df_all.sample(n=5000, random_state=42)
# else:
#     df = df_all.loc[:,:]

# # Инициализация и обучение
# classifier = AssetClassifier(max_features=20000)
# classifier.train(df, text_column='name', target_column='group')


# # Сохранение и загрузка модели
# model_path, encoder_path = classifier.save_model("my_model")
# loaded_classifier = AssetClassifier.load_model(model_path, encoder_path)

# # Предсказание на новых данных
# new_data = pd.DataFrame({'name': ["Компьютер Dell", "Офисное кресло"]})
# predictions = loaded_classifier.predict(new_data, return_proba=True)
# print(predictions)
# predictions.to_excel('1.xlsx')