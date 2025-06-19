import pandas as pd
from complementNB import AssetClassifier
from configuration import NAME_FILE_DATA, RANDOM_SAMPLING, MIN_SAMPLES
from joblib import parallel_backend

from typing import List
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Grid, ScrollableContainer, Container
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Label, Markdown

from additional_functions import (select_excel_file,
                                  load_and_validate_excel,
                                  find_cls_model_files,

)

from data_text import (NAME_APP,
                       NAME_OUTPUT_FILE,
                       SUB_TITLE_APP,
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


class QuitScreen(ModalScreen):
    """Screen with a dialog to quit."""

    def compose(self) -> ComposeResult:
        yield Grid(
            Label("Укажите файл с данными для обучения", id="overview_window"),
            Button("Обзор", variant="success", id="review"),
            id="dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.app.exit()
        else:
            self.app.pop_screen()


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
    grid-size: 1;
    grid-gutter: 1 1;
    grid-rows: 1fr 3;
    padding: 0 1;
    width: 60;
    height: 11;
    border: thick $background 80%;
    background: $surface;
}
    #overview_window {
    column-span: 2;
    height: 1fr;
    width: 1fr;
    content-align: center middle;
}
    #review {
    column-span: 1;
    height: 1fr;
    width: 50%;
    content-align: center middle;
}
    #button-grid {
        height: auto;
        dock: bottom;
        width: 100%;
        padding: 1;
        background: $surface;
    }

    QuitScreen {
    align: center middle;
}

    Grid {
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
                Button("Обучить модель", variant="success", id="train"),
                Button("Классифицировать", variant="primary", id="classify"),
                id="button-grid"
            )
        )
        yield Footer()

    def on_mount(self) -> None:
        self.title = NAME_APP
        self.sub_title = SUB_TITLE_APP

    def action_request_quit(self) -> None:
        """Action to display the quit dialog."""
        self.push_screen(QuitScreen())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "train":
            self.push_screen(QuitScreen())
        elif event.button.id == "classify":
            print("Нажата кнопка Классифицировать")


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