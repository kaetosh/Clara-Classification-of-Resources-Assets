from pathlib import Path
from typing import Optional

# from textual.web import run

from textual.app import App, ComposeResult
from textual.containers import Grid, ScrollableContainer, Container
from textual.widgets import Button, Footer, Header, Markdown
from textual.reactive import reactive

from data_text import TEXT_INTRODUCTION, NAME_APP, SUB_TITLE_APP
from screens import TrainingWarningModal, PredictWarningModal
import os

# --- Код установки шрифта ---
# def install_font():
#     font_path = Path(__file__).parent / "CascadiaCode.ttf"
#     if os.name == 'nt' and font_path.exists():
#         try:
#             ctypes.windll.gdi32.AddFontResourceW(str(font_path))
#             # Обновляем кэш шрифтов
#             ctypes.windll.user32.SendMessageW(0xFFFF, 0x001D, 0, 0)
#         except Exception as e:
#             print(f"Не удалось установить шрифт: {e}")

# Вызываем при старте приложения
# install_font()
# ---------------------------

class ClaraApp(App):
    """An app with a modal dialog."""

    selected_path_file_train: reactive[Optional[Path]] = reactive(None)
    selected_path_file_predict: reactive[Optional[Path]] = reactive(None)
    selected_path_file_joblib: reactive[Optional[Path]] = reactive(None)
    report: reactive[Optional[str]] = reactive(None)
    result: reactive[Optional[str]] = reactive(None)

    CSS = """
    Screen {
        layout: vertical;
        }

        LoaderIndicatorCustom {
            align: center middle;
        }

            #grid-loader_indicator {
                grid-size: 1 2;
                grid-gutter: 1 2;
                padding: 0 1;
                width: 80;
                height: 22;
                border: solid $accent;
                background: $surface;
            }

        TrainingWarningModal {
            align: center middle;
        }

            #grid-training-warning-modal {
                grid-size: 2;
                grid-gutter: 1 2;
                grid-rows: 1fr 3;
                padding: 0 1;
                width: 60;
                height: 10;
                border: solid $accent;
                background: $surface;
            }

            #label-training-warning-modal {
                column-span: 2;
                height: 1fr;
                width: 1fr;
                content-align: center middle;
                }

            Button {
                width: 100%;
            }

        FileSelectTrainModal {
            align: center middle;
        }

            #grid-file-select-train-modal {
                grid-size: 2 3;
                grid-gutter: 1 1;
                grid-rows: 8% 66% 26%;
                padding: 0 1;
                width: 70;
                height: 21;
                border: solid $accent;
                background: $surface;
            }

            #label-file-select-train-modal {
                column-span: 2;
                content-align: center middle;
                }

            #tree-file-select-train-modal {
                column-span: 2;
                }

            Button {
                width: 100%;
            }

        SetNameModelModal {
            align: center middle;
        }

            #grid-set-name-model-modal {
                grid-size: 2 3;
                grid-gutter: 1 1;
                padding: 0 1;
                width: 70;
                height: 15;
                border: solid $accent;
                background: $surface;
            }

            #label-set-name-model-modal {
                column-span: 2;
                content-align: center middle;
                }

            #input-set-name-model-modal {
                column-span: 2;
                }

            Button {
                width: 100%;
            }

        PredictWarningModal {
            align: center middle;
        }

            #grid-predict-warning-modal {
                grid-size: 2;
                grid-gutter: 1 2;
                grid-rows: 1fr 3;
                padding: 0 1;
                width: 60;
                height: 15;
                border: solid $accent;
                background: $surface;
            }

            #label-predict-warning-modal {
                column-span: 2;
                height: 1fr;
                width: 1fr;
                content-align: center middle;
                }

            Button {
                width: 100%;
            }

        FileSelectPredictModal {
            align: center middle;
        }

            #grid-file-select-predict-modal {
                grid-size: 2 3;
                grid-gutter: 1 1;
                grid-rows: 10% 60% 30%;
                padding: 0 1;
                width: 70;
                height: 21;
                border: solid $accent;
                background: $surface;
            }

            #label-file-select-predict-modal {
                column-span: 2;
                content-align: center middle;
                }

            #tree-file-select-predict-modal {
                column-span: 2;
                }

            Button {
                width: 100%;
            }

        FileSelectModelModal {
            align: center middle;
        }

            #grid-file-select-model-modal {
                grid-size: 2 3;
                grid-gutter: 1 1;
                grid-rows: 10% 60% 30%;
                padding: 0 1;
                width: 70;
                height: 21;
                border: solid $accent;
                background: $surface;
            }

            #label-file-select-model-modal {
                column-span: 2;
                content-align: center middle;
                }

            #tree-file-select-model-modal {
                column-span: 2;
                }

            Button {
                width: 100%;
            }

        PrintReportModal {
            align: center middle;
        }

            #grid-print-report-modal {
                grid-size: 2 3;
                grid-gutter: 1 1;
                grid-rows: 5% 80% 15%;
                padding: 0 0;
                border: solid $accent;
                background: $surface;
            }

            #label-print-report-modal {
                column-span: 2;
                content-align: center middle;
                }

            #markdown-print-report-modal {
                column-span: 2;
                }

            Button {
                width: 100%;
            }

        PrintPredictModal {
            align: center middle;
        }

            #grid-print-predict-modal {
                grid-size: 2 3;
                grid-gutter: 1 1;
                grid-rows: 10% 75% 15%;
                padding: 0 1;
                border: solid $accent;
                background: $surface;
            }

            #label-print-predict-modal {
                column-span: 2;
                content-align: center middle;
                }

            #markdown-print-predict-modal {
                column-span: 2;
                }

            Button {
                width: 100%;
            }

    #scroll-container-introduction {
        height: 90%;
        overflow-y: auto;
        border: solid $accent;
        margin: 1;
        }

    #grid-main-buttons {
        layout: grid;
        grid-size: 2 1;
        grid-gutter: 1;
        }
    #container-main-buttons {
        height: 10%;
        }

    """



    def compose(self) -> ComposeResult:
        yield Header()
        yield ScrollableContainer(
            Markdown(TEXT_INTRODUCTION),
            id="scroll-container-introduction"
        )
        yield Container(
            Grid(
                Button("Обучить модель", variant="default", id="button-train"),
                Button("Классифицировать", variant="default", id="button-classify"),
                id="grid-main-buttons"
            ), id="container-main-buttons"
        )
        yield Footer()



    def on_mount(self) -> None:
        self.title = NAME_APP
        self.sub_title = SUB_TITLE_APP

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "button-train":
            self.push_screen(TrainingWarningModal())
        elif event.button.id == "button-classify":
            self.push_screen(PredictWarningModal())


if __name__ == "__main__":
    app = ClaraApp()
    app.run()

