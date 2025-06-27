from pathlib import Path
from typing import Optional, List

from textual.app import App, ComposeResult
from textual.containers import Grid, ScrollableContainer, Container
from textual.widgets import Button, Footer, Header, Markdown
from textual.reactive import reactive

from data_text import TEXT_INTRODUCTION, NAME_APP, SUB_TITLE_APP
from screens import TrainingWarningModal, PredictWarningModal, FontWarningModal, ClearDirModal
from additional_functions import check_font




import os
import sys
import ctypes
from ctypes import wintypes

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






class ClaraApp(App):
    """An app with a modal dialog."""
    ENABLE_COMMAND_PALETTE = True
    selected_path_file_train: reactive[Optional[Path]] = reactive(None)
    selected_path_file_predict: reactive[Optional[Path]] = reactive(None)
    selected_path_file_joblib: reactive[Optional[Path]] = reactive(None)
    report: reactive[Optional[str]] = reactive(None)
    result: reactive[Optional[str]] = reactive(None)
    selected_files_for_clear: List[str] = reactive(None)

    CSS = """

        ToastRack {
            position: relative;
            offset: 0 -5;

        }
    Screen {
        layout: vertical;
        }


        Button {
            width: 100%;
        }

        FontWarningModal {
            align: center middle;
        }
            #grid-font-warning-modal {
                grid-size: 1 2;
                grid-gutter: 1 0;
                grid-rows: 65% 35%;
                width: 65;
                height: 12;
                border: solid $accent;
                background: $surface;
            }

            #label-font-warning-modal {
                align: center top;
                text-align: center;
            }
            #button-font-warning-modal {
                align: center bottom;
                width: 50%;
                offset: 25% 0;
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
                text-align: center;
                }


        FileSelectTrainModal {
            align: center middle;
        }

            #grid-file-select-train-modal {
                grid-size: 2 3;
                grid-gutter: 1 1;
                grid-rows: 10% 66% 24%;
                padding: 0 1;
                width: 70;
                height: 25;
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


        ClearDirModal {
            align: center middle;
        }

            #grid-clear-dir-modal {
                grid-size: 2 3;
                grid-gutter: 1 1;
                grid-rows: 10% 66% 24%;
                padding: 0 1;
                width: 70;
                height: 25;
                border: solid $accent;
                background: $surface;
            }

            #label-clear-dir-modal {
                column-span: 2;
                content-align: center middle;
                }

            SelectionList {
                column-span: 2;
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


        PredictWarningModal {
            align: center middle;
        }

            #grid-predict-warning-modal {
                grid-size: 2;
                grid-gutter: 1 2;
                grid-rows: 1fr 3;
                padding: 0 1;
                width: 60;
                height: 12;
                border: solid $accent;
                background: $surface;
            }

            #label-predict-warning-modal {
                column-span: 2;
                height: 1fr;
                width: 1fr;
                content-align: center middle;
                text-align: center;
                }


        FileSelectPredictModal {
            align: center middle;
        }

            #grid-file-select-predict-modal {
                grid-size: 2 3;
                grid-gutter: 1 1;
                grid-rows: 10% 66% 24%;
                padding: 0 1;
                width: 70;
                height: 25;
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


        FileSelectModelModal {
            align: center middle;
        }

            #grid-file-select-model-modal {
                grid-size: 2 3;
                grid-gutter: 1 1;
                grid-rows: 10% 66% 24%;
                padding: 0 1;
                width: 70;
                height: 25;
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


        PrintReportModal {
            align: center middle;
        }

            #grid-print-report-modal {
                grid-size: 1 3;
                grid-gutter: 1 1;
                grid-rows: 5% 80% 15%;
                padding: 0 0;
                border: solid $accent;
                background: $surface;
            }


            .grid-buttons {
                grid-size: 2 1;
                grid-gutter: 1 1;
                width: 60%;
                offset: 21% 0;
                content-align: center middle;

            }


        PrintPredictModal {
            align: center middle;
        }

            #grid-print-predict-modal {
                grid-size: 1 3;
                grid-gutter: 1 1;
                grid-rows: 5% 80% 15%;
                padding: 0 0;
                border: solid $accent;
                background: $surface;
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
        grid-gutter: 1 1;
        width: 60%;
        align: center middle;

        }
    #container-main-buttons {
        height: 10%;
        align: center bottom;
        }

    """

    BINDINGS = [
        ("ctrl+d", "clear_dir", "Очистить папку")]


    def compose(self) -> ComposeResult:
        yield Header(show_clock=True, icon='')
        yield ScrollableContainer(
            Markdown(TEXT_INTRODUCTION),
            id="scroll-container-introduction"
        )
        yield Container(
            Grid(
                Button("▶ Обучить модель ◀", variant="primary", id="button-train"),
                Button("▶ Классифицировать ◀", variant="warning", id="button-classify"),
                id="grid-main-buttons"
                ), id="container-main-buttons"
        )
        yield Footer()



    def on_mount(self) -> None:
        self.title = NAME_APP
        self.sub_title = SUB_TITLE_APP

        if os.name == 'nt':
            font_path = resource_path("CascadiaCode.ttf")
            print(f"Attempting to load font from: {font_path}")  # Debug
            print(f"Font exists: {os.path.exists(font_path)}")    # Debug
            if not os.access(font_path, os.R_OK):
                print(f"ERROR: No read access to font file at {font_path}")
            else:
                if load_font_temp(font_path):
                    print("Font loaded successfully")
                else:
                    print("Failed to load font")

        if not check_font():
            self.push_screen(FontWarningModal())

    def on_shutdown(self) -> None:
        if os.name == 'nt':
            font_path = resource_path("CascadiaCode.ttf")
            unload_font_temp(font_path)


    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "button-train":
            self.push_screen(TrainingWarningModal())
        elif event.button.id == "button-classify":
            self.push_screen(PredictWarningModal())

    def action_clear_dir(self) -> None:
        self.push_screen(ClearDirModal())


if __name__ == "__main__":
    app = ClaraApp()
    app.run()

