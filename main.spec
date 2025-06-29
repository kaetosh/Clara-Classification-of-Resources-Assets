# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

# Собираем все необходимые зависимости
hiddenimports = collect_submodules('sklearn') + collect_submodules('scipy') + collect_submodules('numpy')
binaries = collect_dynamic_libs('scipy') + collect_dynamic_libs('numpy')

# Включаем шрифт и данные sklearn
datas = [
    ('CascadiaCode.ttf', '.'),  # ваш шрифт
    *collect_data_files('sklearn')  # все дополнительные файлы sklearn
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Изменения для сборки в папку (не onefile)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # Важно!
    name='ClaraApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon='icon.ico'  # можно добавить иконку
)

# Критически важная секция для сборки в папку
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ClaraApp'
)