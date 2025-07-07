# build_ennoia.spec
# Build with: pyinstaller --clean build_ennoia.spec

import os
import streamlit
from PyInstaller.utils.hooks import (
    copy_metadata,
    collect_submodules,
    collect_dynamic_libs
)

block_cipher = None

# 📦 Metadata for all required packages
datas = []
for pkg in [
    'streamlit', 'cryptography', 'pandas', 'numpy', 'requests',
    'torch', 'transformers', 'peft', 'pyserial', 'matplotlib', 'scipy'
]:
    datas += copy_metadata(pkg)

# 🖼️ Streamlit frontend assets
streamlit_static = os.path.join(os.path.dirname(streamlit.__file__), "static")
datas += [(streamlit_static, "streamlit/static")]

# 🧩 Your custom .py modules and assets
datas += [
    ('ennoiaCAT_RAG_INT.py', '.'),
    ('build_ennoia.py', '.'),  # launcher script
    ('ennoia_client_lic.py', '.'),
    ('map_api.py', '.'),
    ('tinySA_config.py', '.'),
    ('tinySA.py', '.'),  # If it's local
    ('ennoia.jpg', '.'),
    ('license.json', '.'),
    ('max_signal_strengths.csv', '.')
]

# 🧬 Dynamic binaries from PyTorch
binaries = collect_dynamic_libs('torch')

# 🧠 Hidden imports — critical for dynamic modules
hiddenimports = (
    collect_submodules('torch') +
    collect_submodules('transformers') +
    collect_submodules('matplotlib') +
    collect_submodules('scipy') +
    [
        'peft',
        'serial.tools.list_ports',
        'streamlit.runtime',
        'streamlit.runtime.scriptrunner',
        'streamlit.runtime.scriptrunner.magic_funcs'
    ]
)

a = Analysis(
    ['build_ennoia.py'],  # 👈 launcher script
    pathex=[os.path.abspath('.')], 
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='build_ennoia',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True  # Set to False for a GUI-only launcher
)