# PyInstaller spec（V2.0）
# 用法：pyinstaller packaging/house_ai_gis_v2.spec

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

hiddenimports = []
hiddenimports += collect_submodules("pandas")
hiddenimports += collect_submodules("matplotlib")

# 可选依赖：若安装了再一起打包
try:
    hiddenimports += collect_submodules("pyecharts")
except Exception:
    pass
try:
    hiddenimports += collect_submodules("ezdxf")
except Exception:
    pass
try:
    hiddenimports += collect_submodules("paddleocr")
except Exception:
    pass
try:
    hiddenimports += collect_submodules("easyocr")
except Exception:
    pass

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[("assets", "assets")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="房屋影像识别与属性建库制图系统V2.0",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon="icon.ico",
)

