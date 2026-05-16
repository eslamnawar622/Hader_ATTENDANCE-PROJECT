# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['/home/eslamnawar/AI/my_vision_server/attendace_version2/app-final2.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['PIL._imagingtk', 'cv2', 'dlib', 'face_recognition', 'face_recognition_models', 'pygame', 'gtts', 'customtkinter', 'numpy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AxisAttendance',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AxisAttendance',
)
