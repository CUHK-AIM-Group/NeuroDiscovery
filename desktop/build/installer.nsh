!include nsDialogs.nsh
!include LogicLib.nsh

!ifndef BUILD_UNINSTALLER
Var RuntimePage
Var RuntimeBuiltInRadio
Var RuntimeCondaRadio
Var RuntimePythonRadio
Var RuntimeDetectButton
Var RuntimePythonField
Var RuntimeCondaField
Var RuntimeCondaEnvField
Var RuntimeFslField
Var RuntimeProxyField
Var RuntimeHintLabel
Var InstallerRuntimeMode
Var InstallerPythonExe
Var InstallerCondaExe
Var InstallerCondaEnv
Var InstallerFslDir
Var InstallerProxyUrl

Page custom NeuroClawRuntimePage NeuroClawRuntimePageLeave

Function NeuroClawRuntimeDetect
  IfFileExists "$PROFILE\anaconda3\Scripts\conda.exe" 0 +3
    StrCpy $InstallerCondaExe "$PROFILE\anaconda3\Scripts\conda.exe"
    Goto detectCondaDone
  IfFileExists "$PROFILE\miniconda3\Scripts\conda.exe" 0 +3
    StrCpy $InstallerCondaExe "$PROFILE\miniconda3\Scripts\conda.exe"
    Goto detectCondaDone
  IfFileExists "$PROFILE\miniforge3\Scripts\conda.exe" 0 +2
    StrCpy $InstallerCondaExe "$PROFILE\miniforge3\Scripts\conda.exe"

detectCondaDone:
  StrCmp $InstallerCondaEnv "" 0 +2
    StrCpy $InstallerCondaEnv "neuroclaw"

  IfFileExists "$PROFILE\anaconda3\envs\neuroclaw\python.exe" 0 +3
    StrCpy $InstallerPythonExe "$PROFILE\anaconda3\envs\neuroclaw\python.exe"
    Goto detectPythonDone
  IfFileExists "$PROFILE\miniconda3\envs\neuroclaw\python.exe" 0 +3
    StrCpy $InstallerPythonExe "$PROFILE\miniconda3\envs\neuroclaw\python.exe"
    Goto detectPythonDone
  IfFileExists "$PROFILE\miniforge3\envs\neuroclaw\python.exe" 0 +2
    StrCpy $InstallerPythonExe "$PROFILE\miniforge3\envs\neuroclaw\python.exe"

detectPythonDone:
  IfFileExists "C:\Program Files\FSL" 0 +2
    StrCpy $InstallerFslDir "C:\Program Files\FSL"

  StrCmp $InstallerProxyUrl "" 0 +2
    StrCpy $InstallerProxyUrl "http://127.0.0.1:7897"

  ${NSD_SetText} $RuntimePythonField "$InstallerPythonExe"
  ${NSD_SetText} $RuntimeCondaField "$InstallerCondaExe"
  ${NSD_SetText} $RuntimeCondaEnvField "$InstallerCondaEnv"
  ${NSD_SetText} $RuntimeFslField "$InstallerFslDir"
  ${NSD_SetText} $RuntimeProxyField "$InstallerProxyUrl"
FunctionEnd

Function NeuroClawRuntimePage
  nsDialogs::Create 1018
  Pop $RuntimePage

  ${If} $RuntimePage == error
    Abort
  ${EndIf}

  StrCmp $InstallerRuntimeMode "" 0 +2
    StrCpy $InstallerRuntimeMode "bundled"
  StrCmp $InstallerCondaEnv "" 0 +2
    StrCpy $InstallerCondaEnv "neuroclaw"
  StrCmp $InstallerProxyUrl "" 0 +2
    StrCpy $InstallerProxyUrl "http://127.0.0.1:7897"

  ${NSD_CreateLabel} 0u 0u 300u 12u "Configure NeuroClaw runtime"
  ${NSD_CreateLabel} 0u 15u 300u 18u "Choose the Python runtime and optional dependency paths."
  ${NSD_CreateLabel} 0u 42u 300u 12u "Default Python runtime"
  ${NSD_CreateRadioButton} 10u 62u 270u 14u "Built-in Python"
  Pop $RuntimeBuiltInRadio
  ${NSD_CreateRadioButton} 10u 80u 270u 14u "Use local conda environment"
  Pop $RuntimeCondaRadio
  ${NSD_CreateRadioButton} 10u 98u 270u 14u "Use custom python.exe"
  Pop $RuntimePythonRadio

  StrCmp $InstallerRuntimeMode "conda" 0 +3
    SendMessage $RuntimeCondaRadio ${BM_SETCHECK} ${BST_CHECKED} 0
    Goto runtimeRadioDone
  StrCmp $InstallerRuntimeMode "python" 0 +3
    SendMessage $RuntimePythonRadio ${BM_SETCHECK} ${BST_CHECKED} 0
    Goto runtimeRadioDone
  SendMessage $RuntimeBuiltInRadio ${BM_SETCHECK} ${BST_CHECKED} 0

runtimeRadioDone:
  ${NSD_CreateButton} 190u 118u 90u 16u "Auto detect"
  Pop $RuntimeDetectButton
  ${NSD_OnClick} $RuntimeDetectButton NeuroClawRuntimeDetect

  ${NSD_CreateLabel} 0u 142u 75u 12u "python.exe"
  ${NSD_CreateText} 78u 138u 206u 16u "$InstallerPythonExe"
  Pop $RuntimePythonField

  ${NSD_CreateLabel} 0u 164u 75u 12u "conda.exe"
  ${NSD_CreateText} 78u 160u 206u 16u "$InstallerCondaExe"
  Pop $RuntimeCondaField

  ${NSD_CreateLabel} 0u 186u 75u 12u "conda env"
  ${NSD_CreateText} 78u 182u 206u 16u "$InstallerCondaEnv"
  Pop $RuntimeCondaEnvField

  ${NSD_CreateLabel} 0u 208u 75u 12u "FSLDIR"
  ${NSD_CreateText} 78u 204u 206u 16u "$InstallerFslDir"
  Pop $RuntimeFslField

  ${NSD_CreateLabel} 0u 230u 75u 12u "Proxy"
  ${NSD_CreateText} 78u 226u 206u 16u "$InstallerProxyUrl"
  Pop $RuntimeProxyField

  ${NSD_CreateLabel} 0u 252u 286u 24u "Advanced users can configure local dependency paths now. These values can also be changed later in Settings."
  Pop $RuntimeHintLabel

  nsDialogs::Show
FunctionEnd

Function NeuroClawRuntimePageLeave
  SendMessage $RuntimeBuiltInRadio ${BM_GETCHECK} 0 0 $0
  ${If} $0 == ${BST_CHECKED}
    StrCpy $InstallerRuntimeMode "bundled"
  ${Else}
    SendMessage $RuntimeCondaRadio ${BM_GETCHECK} 0 0 $0
    ${If} $0 == ${BST_CHECKED}
      StrCpy $InstallerRuntimeMode "conda"
    ${Else}
      StrCpy $InstallerRuntimeMode "python"
    ${EndIf}
  ${EndIf}

  ${NSD_GetText} $RuntimePythonField $InstallerPythonExe
  ${NSD_GetText} $RuntimeCondaField $InstallerCondaExe
  ${NSD_GetText} $RuntimeCondaEnvField $InstallerCondaEnv
  ${NSD_GetText} $RuntimeFslField $InstallerFslDir
  ${NSD_GetText} $RuntimeProxyField $InstallerProxyUrl

  ${If} $InstallerRuntimeMode == "conda"
  ${AndIf} $InstallerCondaExe == ""
    MessageBox MB_ICONEXCLAMATION "Please provide conda.exe or choose Built-in Python."
    Abort
  ${EndIf}

  ${If} $InstallerRuntimeMode == "python"
  ${AndIf} $InstallerPythonExe == ""
    MessageBox MB_ICONEXCLAMATION "Please provide python.exe or choose Built-in Python."
    Abort
  ${EndIf}
FunctionEnd

Function NeuroClawJsonPath
  Exch $0
  StrCpy $1 ""
  StrCpy $2 0
jsonPathLoop:
  StrCpy $3 $0 1 $2
  StrCmp $3 "" jsonPathDone
  StrCmp $3 "\" 0 +3
    StrCpy $1 "$1/"
    Goto jsonPathNext
  StrCpy $1 "$1$3"
jsonPathNext:
  IntOp $2 $2 + 1
  Goto jsonPathLoop
jsonPathDone:
  StrCpy $0 $1
  Exch $0
FunctionEnd

!macro customInstall
  CreateDirectory "$APPDATA\NeuroClaw"

  Push "$InstallerPythonExe"
  Call NeuroClawJsonPath
  Pop $R0

  Push "$InstallerCondaExe"
  Call NeuroClawJsonPath
  Pop $R1

  Push "$InstallerFslDir"
  Call NeuroClawJsonPath
  Pop $R2

  Push "$INSTDIR\resources\runtime\backend"
  Call NeuroClawJsonPath
  Pop $R3

  FileOpen $R4 "$APPDATA\NeuroClaw\desktop-config.json" w
  FileWrite $R4 "{$\r$\n"
  FileWrite $R4 '  "host": "127.0.0.1",$\r$\n'
  FileWrite $R4 '  "port": 7080,$\r$\n'
  FileWrite $R4 '  "runtimeMode": "$InstallerRuntimeMode",$\r$\n'
  FileWrite $R4 '  "pythonExe": "$R0",$\r$\n'
  FileWrite $R4 '  "condaExe": "$R1",$\r$\n'
  FileWrite $R4 '  "condaEnv": "$InstallerCondaEnv",$\r$\n'
  FileWrite $R4 '  "repoRoot": "$R3",$\r$\n'
  FileWrite $R4 '  "fslDir": "$R2",$\r$\n'
  FileWrite $R4 '  "language": "English",$\r$\n'
  FileWrite $R4 '  "proxyUrl": "$InstallerProxyUrl"$\r$\n'
  FileWrite $R4 "}$\r$\n"
  FileClose $R4
!macroend
!endif
