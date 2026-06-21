const fs = require('node:fs');
const path = require('node:path');

const templatePath = path.resolve(
  __dirname,
  '..',
  'node_modules',
  'app-builder-lib',
  'templates',
  'nsis',
  'portable.nsi',
);

const template = `!include "common.nsh"
!include "extractAppPackage.nsh"

# https://github.com/electron-userland/electron-builder/issues/3972#issuecomment-505171582
CRCCheck off
WindowIcon Off
AutoCloseWindow True
RequestExecutionLevel \${REQUEST_EXECUTION_LEVEL}
InstProgressFlags smooth
Caption "Starting \${PRODUCT_NAME}"
SubCaption 3 "Preparing portable runtime"
CompletedText "Launching \${PRODUCT_NAME}..."

Page instfiles

Function .onInit
  !insertmacro check64BitAndSetRegView
FunctionEnd

Section
  DetailPrint "Preparing portable runtime..."

  StrCpy $INSTDIR "$PLUGINSDIR\\app"
  !ifdef UNPACK_DIR_NAME
    StrCpy $INSTDIR "$TEMP\\\${UNPACK_DIR_NAME}"
  !endif

  RMDir /r $INSTDIR
  SetOutPath $INSTDIR

  !ifdef APP_DIR_64
    !ifdef APP_DIR_ARM64
      !ifdef APP_DIR_32
        \${if} \${IsNativeARM64}
          File /r "\${APP_DIR_ARM64}\\*.*"
        \${elseif} \${RunningX64}
          File /r "\${APP_DIR_64}\\*.*"
        \${else}
          File /r "\${APP_DIR_32}\\*.*"
        \${endIf}
      !else
        \${if} \${IsNativeARM64}
          File /r "\${APP_DIR_ARM64}\\*.*"
        \${else}
          File /r "\${APP_DIR_64}\\*.*"
        \${endIf}
      !endif
    !else
      !ifdef APP_DIR_32
        \${if} \${RunningX64}
          File /r "\${APP_DIR_64}\\*.*"
        \${else}
          File /r "\${APP_DIR_32}\\*.*"
        \${endIf}
      !else
        File /r "\${APP_DIR_64}\\*.*"
      !endif
    !endif
  !else
    !ifdef APP_DIR_32
      File /r "\${APP_DIR_32}\\*.*"
    !else
      !insertmacro extractEmbeddedAppPackage
    !endif
  !endif

  System::Call 'Kernel32::SetEnvironmentVariable(t, t)i ("PORTABLE_EXECUTABLE_DIR", "$EXEDIR").r0'
  System::Call 'Kernel32::SetEnvironmentVariable(t, t)i ("PORTABLE_EXECUTABLE_FILE", "$EXEPATH").r0'
  System::Call 'Kernel32::SetEnvironmentVariable(t, t)i ("PORTABLE_EXECUTABLE_APP_FILENAME", "\${APP_FILENAME}").r0'
  \${StdUtils.GetAllParameters} $R0 0

  DetailPrint "Launching \${PRODUCT_NAME}..."
  HideWindow
  ExecWait "$INSTDIR\\\${APP_EXECUTABLE_FILENAME} $R0" $0
  SetErrorLevel $0

  SetOutPath $EXEDIR
  RMDir /r $INSTDIR
SectionEnd
`;

fs.writeFileSync(templatePath, template, 'utf8');
console.log(`Patched portable NSIS template: ${templatePath}`);
