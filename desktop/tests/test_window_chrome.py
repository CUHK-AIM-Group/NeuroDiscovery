"""Offline title-bar contracts. No Electron, credentials or user state are opened."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_native_window_menu_security_and_theme_contracts():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js required')
    program = r'''
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('desktop/main.js','utf8');
const ctx={};
vm.runInNewContext(source.match(/^function windowChromeOptions\([^\n]*\)[\s\S]+?^}/m)[0],ctx);
const options=ctx.windowChromeOptions('win32',false);
assert.equal(options.autoHideMenuBar,false);
assert.equal('titleBarStyle' in options,false);
assert.equal('titleBarOverlay' in options,false);
assert.equal(ctx.windowChromeOptions('darwin',false).autoHideMenuBar,false);
assert.equal(ctx.windowChromeOptions('linux',false).autoHideMenuBar,false);
const frame={}, contents={mainFrame:frame}, calls=[], handlers={};
const win={webContents:contents,isDestroyed:()=>false};
const popup={popup:options=>calls.push(options)};
const menuContext={mainWindow:win,Menu:{getApplicationMenu:()=>popup},ipcMain:{handle:(name,fn)=>handlers[name]=fn}};
vm.runInNewContext(source.match(/^ipcMain.handle\('neuroclaw:show-application-menu'[\s\S]+?^\}\);/m)[0],menuContext);
const show=handlers['neuroclaw:show-application-menu'];
assert.equal(show({sender:contents,senderFrame:frame}),true);
assert.equal(calls[0].window,win);
assert.equal(show({sender:contents,senderFrame:{}}),false);
assert.equal(show({sender:{},senderFrame:frame}),false);
win.isDestroyed=()=>true;
assert.equal(show({sender:contents,senderFrame:frame}),false);
assert.equal(calls.length,1);
for(const file of ['desktop/main.js','desktop/preload.js']) new vm.Script(fs.readFileSync(file,'utf8'));
'''
    result = subprocess.run([node, '-e', program], cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_menu_and_sidebar_controls_preserve_v1_layout():
    css = (ROOT / 'core/web/static/research-workspace.css').read_text(encoding='utf-8')
    host = (ROOT / 'core/web/static/index.html').read_text(encoding='utf-8')
    main = (ROOT / 'desktop/main.js').read_text(encoding='utf-8')
    preload = (ROOT / 'desktop/preload.js').read_text(encoding='utf-8')
    assert '...windowChromeOptions(),' in main
    assert 'mainWindow.setMenuBarVisibility(true)' in main
    assert 'mainWindow.setMenuBarVisibility(false)' not in main
    assert 'titleBarStyle' not in main
    assert "showApplicationMenu: () => ipcRenderer.invoke('neuroclaw:show-application-menu')" in preload
    assert 'titleBarOverlay: false' in preload
    assert 'window.neuroclawDesktop?.titleBarOverlay && window.parent === window' in host
    assert 'id="desktop-menu-btn"' in host and 'aria-haspopup="menu" hidden' in host
    assert 'window.neuroclawDesktop?.titleBarOverlay === true' in host
    assert 'id="sidebar-toggle"' in host
    assert "sidebarToggleBtn.addEventListener('click'" in host
    assert "mainEl.classList.toggle('sidebar-collapsed', collapsed)" in host
    assert 'class="theme-switch"' not in host
    assert 'id="refresh-checkpoints-btn"' not in host
