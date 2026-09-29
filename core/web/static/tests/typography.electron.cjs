const {app, BrowserWindow} = require('electron');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8')
  .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, '')
  .replace(/<link\b[^>]*href="\/static\/([^"?]+)[^"]*"[^>]*>/gi,
    (_, asset) => `<style>${fs.readFileSync(path.join(root, asset), 'utf8')}</style>`);
app.whenReady().then(async () => {
  const window = new BrowserWindow({show:false, width:1100, height:800, webPreferences:{sandbox:true}});
  try {
    await window.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(html));
    for (const theme of ['light','dark']) {
      for (const scale of [1,1.3]) {
        const result = await window.webContents.executeJavaScript(`(() => {
          document.documentElement.dataset.theme = '${theme}';
          document.documentElement.dataset.harness = 'true';
          document.body.style.setProperty('--text-scale','${scale}');
          const host = document.createElement('div');
          host.innerHTML = '<div class="msg assistant"><div class="bubble"><p>中文正文 English text</p><code>print(1)</code></div></div><div class="project-chat-item"><span class="chat-title">会话</span></div><span class="model-menu-name">模型</span><span class="model-menu-sub">描述</span><span class="settings-row-label">设置</span><div class="harness-capabilities"><button>知识图谱</button></div>';
          document.body.append(host);
          const read = selector => {const style=getComputedStyle(document.querySelector(selector));return {size:parseFloat(style.fontSize),line:parseFloat(style.lineHeight),font:style.fontFamily};};
          const result = {body:read('.bubble p'),code:read('.bubble code'),input:read('#msg-input'),sidebar:read('.project-chat-item .chat-title'),menu:read('.model-menu-name'),sub:read('.model-menu-sub'),setting:read('.settings-row-label'),nav:read('.harness-capabilities button')};
          host.remove(); return result;
        })()`);
        for (const key of ['body','input']) assert.ok(Math.abs(result[key].size - 16*scale)<0.02, `${key}: ${JSON.stringify(result[key])}`);
        for (const key of ['sidebar','menu','setting','nav']) assert.ok(Math.abs(result[key].size - 14*scale)<0.02, `${key}: ${JSON.stringify(result[key])}`);
        for (const key of ['code','sub']) assert.ok(Math.abs(result[key].size - 13*scale)<0.02, `${key}: ${JSON.stringify(result[key])}`);
        assert.ok(Math.abs(result.body.line-28*scale)<0.02);
        assert.ok(Math.abs(result.input.line-24*scale)<0.02);
        assert.match(result.body.font,/Segoe UI/);
        assert.match(result.code.font,/Consolas/);
      }
    }
    console.log('PASS: Camellia typography, light/dark, default/custom scale; 8 component classes.');
  } catch (error) { console.error(error); process.exitCode=1; }
  finally { window.destroy(); setImmediate(() => app.exit(process.exitCode || 0)); }
});
