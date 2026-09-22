const {app, BrowserWindow} = require('electron');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const staticRoot = path.resolve(__dirname, '..');
const components = [
  ['language', '.language-switch button', '.study-language-option'],
  ['back', '#close-study', '.study-close'],
  ['heading', '.intro h1', '.setup-intro h1'],
  ['eyebrow', '.intro .eyebrow', '.setup-intro .eyebrow'],
  ['step', '.flow span', '.protocol-num'],
  ['label', '#setup-form label', '.setup-form label'],
  ['input', '#setup-form input', '#participant-id'],
  ['select', '#setup-form select', '#participant-experience'],
  ['start', '#start', '#start-btn'],
  ['exportSaved', '#export-saved', '#export-saved'],
  ['advanced', '.restore summary', '.setup-advanced summary'],
  ['save', '#save-progress', '#save-progress'],
  ['closeTitle', '#close-dialog h2', '#study-close-dialog h2'],
  ['closeText', '#close-dialog p', '#study-close-dialog p'],
  ['closeSave', '#close-save', '#close-save-btn'],
  ['discard', '#close-discard', '#close-discard-btn'],
  ['exportTitle', '.evaluation-export-dialog h2', '.evaluation-export-dialog h2'],
  ['exportButton', '.evaluation-export-dialog .primary', '.evaluation-export-dialog .primary'],
];
const properties = ['fontFamily', 'fontSize', 'fontWeight', 'lineHeight', 'color'];
const controls = new Set(['language', 'back', 'input', 'select', 'start', 'exportSaved', 'save', 'closeSave', 'discard', 'exportButton']);

function isolatedPage(filename) {
  return fs.readFileSync(path.join(staticRoot, filename), 'utf8')
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, '')
    .replace(/<link\b[^>]*href="\/static\/([^"?]+)[^"]*"[^>]*>/gi,
      (_, asset) => `<style>${fs.readFileSync(path.join(staticRoot, asset), 'utf8')}</style>`);
}

async function inspect(window, index, theme, scale, embedded, language) {
  return window.webContents.executeJavaScript(`(() => {
    const root = document.documentElement;
    root.dataset.theme = ${JSON.stringify(theme)};
    root.dataset.embedded = ${JSON.stringify(String(embedded))};
    root.lang = ${JSON.stringify(language)};
    root.style.setProperty('--study-text-scale', ${scale});
    document.body.classList.toggle('embedded-route', ${embedded});
    document.querySelector('#close-study')?.removeAttribute('hidden');
    document.querySelector('#auth-view')?.classList.remove('active');
    document.querySelector('#setup-view')?.classList.add('active');
    for (const button of document.querySelectorAll('[data-study-language], [data-discovery-language]')) {
      const active = (button.dataset.studyLanguage || button.dataset.discoveryLanguage) === ${JSON.stringify(language)};
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    }
    const values = {};
    for (const [name, discovery, ranking] of ${JSON.stringify(components)}) {
      const element = document.querySelector(${index} === 0 ? discovery : ranking);
      if (!element) throw new Error('Missing component: ' + name);
      const style = getComputedStyle(element);
      const keys = ${JSON.stringify(properties)};
      if (${JSON.stringify([...controls])}.includes(name)) keys.push('padding', 'borderRadius', 'backgroundColor', 'minHeight');
      values[name] = Object.fromEntries(keys.map(key => [key, style[key]]));
    }
    values.exportSurface = getComputedStyle(document.querySelector('.evaluation-export-dialog')).backgroundColor;
    values.surface = getComputedStyle(root).getPropertyValue('--surface-elev').trim();
    values.overflow = document.documentElement.scrollWidth > innerWidth;
    const input = document.querySelector(${index} === 0 ? '#setup-form input' : '#participant-id');
    input.focus();
    const focus = getComputedStyle(input);
    values.focus = [focus.borderColor, focus.boxShadow, focus.outlineStyle, focus.outlineWidth];
    const save = document.querySelector('#save-progress');
    save.disabled = true;
    values.disabled = [getComputedStyle(save).opacity, getComputedStyle(save).cursor];
    save.disabled = false;
    values.dialogs = [];
    for (const selector of [${index} === 0 ? '#close-dialog' : '#study-close-dialog', '.evaluation-export-dialog']) {
      const dialog = document.querySelector(selector);
      dialog.showModal();
      const rect = dialog.getBoundingClientRect();
      values.dialogs.push({width: rect.width, padding: getComputedStyle(dialog).padding,
        fits: rect.left >= 0 && rect.right <= innerWidth && rect.top >= 0 && rect.bottom <= innerHeight && dialog.scrollWidth <= dialog.clientWidth});
      dialog.close();
    }
    return values;
  })()`);
}

app.whenReady().then(async () => {
  const windows = [];
  try {
    for (const filename of ['discovery-study.html', 'study.html']) {
      const window = new BrowserWindow({show: false, width: 1280, height: 900, webPreferences: {sandbox: true, contextIsolation: true, nodeIntegration: false}});
      windows.push(window);
      window.webContents.session.webRequest.onBeforeRequest((details, callback) => callback({cancel: /^https?:/.test(details.url)}));
      await window.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(isolatedPage(filename)));
      await window.webContents.executeJavaScript(`(() => {
        const modal = document.createElement('dialog');
        modal.className = 'evaluation-export-dialog';
        modal.innerHTML = '<h2>Export results</h2><p>Saved evaluation</p><div class="evaluation-actions"><button class="secondary">Cancel</button><button class="primary">Export JSON</button></div>';
        document.body.append(modal);
        const stable = document.createElement('style');
        stable.textContent = '* { transition: none !important; animation: none !important; }';
        document.head.append(stable);
      })()`);
    }
    let scenarios = 0;
    for (const width of [1280, 820, 390]) {
      for (const window of windows) window.setContentSize(width, 900);
      for (const window of windows) {
        for (let attempt = 0; attempt < 50; attempt++) {
          if (await window.webContents.executeJavaScript('innerWidth') === width) break;
          await new Promise(resolve => setTimeout(resolve, 20));
        }
        assert.equal(await window.webContents.executeJavaScript('innerWidth'), width, 'Renderer viewport resized');
      }
      for (const theme of ['light', 'dark']) {
        for (const scale of [1, 1.5]) {
          for (const embedded of [true, false]) {
            for (const language of ['zh', 'en']) {
            const results = await Promise.all(windows.map((window, index) => inspect(window, index, theme, scale, embedded, language)));
            const context = `${width}px ${theme} scale=${scale} embedded=${embedded} language=${language}`;
            for (const [name] of components) assert.deepEqual(results[0][name], results[1][name], `${context}: ${name}`);
            assert.deepEqual(results[0].focus, results[1].focus, `${context}: focus`);
            assert.deepEqual(results[0].disabled, results[1].disabled, `${context}: disabled`);
            assert.deepEqual(results[0].dialogs, results[1].dialogs, `${context}: dialog layout`);
            for (const result of results) {
              assert.equal(result.overflow, false, `${context}: horizontal overflow`);
              assert.ok(result.dialogs.every(dialog => dialog.fits), `${context}: dialog overflow`);
              if (theme === 'dark') assert.notEqual(result.exportSurface, 'rgb(255, 255, 255)', `${context}: white export dialog`);
            }
            scenarios++;
            }
          }
        }
      }
    }
    console.log(`PASS: ${scenarios} presentation scenarios; ${components.length} paired components; scripts removed and HTTP blocked.`);
  } catch (error) {
    console.error(error);
    process.exitCode = 1;
  } finally {
    for (const window of windows) window.destroy();
    app.exit(process.exitCode || 0);
  }
});
