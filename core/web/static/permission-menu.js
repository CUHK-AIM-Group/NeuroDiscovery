(function (root) {
  'use strict';
  const choices = [
    {value:'ask', en:'Ask before acting', zh:'行动前询问'},
    {value:'risk', en:'Act, ask when risky', zh:'直接行动，有风险时询问'},
    {value:'never', en:'Never ask', zh:'从不询问'},
  ];
  function mount({host, getMode, onChange, language, onOpen}) {
    const trigger = host.querySelector('[data-permission-trigger]');
    const menu = host.querySelector('[data-permission-menu]');
    const label = host.querySelector('[data-permission-label]');
    const buttons = choices.map(choice => {
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'permission-option';
      button.dataset.permissionMode = choice.value;
      button.setAttribute('role', 'menuitemradio');
      button.tabIndex = -1;
      const text = document.createElement('span'), check = document.createElement('span');
      check.className = 'permission-check'; check.textContent = '✓'; check.setAttribute('aria-hidden', 'true');
      button.append(text, check); menu.append(button);
      button.onclick = () => { onChange(choice.value); sync(); close(true); };
      return button;
    });
    function sync() {
      const lang = language() === 'zh' ? 'zh' : 'en';
      const selected = choices.find(choice => choice.value === getMode());
      label.textContent = selected ? selected[lang] : (lang === 'zh' ? '只读（旧设置）' : 'Read only (legacy)');
      trigger.setAttribute('aria-label', `${lang === 'zh' ? '执行权限' : 'Permissions'}: ${label.textContent}`);
      trigger.title = label.textContent;
      menu.setAttribute('aria-label', lang === 'zh' ? '执行权限' : 'Permissions');
      buttons.forEach((button, index) => {
        button.firstChild.textContent = choices[index][lang];
        button.setAttribute('aria-checked', String(choices[index].value === getMode()));
      });
    }
    function close(focus = false) {
      menu.hidden = true; trigger.setAttribute('aria-expanded', 'false');
      if (focus) trigger.focus();
    }
    function open(last = false) {
      onOpen?.();
      sync(); menu.hidden = false; trigger.setAttribute('aria-expanded', 'true');
      const selected = buttons.find(button => button.getAttribute('aria-checked') === 'true');
      (last ? buttons.at(-1) : selected || buttons[0]).focus();
    }
    trigger.onclick = () => menu.hidden ? open() : close(true);
    trigger.onkeydown = event => {
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {event.preventDefault();open(event.key === 'ArrowUp');}
    };
    menu.onkeydown = event => {
      const index = buttons.indexOf(document.activeElement);
      if (['ArrowDown','ArrowUp','Home','End'].includes(event.key)) {
        event.preventDefault();
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length-1 : (index + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length;
        buttons[next].focus();
      }
      if (event.key === 'Tab') close(true);
    };
    host.addEventListener('keydown', event => {
      if (event.key === 'Escape' && !menu.hidden) {event.preventDefault();event.stopPropagation();close(true);}
    });
    host.addEventListener('focusout', event => {if (!host.contains(event.relatedTarget)) close();});
    document.addEventListener('pointerdown', event => {if (!host.contains(event.target)) close();});
    sync();
    return {sync, close};
  }
  root.PermissionMenu = Object.freeze({mount});
})(window);
