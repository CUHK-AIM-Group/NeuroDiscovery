'use strict';
const frame = document.querySelector('#study-frame');
const home = document.querySelector('#home');
document.querySelectorAll('[data-study]').forEach(button => button.addEventListener('click', () => {
  frame.src = `/${button.dataset.study}?embedded=1`;
  frame.hidden = false;
  home.hidden = true;
}));
window.addEventListener('message', event => {
  if (event.origin !== location.origin || event.source !== frame.contentWindow) return;
  if (event.data?.type === 'neurodiscovery:close-study-workspace') {
    frame.hidden = true;
    frame.removeAttribute('src');
    home.hidden = false;
  }
});
