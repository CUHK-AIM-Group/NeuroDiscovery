/* Small, dependency-free controls shared by the desktop and browser client. */
(() => {
  'use strict';
  const MODES = Object.freeze(['strict', 'novelty_first', 'weighted']);
  const normalize = value => MODES.includes(value) ? value : 'strict';
  const copy = {
    en: {
      title: 'Generation parameters', eyebrow: 'HYPOTHESIS GENERATOR', label: 'Novelty strategy',
      intro: 'Choose how reviewed hypotheses are selected.',
      strict: ['Strict', 'Only evidence-backed novelty. Fewer candidates—or none—is a valid result.'],
      novelty_first: ['Novelty first', 'Prioritize novel candidates, then uncertain exploration, then known replications.'],
      weighted: ['Weighted', 'Balance novelty with structural evidence, GNN scores, and scientific review.'],
      weights: ['Novelty', 'Structure', 'GNN', 'Review'],
      note: 'Selection does not change literature facts. Known relations remain replications, never new discoveries.',
      scope: 'Saved for this chat. Applies to new workflows, not frozen or running experiments.',
      done: 'Done', busy: 'A request is running. Change parameters after it finishes.',
    },
    zh: {
      title: '生成参数', eyebrow: 'HYPOTHESIS GENERATOR', label: '新颖性策略',
      intro: '选择如何筛选已完成科学评审的假设。',
      strict: ['严格', '仅选择有证据支持的新颖候选；宁可少选，也可以不选。'],
      novelty_first: ['新颖优先', '先选新颖候选，不足时依次以不确定探索、已知关系复现补位。'],
      weighted: ['综合加权', '结合新颖性、结构证据、GNN 评分与科学评审排序。'],
      weights: ['新颖性', '结构', 'GNN', '科学评审'],
      note: '选择策略不改变文献事实。已知关系即使入选，也只能算复现，不能算新发现。',
      scope: '按当前对话保存。仅用于新工作流，不修改已冻结或正在运行的实验。',
      done: '完成', busy: '请求正在运行，请在结束后调整参数。',
    },
  };

  function mount({ button, dialog, getMode, onChange, isBusy, language }) {
    const radios = [...dialog.querySelectorAll('input[name="novelty-mode"]')];
    function sync() {
      const c = copy[language() === 'zh' ? 'zh' : 'en'];
      const mode = normalize(getMode());
      const busy = isBusy();
      button.disabled = busy;
      button.title = busy ? c.busy : c.title;
      button.setAttribute('aria-label', `${c.title}: ${c[mode][0]}`);
      button.querySelector('[data-parameter-button-label]').textContent = c.title;
      button.querySelector('[data-parameter-current]').textContent = c[mode][0];
      dialog.querySelectorAll('[data-parameter-copy]').forEach(el => {
        el.textContent = c[el.dataset.parameterCopy];
      });
      radios.forEach(input => {
        input.checked = input.value === mode;
        input.disabled = busy;
        const label = input.closest('label');
        label.querySelector('strong').textContent = c[input.value][0];
        label.querySelector('[data-mode-description]').textContent = c[input.value][1];
      });
      dialog.querySelectorAll('[data-weight-label]').forEach((el, i) => { el.textContent = c.weights[i]; });
      dialog.querySelector('.novelty-weights').hidden = mode !== 'weighted';
      if (busy && dialog.open) dialog.close();
    }
    button.addEventListener('click', () => {
      if (isBusy()) return;
      sync();
      dialog.showModal();
      button.setAttribute('aria-expanded', 'true');
      radios.find(input => input.checked)?.focus();
    });
    radios.forEach(input => input.addEventListener('change', () => {
      if (input.checked && !isBusy()) onChange(input.value);
      sync();
    }));
    dialog.querySelector('[data-parameter-done]').addEventListener('click', () => dialog.close());
    dialog.addEventListener('close', () => button.setAttribute('aria-expanded', 'false'));
    dialog.addEventListener('click', event => {
      if (event.target !== dialog) return;
      const rect = dialog.getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
    });
    sync();
    return { sync };
  }
  window.HypothesisControls = Object.freeze({ MODES, normalize, mount });
})();
