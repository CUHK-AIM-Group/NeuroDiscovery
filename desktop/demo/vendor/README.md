# Local production renderer dependencies

Unmodified copies of the exact CDN assets referenced by `core/web/static/index.html`.
Electron redirects only these URLs to the local demo server so the production HTML
stays byte-identical and renders Markdown tables and highlighting offline.

- Marked 9: https://cdn.jsdelivr.net/npm/marked@9/marked.min.js (MIT)
- Highlight.js 11.9.0: https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js (BSD-3-Clause)
- Highlight.js GitHub theme 11.9.0: https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css (BSD-3-Clause)

The runtime makes no CDN, provider, or research-graph requests. Dataset-source
links open only after an explicit user click, using the system browser.
