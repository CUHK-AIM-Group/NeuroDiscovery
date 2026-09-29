/* Shared SVG export and interactive view; never executes a graph query. */
(function(root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.NeuroEvidenceGraph = factory();
})(typeof window === 'object' ? window : globalThis, function() {
  'use strict';
  const kind = 'neurodiscovery-evidence-graph';
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function valid(g) {
    if (g?.kind !== kind || !Array.isArray(g.nodes) || !g.nodes.length || g.nodes.length > 200 || !Array.isArray(g.edges) || g.edges.length > 500) return false;
    const text = s => typeof s === 'string' && s.length > 0 && s.length <= 200;
    const point = n => Number.isFinite(n) && n >= 0 && n <= 1000;
    if(!text(g.title)||!g.nodes.every(n=>n && text(n.id) && text(n.name) && text(n.type) && /^#[a-f\d]{6}$/i.test(n.color) && point(n.x) && point(n.y) && Number.isFinite(n.r) && n.r>=8 && n.r<=50))return false;
    const ids = new Set(g.nodes.map(n => n.id));
    return ids.size === g.nodes.length &&
      Array.isArray(g.routes) && g.routes.length === g.edges.length && g.routes.every(p=>Array.isArray(p)&&p.length===4&&p.every(point)) &&
      g.edges.every(e => Array.isArray(e) && e.length===3 && ids.has(e[0]) && ids.has(e[1]) && text(e[2]));
  }
  function renderSvg(g, {id = 'evidence', interactive = false} = {}) {
    if (!valid(g)) throw Error('Invalid evidence graph');
    const en = g.language === 'en';
    const nodes = new Map(g.nodes.map(n => [n.id,n]));
    const edges = g.edges.map(([from,to,label], i) => {
      const [cx,cy,lx,ly] = g.routes[i];
      const a = nodes.get(from), b = nodes.get(to), secondary = ['研究条件','影像测量','Study conditions','Imaging measure'].includes(a.type);
      return `<g class="graph-edge" data-source="${esc(from)}" data-target="${esc(to)}">
        <path d="M${a.x} ${a.y}Q${cx} ${cy} ${b.x} ${b.y}" fill="none" stroke="var(--graph-line,#bbc9df)" stroke-width="1.6" ${secondary?'stroke-dasharray="5 5"':''}/>
        <text class="edge-label" x="${lx}" y="${ly}" text-anchor="middle" font-size="13" fill="var(--graph-muted,#6b7d96)" paint-order="stroke" stroke="var(--graph-bg,#f8faff)" stroke-width="7" stroke-linejoin="round">${esc(label)}</text>
      </g>`;
    }).join('');
    const circles = g.nodes.map(n => `<g class="graph-node" data-node="${esc(n.id)}" transform="translate(${n.x} ${n.y})" ${interactive?`tabindex="0" role="button" aria-label="${esc(n.name+' · '+n.type)}" aria-pressed="false"`:''}>
      <title>${esc(n.name+' · '+n.type)}</title>
      <circle class="node-halo" r="${n.r+10}" fill="${n.color}" opacity=".08"/>
      <circle class="node-ring" r="${n.r+4}" fill="none" stroke="${n.color}" stroke-opacity=".25"/>
      <circle class="node-core" r="${n.r}" fill="${n.color}" stroke="var(--graph-paper,#fff)" stroke-width="2.5"/>
      <circle r="${n.r>25?6:4}" fill="#fff" opacity=".95"/>
      <text class="node-name" y="${n.r+31}" text-anchor="middle" font-size="18" font-weight="600" fill="var(--graph-ink,#24344d)" paint-order="stroke" stroke="var(--graph-bg,#f8faff)" stroke-width="7" stroke-linejoin="round">${esc(n.name)}</text>
      <text class="node-type" y="${n.r+51}" text-anchor="middle" font-size="12" fill="var(--graph-muted,#6b7d96)">${esc(n.type)}</text>
    </g>`).join('');
    const legend = g.nodes.map((n,i) => `<g class="graph-legend-item" transform="translate(${34+i*153} 566)"><circle r="4" fill="${n.color}"/><text class="graph-legend-label" x="12" y="4" font-size="12" fill="var(--graph-muted,#6b7d96)">${esc(n.type)}</text></g>`).join('');
    return `<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="600" viewBox="0 0 1000 600" role="${interactive?'group':'img'}" aria-labelledby="${id}-title ${id}-desc">
      <title id="${id}-title">${esc(g.title)}</title><desc id="${id}-desc">${en?'Nodes represent topic concepts. Edges retain study relationships and conditions. Associations do not establish causality.':'节点表示主题概念，连线保留研究关系与条件。关联线索不等于因果关系。'}</desc>
      <metadata>${esc(JSON.stringify(g))}</metadata>
      <defs><pattern id="${id}-dots" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".7" fill="var(--graph-grid,#cdd8ec)"/></pattern><clipPath id="${id}-clip"><rect x="0" y="86" width="1000" height="450"/></clipPath></defs>
      <rect width="1000" height="600" rx="16" fill="var(--graph-paper,#fff)"/>
      <g font-family="Segoe UI,Microsoft YaHei,PingFang SC,sans-serif">
        <text class="graph-title" x="30" y="36" font-size="20" font-weight="600" fill="var(--graph-ink,#24344d)">${en?'Topic evidence subgraph':'主题证据子图'}</text>
        <text class="graph-subtitle" x="30" y="63" font-size="14" fill="var(--graph-muted,#6b7d96)">${esc(g.title)}</text>
        <text class="graph-count" x="968" y="38" text-anchor="end" font-size="12" fill="var(--graph-muted,#6b7d96)">${en?`${g.nodes.length} nodes · ${g.edges.length} relations`:`${g.nodes.length} 个节点 · ${g.edges.length} 条关系`}</text>
        <rect y="86" width="1000" height="450" fill="var(--graph-bg,#f8faff)"/><rect y="86" width="1000" height="450" fill="url(#${id}-dots)" opacity=".55"/>
        <g clip-path="url(#${id}-clip)"><g class="graph-viewport">${edges}${circles}</g></g>
        <path d="M0 86H1000M0 536H1000" stroke="var(--graph-border,#e6ecf5)"/>
        ${legend}
      </g>
    </svg>`;
  }
  function readSvg(text) {
    if (text.length > 512*1024) return null;
    try { const doc = new DOMParser().parseFromString(text,'image/svg+xml'); const g = JSON.parse(doc.querySelector('svg > metadata')?.textContent || 'null'); return valid(g) ? g : null; } catch (_) { return null; }
  }
  let sequence = 0;
  function create(g, {source = '', image = null} = {}) {
    const doc = document, zh = doc.documentElement.lang.startsWith('zh'), t = (en,cn) => zh ? cn : en;
    const el = (tag,cls,text) => {const n=doc.createElement(tag);n.className=cls;if(text!==undefined)n.textContent=text;return n;};
    const figure = el('figure','evidence-graph'); figure.dataset.graphSource = source;
    if (image) {image.classList.add('graph-fallback');figure.append(image);}
    const stage=el('div','graph-stage');stage.innerHTML=renderSvg(g,{id:'evidence-'+(++sequence),interactive:true});
    const toolbar=el('div','graph-toolbar');toolbar.setAttribute('role','group');toolbar.ariaLabel=t('Graph controls','图谱操作');
    const details=el('div','graph-details');details.setAttribute('aria-live','polite');
    const svg=stage.querySelector('svg'),viewport=stage.querySelector('.graph-viewport');
    let zoom=1,dx=0,dy=0,selected=null,drag=null;
    const buttons={};
    function button(name,title,path,fn) {const b=el('button','graph-action');b.type='button';b.title=b.ariaLabel=title;b.dataset.graphAction=name;b.innerHTML='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="'+path+'"/></svg>';b.addEventListener('click',fn);toolbar.append(b);buttons[name]=b;return b;}
    function transform() {viewport.setAttribute('transform',`translate(${dx} ${dy}) scale(${zoom})`);figure.dataset.zoom=String(zoom);buttons.out.disabled=zoom<=.6;buttons.in.disabled=zoom>=3;}
    function scale(next,x=500,y=310) {const k=Math.max(.6,Math.min(3,next));dx=x-(x-dx)*k/zoom;dy=y-(y-dy)*k/zoom;zoom=k;transform();}
    button('out',t('Zoom out','缩小'),'M5 12h14',()=>scale(zoom/1.25));
    button('in',t('Zoom in','放大'),'M5 12h14M12 5v14',()=>scale(zoom*1.25));
    button('fit',t('Fit graph','适应画布'),'M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5',()=>{zoom=1;dx=dy=0;transform();select(null);});
    if(source){const open=el('a','graph-action');open.href=source;open.title=open.ariaLabel=t('Open graph artifact','在产物中打开图谱');open.innerHTML='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M14 3h7v7m0-7L10 14M10 3H3v18h18v-7"/></svg>';toolbar.append(open);}
    function select(id) {
      selected=id;const nearby=new Set(id?[id]:[]),relations=g.edges.filter(e=>e[0]===id||e[1]===id);
      for(const e of relations){nearby.add(e[0]);nearby.add(e[1]);}
      stage.querySelectorAll('.graph-node').forEach(n=>{n.classList.toggle('is-dimmed',!!id&&!nearby.has(n.dataset.node));n.classList.toggle('is-selected',n.dataset.node===id);n.setAttribute('aria-pressed',String(n.dataset.node===id));});
      stage.querySelectorAll('.graph-edge').forEach(n=>{const match=n.dataset.source===id||n.dataset.target===id;n.classList.toggle('is-dimmed',!!id&&!match);n.classList.toggle('is-selected',!!id&&match);});
      details.replaceChildren();
      if(!id){details.append(el('span','graph-hint',t('Select a node to inspect relations · Drag the canvas to pan','点击节点查看关系 · 拖动画布平移')));return;}
      const node=g.nodes.find(n=>n.id===id),name=el('strong','graph-detail-name',node.name),type=el('span','graph-detail-type',node.type);
      const head=el('div','graph-detail-heading');head.append(name,type);details.append(head);
      for(const [from,to,relation] of relations){const other=g.nodes.find(n=>n.id===(from===id?to:from));const row=el('button','graph-relation');row.type='button';row.append(el('span','',relation),el('span','',other.name));row.addEventListener('click',()=>select(other.id));details.append(row);}
    }
    svg.addEventListener('click',event=>{if(drag?.moved)return;const node=event.target.closest('.graph-node');if(node)select(selected===node.dataset.node?null:node.dataset.node);});
    svg.addEventListener('keydown',event=>{const node=event.target.closest('.graph-node');if(node&&['Enter',' '].includes(event.key)){event.preventDefault();select(node.dataset.node);}if(event.key==='Escape')select(null);});
    const point=e=>new DOMPoint(e.clientX,e.clientY).matrixTransform(svg.getScreenCTM().inverse());
    svg.addEventListener('pointerdown',event=>{if(event.button!==0||event.target.closest('.graph-node'))return;const p=point(event);drag={x:p.x,y:p.y,dx,dy,moved:false};svg.setPointerCapture(event.pointerId);});
    svg.addEventListener('pointermove',event=>{if(!drag||!svg.hasPointerCapture(event.pointerId))return;const p=point(event);if(Math.hypot(p.x-drag.x,p.y-drag.y)>3)drag.moved=true;dx=Math.max(-2000,Math.min(2000,drag.dx+p.x-drag.x));dy=Math.max(-1200,Math.min(1200,drag.dy+p.y-drag.y));transform();});
    svg.addEventListener('pointerup',event=>{if(svg.hasPointerCapture(event.pointerId))svg.releasePointerCapture(event.pointerId);setTimeout(()=>{drag=null;},0);});
    svg.addEventListener('pointercancel',()=>{drag=null;});
    svg.addEventListener('wheel',event=>{if(!event.ctrlKey&&!event.metaKey)return;event.preventDefault();const p=point(event);scale(zoom*(event.deltaY>0?.9:1.1),p.x,p.y);},{passive:false});
    // A narrow sidebar must not shrink labels into unreadable image captions.
    if(typeof ResizeObserver!=='undefined')new ResizeObserver(entries=>{
      const width=entries[0].contentRect.width;if(!width)return;const scale=width/1000,compact=width<600;
      for(const [selector,base,min] of [['.node-name',18,12],['.edge-label',13,10],['.graph-title',20,12],['.graph-subtitle',14,10],['.graph-count',12,9],['.graph-legend-label',12,9]])
        stage.querySelectorAll(selector).forEach(n=>n.setAttribute('font-size',Math.max(base,min/scale)));
      stage.querySelectorAll('.node-type').forEach(n=>n.style.display=compact?'none':'');
      stage.querySelectorAll('.graph-legend-item').forEach((n,i)=>n.setAttribute('transform',compact?`translate(${34+(i%3)*310} ${554+Math.floor(i/3)*27})`:`translate(${34+i*153} 566)`));
    }).observe(stage);
    stage.append(toolbar);figure.append(stage,details);transform();select(null);return figure;
  }
  const cache=new Map();
  function capture(root) {return new Map([...root.querySelectorAll('.evidence-graph')].map(n=>[n.dataset.graphSource,n]));}
  function enhance(root, retained=new Map()) {
    root.querySelectorAll('img[src]').forEach(async img=>{
      if(img.closest('.evidence-graph'))return;
      const source=img.getAttribute('src');let url;
      try {url=new URL(source,location.href);}catch(_){return;}
      if(url.origin!==location.origin||!(/\/evidence\.svg$/.test(url.pathname)||url.pathname==='/api/workbench/artifact'&&/\.svg$/i.test(url.searchParams.get('path')||'')))return;
      const previous=retained.get(source);if(previous){img.replaceWith(previous);return;}
      if(!cache.has(source)){
        if(cache.size>=64)cache.delete(cache.keys().next().value);
        cache.set(source,fetch(url,{redirect:'error'}).then(async r=>{if(!r.ok||Number(r.headers.get('content-length'))>512*1024)return null;const reader=r.body.getReader(),chunks=[];let length=0;try{while(true){const {done,value}=await reader.read();if(done)break;length+=value.length;if(length>512*1024)return null;chunks.push(value);}return readSvg(await new Blob(chunks).text());}finally{await reader.cancel().catch(()=>{});}}).catch(()=>null));
      }
      const graph=await cache.get(source);if(!graph||!root.contains(img))return;
      const marker=document.createComment('evidence graph');img.replaceWith(marker);
      marker.replaceWith(create(graph,{source,image:img}));
    });
  }
  return {kind,valid,renderSvg,readSvg,create,capture,enhance};
});
