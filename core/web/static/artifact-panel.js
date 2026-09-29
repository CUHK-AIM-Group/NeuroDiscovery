/* Shared, dependency-free artifact workspace. Both desktop transports use this UI. */
(function(root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.NeuroArtifacts = factory();
})(typeof window === 'object' ? window : globalThis, function() {
  'use strict';
  const TEXT_LIMIT = 2 * 1024 * 1024, BINARY_LIMIT = 32 * 1024 * 1024;
  const TYPES = {
    markdown: ['md','markdown','mdown'], table: ['csv','tsv'], json: ['json'],
    html: ['html','htm'], image: ['png','jpg','jpeg','gif','webp','svg','bmp','avif'], pdf: ['pdf'],
    text: ['txt','log','py','js','cjs','mjs','ts','tsx','jsx','css','scss','yaml','yml','toml','ini','xml','jsonl','ndjson','r','rmd','sql','sh','ps1','bat','c','cpp','h','rs','go','java','m','tex','bib','rst','ipynb'],
  };
  function extension(name) { return String(name).split(/[?#]/)[0].split('.').pop().toLowerCase(); }
  function formatSize(bytes) {
    if(!Number.isSafeInteger(bytes)||bytes<0)return '';
    if(bytes<1024)return bytes+' B';
    const power=Math.min(3,Math.floor(Math.log(bytes)/Math.log(1024))),amount=bytes/(1024**power);
    return (amount<10?Number(amount.toFixed(1)):Math.round(amount))+' '+['B','KB','MB','GB'][power];
  }
  function fileKind(name) { const ext=extension(name); return Object.keys(TYPES).find(kind=>TYPES[kind].includes(ext)) || 'file'; }
  function fileReference(raw, {origin, chatId}={}) {
    if (!raw || !chatId) return null;
    let value=String(raw).trim();
    if (value.startsWith('#') || value.startsWith('//') || value.startsWith('\\\\')) return null;
    if (/^https?:/i.test(value)) {
      try { const url=new URL(value); if(url.origin!==origin) return null; value=url.pathname+url.search; } catch (_) { return null; }
    }
    if (/^[a-z][\w+.-]*:/i.test(value) && !/^[a-z]:(?:[\\/]|%2f|%5c)/i.test(value)) return null;
    let name, url, reference;
    if (/^\/(?:demo\/artifacts\/|api\/workbench\/artifact\?)/.test(value)) {
      const parsed=new URL(value,origin);
      if(parsed.pathname==='/api/workbench/artifact' && parsed.searchParams.get('chat_id')!==chatId) return null;
      reference=parsed.searchParams.get('path');
      try { name=(reference || decodeURIComponent(parsed.pathname)).split(/[\\/]/).pop(); } catch (_) { return null; }
      url=parsed.pathname+parsed.search;
    } else {
      try { value=decodeURIComponent(value); } catch (_) { return null; }
      if (/[\x00-\x1f<>|?*]/.test(value) || value.length>4096 || !/\.[a-z0-9]{1,12}$/i.test(value)) return null;
      if (value.startsWith('/api/') || value.startsWith('/static/')) return null;
      reference=value; name=value.split(/[\\/]/).pop();
      url='/api/workbench/artifact?'+new URLSearchParams({chat_id:chatId,path:value});
    }
    if (!name || !/\.[a-z0-9]{1,12}$/i.test(name)) return null;
    return {key:url, url, name, reference, kind:fileKind(name), ext:extension(name)};
  }
  // RFC-style quoted fields, BOM, CRLF, multiline cells, and literal formula strings.
  function parseDelimited(source, delimiter=',', maxRows=20000, maxColumns=200) {
    const text=String(source).replace(/^\uFEFF/,'');
    const rows=[]; let row=[],cell='',quoted=false,atStart=true,warning=false,truncated=false;
    const field=()=>{ if(row.length<maxColumns)row.push(cell);else truncated=true; cell='';atStart=true; };
    const record=()=>{field();rows.push(row);row=[];};
    for(let i=0;i<text.length;i++) {
      const char=text[i];
      if(quoted) { if(char==='"') { if(text[i+1]==='"'){cell+='"';i++;}else quoted=false; } else cell+=char; }
      else if(char==='"' && atStart){quoted=true;atStart=false;}
      else if(char===delimiter)field();
      else if(char==='\n'||char==='\r'){
        if(char==='\r'&&text[i+1]==='\n')i++;
        record();if(rows.length>=maxRows+1){truncated=i<text.length-1;break;}
      } else {cell+=char;atStart=false;}
    }
    warning=quoted;
    if(rows.length<maxRows+1 && (cell!=='' || row.length || text.endsWith(delimiter))) record();
    return {rows,warning,truncated};
  }
  function escapeHtml(value) { return String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
  function staticHtml(source) {
    // Opaque-origin, scriptless document. Neither linked resources nor forms can leave it.
    return '<!doctype html><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\'; font-src data:; base-uri \'none\'; form-action \'none\'; frame-src \'none\'"><style>body{font:15px/1.7 system-ui;margin:24px;overflow-wrap:anywhere}img{max-width:100%}</style>'+source;
  }
  function mount({document:doc, getContext}) {
    const win=doc.defaultView, app=doc.getElementById('app'), main=doc.querySelector('.main'), messages=doc.getElementById('messages');
    if(!main || !messages) return null;
    const el=(tag,cls,text)=>{const node=doc.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=text;return node;};
    const button=(cls,text,handler)=>{const node=el('button',cls,text);node.type='button';if(handler)node.addEventListener('click',handler);return node;};
    let context={},files=[],selected=null,opened=false,tab='list',sourceMode=false,signature='',frame=0,controller=null,objectUrl=null,lastFocus=null,loadId=0,content=null;
    const extraFiles=new Map();
    const turnCards=new WeakMap(),sizes=new Map(),sizeQueue=[];
    let sizeReads=0,menuAnchor=null;
    const fileMenu=el('div','output-file-menu');fileMenu.hidden=true;fileMenu.id='output-file-menu';fileMenu.setAttribute('role','menu');doc.body.append(fileMenu);
    const setText=(node,value)=>{if(node.textContent!==value)node.textContent=value;};
    function closeFileMenu(focus=false){
      const anchor=menuAnchor;menuAnchor=null;fileMenu.hidden=true;anchor?.setAttribute('aria-expanded','false');
      if(focus&&anchor?.isConnected)anchor.focus({preventScroll:true});
    }
    function placeFileMenu(){
      if(!menuAnchor?.isConnected){closeFileMenu();return;}
      const rect=menuAnchor.getBoundingClientRect(),width=fileMenu.offsetWidth,height=fileMenu.offsetHeight;
      fileMenu.style.left=Math.max(8,Math.min(win.innerWidth-width-8,rect.right-width))+'px';
      fileMenu.style.top=Math.max(8,Math.min(win.innerHeight-height-8,rect.bottom+height+6<win.innerHeight?rect.bottom+6:rect.top-height-6))+'px';
    }
    function downloadFile(file){const link=el('a');link.href=file.url+(file.url.includes('?')?'&':'?')+'download=1';link.download=file.name;doc.body.append(link);link.click();link.remove();}
    function showFileMenu(anchor,file){
      if(menuAnchor===anchor){closeFileMenu(true);return;}closeFileMenu();menuAnchor=anchor;
      fileMenu.replaceChildren();fileMenu.setAttribute('aria-label',t('Open '+file.name,'打开 '+file.name));
      const add=(action,label,run)=>{const item=button('output-menu-item',label,()=>{closeFileMenu(true);run();});item.dataset.fileAction=action;item.setAttribute('role','menuitem');fileMenu.append(item);};
      add('preview',t('Open in NeuroDiscovery','在 NeuroDiscovery 中打开'),()=>openFile(file));
      if(['markdown','table','json','html','text'].includes(file.kind))add('source',t('View source','查看原文'),()=>openFile(file,true,true));
      add('download',t('Download file','下载文件'),()=>downloadFile(file));
      fileMenu.hidden=false;anchor.setAttribute('aria-expanded','true');placeFileMenu();fileMenu.firstElementChild.focus({preventScroll:true});
    }
    doc.addEventListener('pointerdown',event=>{if(menuAnchor&&!fileMenu.contains(event.target)&&!menuAnchor.contains(event.target))closeFileMenu();});
    doc.addEventListener('keydown',event=>{
      if(!menuAnchor)return;
      if(event.key==='Escape'){event.preventDefault();event.stopImmediatePropagation();closeFileMenu(true);return;}
      if(event.key==='Tab'){closeFileMenu(true);return;}
      if(!['ArrowDown','ArrowUp','Home','End'].includes(event.key))return;
      event.preventDefault();const items=[...fileMenu.querySelectorAll('[role="menuitem"]')],index=items.indexOf(doc.activeElement);
      items[event.key==='Home'?0:event.key==='End'?items.length-1:(index+(event.key==='ArrowDown'?1:-1)+items.length)%items.length]?.focus();
    });
    win.addEventListener('resize',()=>menuAnchor&&placeFileMenu());
    doc.addEventListener('scroll',()=>menuAnchor&&placeFileMenu(),true);
    function readSizes(){
      while(sizeReads<4&&sizeQueue.length){
        const file=sizeQueue.shift();sizeReads++;
        win.fetch(file.url,{method:'HEAD',redirect:'error'}).then(response=>{
          const raw=response.headers.get('content-length'),bytes=raw!==null&&/^\d+$/.test(raw)?Number(raw):null;
          sizes.set(file.key,{bytes:response.ok&&Number.isSafeInteger(bytes)?bytes:null});
        }).catch(()=>sizes.set(file.key,{bytes:null})).finally(()=>{sizeReads--;sync();readSizes();});
      }
    }
    function fileCaption(file){
      const label=file.kind==='image'?t('Image','图像'):file.kind==='table'?t('Spreadsheet','表格'):file.kind==='pdf'||file.kind==='file'?t('Document','文档'):t('Text','文本');
      return [label,file.ext.toUpperCase(),formatSize(sizes.get(file.key)?.bytes)].filter(Boolean).join(' · ');
    }
    function renderTurnCards(bubble,turn){
      // Only standalone output-file lists become cards; citations and inline links stay in the prose.
      const sources=[],output=new Map();
      for(const source of bubble.querySelectorAll('ul,ol,p')){
        if(source.closest('.turn-artifacts,.evidence-graph,pre,blockquote,table,li'))continue;
        const lines=source.tagName==='P'?[source]:[...source.children];
        if(!lines.length)continue;
        const entries=lines.map(line=>{
          const links=line.querySelectorAll('a[href]');if(links.length!==1||line.textContent.trim()!==links[0].textContent.trim())return null;
          const file=fileReference(links[0].getAttribute('href'),{origin:win.location.origin,chatId:context.chatId});
          if(!file||links[0].textContent.trim()!==file.name)return null;
          return {...file,turn};
        });
        if(entries.some(file=>!file))continue;
        sources.push(source);for(const file of entries)output.set(file.key,file);
      }
      for(const source of bubble.querySelectorAll('[data-output-file-source]'))if(!sources.includes(source)){source.hidden=false;delete source.dataset.outputFileSource;}
      let record=turnCards.get(bubble);
      if(!output.size){record?.box.remove();turnCards.delete(bubble);return;}
      if(!record){record={box:el('div','turn-artifacts'),rows:new Map(),stamp:''};record.box.setAttribute('role','group');turnCards.set(bubble,record);}
      record.box.setAttribute('aria-label',t('Files from this reply','本次回复的输出文件'));
      for(const file of output.values()){
        if(!sizes.has(file.key)){if(sizes.size>=512)sizes.delete(sizes.keys().next().value);sizes.set(file.key,{bytes:null});sizeQueue.push(file);}
        let row=record.rows.get(file.key);
        if(!row){
          row=el('div','output-file-row');row.dataset.outputKey=file.key;
          const open=button('output-file','',()=>openFile(row.file));open.title=file.reference||file.name;
          const icon=el('span','output-file-icon '+(file.kind==='table'?'is-table':file.kind==='pdf'?'is-pdf':''),file.ext.toUpperCase().slice(0,5));icon.setAttribute('aria-hidden','true');
          const info=el('span','output-file-info'),name=el('strong','output-file-name',file.name),meta=el('span','output-file-meta');info.append(name,meta);open.append(icon,info);
          const menu=button('output-file-open','',()=>showFileMenu(menu,row.file));menu.setAttribute('aria-haspopup','menu');menu.setAttribute('aria-expanded','false');menu.setAttribute('aria-controls',fileMenu.id);
          menu.addEventListener('keydown',event=>{if(event.key==='ArrowDown'){event.preventDefault();showFileMenu(menu,row.file);}});
          row.append(open,menu);record.rows.set(file.key,row);
        }
        row.file=file;setText(row.querySelector('.output-file-meta'),fileCaption(file));
        setText(row.querySelector('.output-file-open'),t('Open with','打开方式'));
      }
      const stamp=[...output.keys()].join('\n')+'|'+context.zh;
      if(stamp!==record.stamp){
        const expanded=record.box.querySelector('details')?.open;record.stamp=stamp;
        record.box.replaceChildren();const rows=[...output.keys()].map(key=>record.rows.get(key));record.box.append(...rows.slice(0,4));
        if(rows.length>4){
          const more=el('details','output-file-overflow'),summary=el('summary');more.open=!!expanded;
          summary.append(el('span','output-show-more',t(`Show ${rows.length-4} more files`,`展开其余 ${rows.length-4} 个文件`)),el('span','output-show-less',t('Show fewer files','收起文件')));
          more.append(summary,...rows.slice(4));record.box.append(more);
        }
        for(const key of record.rows.keys())if(!output.has(key))record.rows.delete(key);
      }
      for(const source of sources){source.hidden=true;source.dataset.outputFileSource='true';}
      const last=sources.at(-1);if(last.nextElementSibling!==record.box)last.after(record.box);
      readSizes();
    }
    let requestedWidth=460;
    try {requestedWidth=Number(win.localStorage.getItem('neurodiscovery.artifact-width')) || 460;} catch (_) {}
    const t=(en,zh)=>context.zh?zh:en;
    const toggle=button('artifact-toggle','',()=>{opened?close():show('list');});toggle.id='artifact-toggle';toggle.setAttribute('aria-controls','artifact-panel');
    toggle.innerHTML='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 3h10l6 6v12H4zM14 3v6h6M8 13h8M8 17h6"/></svg>';
    const toggleLabel=el('span','artifact-toggle-label'),count=el('span','artifact-count','0');toggle.append(toggleLabel,count);
    doc.querySelector('#app > header').insertBefore(toggle,doc.getElementById('desktop-menu-btn'));
    const panel=el('aside','artifact-panel');panel.id='artifact-panel';panel.hidden=true;
    const resizer=el('div','artifact-resizer');resizer.tabIndex=0;resizer.setAttribute('role','separator');resizer.setAttribute('aria-orientation','vertical');
    const head=el('div','artifact-head');head.setAttribute('role','tablist');
    const listTab=button('artifact-tab','',()=>show('list')),previewTab=button('artifact-tab','',()=>show('preview')),closeBtn=button('artifact-close','×',close);
    listTab.id='artifact-list-tab';previewTab.id='artifact-preview-tab';closeBtn.id='artifact-close';
    for(const node of [listTab,previewTab])node.setAttribute('role','tab');
    head.append(listTab,previewTab,closeBtn);
    const listPane=el('section','artifact-list-pane');listPane.id='artifact-list-pane';listPane.setAttribute('role','tabpanel');listPane.setAttribute('aria-labelledby',listTab.id);listTab.setAttribute('aria-controls',listPane.id);
    const listTools=el('div','artifact-list-tools'),search=el('input','artifact-search'),summary=el('p','artifact-list-summary'),list=el('div','artifact-list');
    search.type='search';search.id='artifact-search';search.addEventListener('input',renderList);listTools.append(search,summary);listPane.append(listTools,list);
    const previewPane=el('section','artifact-preview-pane');previewPane.id='artifact-preview-pane';previewPane.setAttribute('role','tabpanel');previewPane.setAttribute('aria-labelledby',previewTab.id);previewTab.setAttribute('aria-controls',previewPane.id);
    const filebar=el('div','artifact-filebar'),filename=el('h2','artifact-filename'),actions=el('div','artifact-actions');
    const viewBtn=button('artifact-action','',()=>{sourceMode=false;renderContent();}),sourceBtn=button('artifact-action','',()=>{sourceMode=true;renderContent();});
    const reloadBtn=button('artifact-action','',()=>selected&&openFile(selected,false));
    const download=el('a','artifact-action artifact-download');download.id='artifact-download';
    actions.append(viewBtn,sourceBtn,reloadBtn,download);filebar.append(filename,actions);
    const body=el('div','artifact-content');body.id='artifact-content';body.setAttribute('aria-live','polite');previewPane.append(filebar,body);
    panel.append(resizer,head,listPane,previewPane);main.append(panel);
    function clearLoad(){loadId++;controller?.abort();controller=null;content=null;if(objectUrl){win.URL.revokeObjectURL(objectUrl);objectUrl=null;}}
    function fitWidth(){
      const docked=win.innerWidth>1200, sidebar=doc.body.classList.contains('sidebar-collapsed')?0:doc.getElementById('app-sidebar')?.getBoundingClientRect().width || 0;
      const max=Math.max(340,Math.min(760,(main.clientWidth||app.clientWidth)-(docked?sidebar+500:24)));
      const width=Math.max(340,Math.min(max,requestedWidth));app.style.setProperty('--artifact-width',width+'px');
      resizer.setAttribute('aria-valuemin','340');resizer.setAttribute('aria-valuemax',String(max));resizer.setAttribute('aria-valuenow',String(Math.round(width)));
    }
    function chrome(){
      const visible=context.view==='chat';toggle.hidden=!visible;panel.hidden=!opened||!visible;
      main.classList.toggle('has-artifact-panel',!panel.hidden);toggle.setAttribute('aria-expanded',String(!panel.hidden));
      app.classList.toggle('has-artifact-panel',!panel.hidden);
      toggleLabel.textContent=t('Artifacts','产物');count.textContent=String(files.length);toggle.title=t('Conversation artifacts','当前对话的产物');
      panel.setAttribute('aria-label',t('Artifacts and file preview','产物与文件预览'));resizer.setAttribute('aria-label',t('Resize preview sidebar','调整预览栏宽度'));
      listTab.textContent=t('Artifacts','产物')+' · '+files.length;previewTab.textContent=t('Preview','预览');previewTab.disabled=!selected;
      listTab.setAttribute('aria-selected',String(tab==='list'));previewTab.setAttribute('aria-selected',String(tab==='preview'));
      listTab.tabIndex=tab==='list'?0:-1;previewTab.tabIndex=tab==='preview'?0:-1;
      listPane.hidden=tab!=='list';previewPane.hidden=tab!=='preview';
      closeBtn.title=t('Close sidebar (Esc)','关闭侧栏（Esc）');closeBtn.setAttribute('aria-label',closeBtn.title);
      search.placeholder=t('Find a file…','搜索产物…');search.setAttribute('aria-label',search.placeholder);
      viewBtn.textContent=t('Preview','预览');sourceBtn.textContent=t('Source','原文');reloadBtn.textContent=t('Reload','重新加载');download.textContent=t('Download','下载');
      fitWidth();
    }
    function renderList(){
      const query=search.value.trim().toLowerCase(),matched=files.filter(f=>(f.name+' '+(f.reference||'')).toLowerCase().includes(query));
      summary.textContent=query?t(`${matched.length} of ${files.length} files`,`${files.length} 个产物，找到 ${matched.length} 个`):t(`${files.length} files in this conversation`, `当前对话共 ${files.length} 个产物`);
      list.replaceChildren();
      if(!matched.length){list.append(el('p','artifact-empty',files.length?t('No matching files.','没有找到匹配的产物。'):t('Files produced in this conversation will appear here. Select a file to preview it.','此对话中生成的文件会显示在这里。点击文件即可预览。')));return;}
      for(const file of matched){
        const row=button('artifact-file','',()=>openFile(file));row.dataset.artifactKey=file.key;row.title=file.reference||file.name;row.setAttribute('aria-current',String(file.key===selected?.key));
        const copy=el('span','artifact-file-copy');copy.append(el('span','artifact-file-name',file.name),el('span','artifact-file-detail',t(`Reply ${file.turn} · ${file.ext.toUpperCase()}`,`回复 ${file.turn} · ${file.ext.toUpperCase()}`)));
        row.append(el('span','artifact-file-icon',file.ext.slice(0,5)),copy,el('span','artifact-file-arrow','›'));list.append(row);
      }
    }
    function show(nextTab){
      if(!opened)lastFocus=doc.activeElement;opened=true;tab=nextTab==='preview'&&selected?'preview':'list';chrome();
      if(tab==='list'){renderList();search.focus({preventScroll:true});}else previewTab.focus({preventScroll:true});
    }
    function close(){opened=false;chrome();(lastFocus?.isConnected?lastFocus:toggle).focus({preventScroll:true});}
    function addFile(map,raw,turn,node){
      const file=fileReference(raw,{origin:win.location.origin,chatId:context.chatId});if(!file)return;
      if(!map.has(file.key))map.set(file.key,{...file,turn});
      if(node){node.dataset.artifactKey=file.key;node.title=t('Open in sidebar','在右侧栏预览')+' · '+file.name;}
    }
    function update(){
      frame=0;const next=getContext() || {},changed=context.chatId!==next.chatId,language=context.zh!==next.zh;context=next;
      if(changed||language||context.view!=='chat')closeFileMenu();
      if(changed){clearLoad();extraFiles.clear();selected=null;opened=false;tab='list';sourceMode=false;signature='';search.value='';body.replaceChildren();}
      const found=new Map();let turn=0;
      for(const bubble of messages.querySelectorAll('.msg.assistant .bubble, #wb-live-execution .bubble')){
        turn++;
        for(const node of bubble.querySelectorAll('a[href],img[src]'))addFile(found,node.getAttribute(node.tagName==='IMG'?'src':'href'),turn,node);
        for(const node of bubble.querySelectorAll('code'))if(!node.closest('pre')&&fileKind(node.textContent)!=='file')addFile(found,node.textContent,turn,node);
        renderTurnCards(bubble,turn);
      }
      for(const [index,message] of (context.messages||[]).filter(m=>m.role==='assistant').entries())
        for(const change of message.workspaceChanges||[])if(change.status!=='deleted'&&change.kind!=='delete')addFile(found,change.path||change.file_path,index+1);
      for(const [key,file] of extraFiles)if(!found.has(key))found.set(key,file);
      files=[...found.values()];const stamp=files.map(f=>f.key).join('\n');
      if(selected&&!found.has(selected.key)){clearLoad();selected=null;tab='list';body.replaceChildren();}
      if(signature!==stamp||language||changed){signature=stamp;renderList();}
      chrome();if(language&&content)renderContent();if(menuAnchor)placeFileMenu();
    }
    function sync(){if(!frame)frame=win.requestAnimationFrame(update);}
    new win.MutationObserver(sync).observe(messages,{childList:true,characterData:true,subtree:true});
    messages.addEventListener('click',event=>{
      if(event.defaultPrevented||event.button!==0||event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;
      const node=event.target.closest('[data-artifact-key]');if(!node)return;
      const file=files.find(item=>item.key===node.dataset.artifactKey);if(file){event.preventDefault();openFile(file);}
    });
    doc.addEventListener('keydown',event=>{if(!event.defaultPrevented&&event.key==='Escape'&&!panel.hidden&&!doc.querySelector('dialog[open]')){event.preventDefault();close();}});
    head.addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight'].includes(event.key)&&selected&&[listTab,previewTab].includes(event.target)){event.preventDefault();show(tab==='list'?'preview':'list');}});
    win.addEventListener('resize',fitWidth);
    if(win.ResizeObserver)new win.ResizeObserver(fitWidth).observe(doc.getElementById('app-sidebar'));
    const storeWidth=()=>{try{win.localStorage.setItem('neurodiscovery.artifact-width',String(requestedWidth));}catch(_){}};
    resizer.addEventListener('pointerdown',event=>{
      if(event.button!==0)return;event.preventDefault();resizer.setPointerCapture(event.pointerId);doc.body.classList.add('artifact-resizing');
      const start=event.clientX,width=panel.getBoundingClientRect().width;
      const move=e=>{requestedWidth=Math.max(340,Math.min(Number(resizer.getAttribute('aria-valuemax')),width+start-e.clientX));fitWidth();};
      const stop=()=>{doc.body.classList.remove('artifact-resizing');resizer.removeEventListener('pointermove',move);resizer.removeEventListener('pointerup',stop);resizer.removeEventListener('lostpointercapture',stop);storeWidth();};
      resizer.addEventListener('pointermove',move);resizer.addEventListener('pointerup',stop);resizer.addEventListener('lostpointercapture',stop);
    });
    resizer.addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){event.preventDefault();const max=Number(resizer.getAttribute('aria-valuemax'));requestedWidth=event.key==='Home'?340:event.key==='End'?max:Math.min(max,Math.max(340,panel.getBoundingClientRect().width+(event.key==='ArrowLeft'?20:-20)));fitWidth();storeWidth();}});
    async function readBounded(response,limit){
      if(Number(response.headers.get('content-length'))>limit)throw Error(t('This file is too large to preview. Download the original.','文件较大，请下载原文件查看。'));
      const reader=response.body.getReader(),chunks=[];let size=0;
      try {while(true){const {done,value}=await reader.read();if(done)break;size+=value.length;if(size>limit)throw Error(t('This file is too large to preview. Download the original.','文件较大，请下载原文件查看。'));chunks.push(value);}}finally{await reader.cancel().catch(()=>{});}
      return new win.Blob(chunks);
    }
    async function openFile(file,focus=true,asSource=false){
      if(!files.some(item=>item.key===file.key)){extraFiles.set(file.key,file);files.push(file);}
      closeFileMenu();clearLoad();selected=file;sourceMode=asSource;filename.textContent=file.name;download.href=file.url+(file.url.includes('?')?'&':'?')+'download=1';download.download=file.name;
      sourceBtn.hidden=!['markdown','table','json','html','text'].includes(file.kind);viewBtn.hidden=sourceBtn.hidden;body.replaceChildren(el('p','artifact-empty',t('Loading…','正在加载…')));
      if(focus)show('preview');else{tab='preview';chrome();}renderList();
      if(file.kind==='file'){body.replaceChildren(el('p','artifact-empty',t('Preview is not available for this format. Download the original file to open it.','此格式暂不支持内嵌预览，可下载原文件打开。')));return;}
      controller=new win.AbortController();const id=loadId;
      try {
        const response=await win.fetch(file.url,{signal:controller.signal,redirect:'error'});
        if(!response.ok)throw Error(response.status===413?t('This file is too large to preview. Download the original.','文件较大，请下载原文件查看。'):t('File is unavailable. It may have moved or be outside this conversation’s workspace.','文件暂时无法读取，可能已移动或不在当前对话的工作区内。'));
        const binary=['image','pdf'].includes(file.kind),blob=await readBounded(response,binary?BINARY_LIMIT:TEXT_LIMIT);
        if(id!==loadId)return;
        if(binary){
          if(file.ext==='svg'){
            const graph=win.NeuroEvidenceGraph?.readSvg(await blob.text());if(id!==loadId)return;
            if(graph){content={graph};renderContent();return;}
          }
          if(file.kind==='pdf' && !(await blob.slice(0,5).text()).startsWith('%PDF-'))throw Error(t('This file is not a readable PDF.','该文件不是有效的 PDF。'));
          if(id!==loadId)return;
          const mime=file.kind==='pdf'?'application/pdf':file.ext==='svg'?'image/svg+xml':file.ext==='jpg'?'image/jpeg':'image/'+file.ext;
          objectUrl=win.URL.createObjectURL(new win.Blob([blob],{type:mime}));content={binary:true,url:objectUrl};
        }else{const text=await blob.text();if(id!==loadId)return;content={text};}
        renderContent();
      }catch(error){if(id===loadId&&error.name!=='AbortError')body.replaceChildren(el('p','artifact-empty',error.message));}
    }
    function sourceText(text){
      const pre=el('pre','artifact-text'),code=el('code','',text);
      code.dataset.language=({py:'python',js:'javascript',cjs:'javascript',ts:'typescript',sh:'bash',yml:'yaml',md:'markdown',txt:'text'})[selected.ext]||selected.ext;
      pre.append(code);return pre;
    }
    function renderContent(){
      if(!content||!selected)return;body.replaceChildren();viewBtn.setAttribute('aria-pressed',String(!sourceMode));sourceBtn.setAttribute('aria-pressed',String(sourceMode));
      if(sourceMode){body.append(sourceText(content.text||''));win.NeuroCodeBlocks?.enhance(body);return;}
      if(selected.kind==='table'){renderTable(content.text);return;}
      if(selected.kind==='image'){
        if(content.graph){body.append(win.NeuroEvidenceGraph.create(content.graph));return;}
        const tools=el('div','artifact-image-tools'),stage=el('div','artifact-image-stage'),img=el('img');img.src=content.url;img.alt=selected.name;
        img.addEventListener('error',()=>stage.replaceChildren(el('p','artifact-empty',t('Image could not be decoded. Download the original file.','无法解析图片，请下载原文件查看。'))));
        tools.append(button('artifact-action',t('Fit','适应宽度'),()=>{stage.classList.remove('is-zoomed');img.style.width='';}),button('artifact-action',t('Actual size','原始大小'),()=>{stage.classList.add('is-zoomed');img.style.width=img.naturalWidth+'px';}));
        stage.append(img);body.append(tools,stage);return;
      }
      if(['html','pdf'].includes(selected.kind)){
        const iframe=el('iframe','artifact-frame');iframe.title=selected.name;iframe.referrerPolicy='no-referrer';
        if(selected.kind==='html'){
          const inert=el('template');inert.innerHTML=content.text;
          inert.content.querySelectorAll('meta[http-equiv],base').forEach(node=>node.remove());
          inert.content.querySelectorAll('a,area').forEach(node=>{if(!String(node.getAttribute('href')||'').startsWith('#'))node.removeAttribute('href');node.removeAttribute('target');node.removeAttribute('xlink:href');});
          iframe.setAttribute('sandbox','');iframe.srcdoc=staticHtml(inert.innerHTML);
        }else{iframe.src=content.url+'#toolbar=1&navpanes=0';}
        body.append(iframe);return;
      }
      if(selected.kind==='markdown'){renderMarkdown(content.text);return;}
      let text=content.text;
      if(selected.kind==='json'){try{text=JSON.stringify(JSON.parse(text.replace(/^\uFEFF/,'')),null,2);}catch(_){body.append(el('p','artifact-notice',t('Invalid JSON; showing the original text.','JSON 格式不完整，显示原文。')));}}
      const pre=sourceText(text);body.append(pre);
      const code=pre.firstElementChild,lang=({py:'python',js:'javascript',cjs:'javascript',ts:'typescript',sh:'bash',yml:'yaml'})[selected.ext]||selected.ext;
      if(text.length<50000&&win.hljs?.getLanguage?.(lang)){code.className='language-'+lang;win.hljs.highlightElement(code);}
      win.NeuroCodeBlocks?.enhance(body);
    }
    function nestedReference(raw){
      if(!raw||/^(?:https?:|#|\/|[a-z]:[\\/])/i.test(raw))return fileReference(raw,{origin:win.location.origin,chatId:context.chatId});
      if(selected.reference){const parent=selected.reference.replace(/[^/\\]+$/,'');return fileReference(parent+raw,{origin:win.location.origin,chatId:context.chatId});}
      return fileReference(new URL(raw,new URL(selected.url,win.location.origin)).href,{origin:win.location.origin,chatId:context.chatId});
    }
    function renderMarkdown(text){
      const article=el('article','artifact-markdown'),template=el('template');
      template.innerHTML=win.marked?.parse?win.marked.parse(text):'<pre>'+escapeHtml(text)+'</pre>';
      const allowed=new Set('P H1 H2 H3 H4 H5 H6 UL OL LI STRONG EM DEL BLOCKQUOTE PRE CODE BR HR TABLE THEAD TBODY TR TH TD A IMG INPUT SUP SUB'.split(' '));
      for(const node of [...template.content.querySelectorAll('*')]){
        if(!allowed.has(node.tagName)){if(['SCRIPT','STYLE','IFRAME','OBJECT','EMBED','SVG','MATH','FORM'].includes(node.tagName))node.remove();else node.replaceWith(...node.childNodes);continue;}
        const href=node.getAttribute('href'),src=node.getAttribute('src'),alt=node.getAttribute('alt'),checked=node.hasAttribute('checked'),type=node.getAttribute('type'),lang=node.className.match(/\blanguage-([\w+-]+)/)?.[1];
        for(const attr of [...node.attributes])node.removeAttribute(attr.name);
        if(node.tagName==='CODE'&&lang)node.dataset.language=lang;
        if(node.tagName==='A'){
          const file=nestedReference(href);
          if(file){node.href=file.url;node.addEventListener('click',event=>{event.preventDefault();openFile({...file,turn:selected.turn});});}
          else if(/^https?:\/\//i.test(href||'')){node.href=href;node.target='_blank';node.rel='noopener noreferrer';}
        }
        if(node.tagName==='IMG'){const file=nestedReference(src);if(file?.kind==='image'){node.src=file.url;node.alt=alt||file.name;}else node.replaceWith(el('span','',alt||''));}
        if(node.tagName==='INPUT'){if(type==='checkbox'){node.type='checkbox';node.checked=checked;node.disabled=true;}else node.remove();}
      }
      article.append(template.content);body.append(article);
      win.NeuroCodeBlocks?.enhance(article);
      win.NeuroEvidenceGraph?.enhance(article);
    }
    function renderTable(text){
      const parsed=parseDelimited(text,selected.ext==='tsv'?'\t':','),section=el('div','artifact-data'),tools=el('div','artifact-data-tools');
      let page=0,hasHeader=true;const query=el('input','artifact-search');query.type='search';query.id='artifact-table-search';query.placeholder=t('Search all rows…','搜索表格全部行…');query.setAttribute('aria-label',query.placeholder);
      const headerLabel=el('label'),check=el('input');check.type='checkbox';check.checked=true;headerLabel.append(check,doc.createTextNode(t(' Header row',' 首行为表头')));
      const info=el('span','artifact-data-summary');tools.append(query,headerLabel,info);
      const scroller=el('div','artifact-table-scroll');scroller.tabIndex=0;scroller.setAttribute('role','region');scroller.setAttribute('aria-label',t('Table, scroll horizontally for more columns','表格，可左右滚动查看全部列'));
      const pagination=el('div','artifact-pagination'),prev=button('artifact-action',t('Previous','上一页'),()=>{page--;draw();}),next=button('artifact-action',t('Next','下一页'),()=>{page++;draw();}),pageInfo=el('span');pagination.append(prev,pageInfo,next);
      section.append(tools,scroller,pagination);body.append(section);
      if(parsed.warning||parsed.truncated)section.prepend(el('p','artifact-notice',parsed.truncated?t('Preview limited to 20,000 rows and 200 columns. Download for the complete file.','预览最多显示 20,000 行、200 列，完整内容请下载原文件。'):t('An unfinished quoted cell was found; check the original file.','发现未闭合的引号，请核对原始文件。')));
      function draw(){
        const headers=hasHeader?(parsed.rows[0]||[]):[],rows=parsed.rows.slice(hasHeader?1:0),filter=query.value.trim().toLowerCase();
        const matched=rows.map((cells,index)=>({cells,index})).filter(row=>!filter||row.cells.some(cell=>cell.toLowerCase().includes(filter)));
        const pages=Math.max(1,Math.ceil(matched.length/100));page=Math.max(0,Math.min(pages-1,page));
        const columns=Math.max(headers.length,...rows.map(row=>row.length),0),table=el('table','artifact-table'),thead=el('thead'),tr=el('tr');
        tr.append(el('th','artifact-row-number','#'));for(let i=0;i<columns;i++)tr.append(el('th','',headers[i]||t('Column '+(i+1),'列 '+(i+1))));thead.append(tr);table.append(thead);
        const tbody=el('tbody');for(const {cells,index} of matched.slice(page*100,(page+1)*100)){
          const row=el('tr');row.append(el('td','artifact-row-number',String(index+1)));
          for(let i=0;i<columns;i++){const value=cells[i]||'';row.append(el('td',value.length>55?'artifact-cell-prose':'',value));}tbody.append(row);
        }
        table.append(tbody);const left=scroller.scrollLeft;scroller.replaceChildren(table);scroller.scrollLeft=left;scroller.scrollTop=0;
        info.textContent=t(`${rows.length} rows · ${columns} columns`,`${rows.length} 行 · ${columns} 列`)+(filter?t(` · ${matched.length} matches`,` · 匹配 ${matched.length} 行`):'');
        pageInfo.textContent=t(`Page ${page+1} of ${pages}`,`第 ${page+1} / ${pages} 页`);prev.disabled=page===0;next.disabled=page>=pages-1;
      }
      query.addEventListener('input',()=>{page=0;draw();});check.addEventListener('change',()=>{hasHeader=check.checked;page=0;draw();});draw();
    }
    sync();return {sync,close,open:openFile};
  }
  return {mount,fileKind,fileReference,parseDelimited,staticHtml,TEXT_LIMIT,BINARY_LIMIT};
});
