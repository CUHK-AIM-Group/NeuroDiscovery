// Production frontend and preload, with a local scripted demo transport.
const {app,BrowserWindow,Menu,dialog,ipcMain,shell,session,nativeTheme}=require('electron');
const path=require('node:path');
const fs=require('node:fs');
const os=require('node:os');
const {createDemoServer}=require('./demo/native-server.cjs');

function staticRoot(){return app.isPackaged?path.join(__dirname,'demo/native-ui'):path.resolve(__dirname,'../core/web/static');}
function workspaceRoot(){return app.isPackaged?path.join(process.resourcesPath,'demo-workspace'):path.resolve(__dirname,'..');}
function dataRoot(){return path.join(workspaceRoot(),'data/demo');}
function createDemoWindow({origin,show=true}){
  const win=new BrowserWindow({title:'NeuroDiscovery',width:1320,height:900,minWidth:960,minHeight:680,show,
    icon:path.join(__dirname,'assets','icon.png'),
    backgroundColor:nativeTheme.shouldUseDarkColors?'#151517':'#ffffff',autoHideMenuBar:false,
    webPreferences:{preload:path.join(__dirname,'preload.js'),contextIsolation:true,nodeIntegration:false,sandbox:true}});
  win.webContents.on('will-navigate',(event,url)=>{if(!url.startsWith(origin+'/'))event.preventDefault();});
  win.webContents.setWindowOpenHandler(({url})=>{
    if(url.startsWith(origin+'/demo/artifacts/'))win.webContents.downloadURL(url);
    else if(/^https?:\/\//.test(url)&&!url.startsWith(origin+'/'))void shell.openExternal(url);
    return {action:'deny'};
  });
  win.loadURL(origin+'/harness');return win;
}
function installNetwork(origin){
  const vendors={
    'https://cdn.jsdelivr.net/npm/marked@9/marked.min.js':'marked.min.js',
    'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js':'highlight.min.js',
    'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css':'github.min.css',
  };
  session.defaultSession.webRequest.onBeforeRequest((details,callback)=>{
    if(vendors[details.url])return callback({redirectURL:origin+'/vendor/'+vendors[details.url]});
    // Chromium's bundled PDF viewer is local, including its extension resources.
    const pdfViewer=details.url.startsWith('chrome-extension://mhjfbmdgcfjbbpaeojofohoefgiehjai/')||details.url.startsWith('chrome://resources/');
    callback({cancel:!details.url.startsWith(origin+'/')&&!/^(data|blob):/.test(details.url)&&!pdfViewer});
  });
}
function installIPC({origin,profile,getWindow}){
  const file=path.join(profile,'appearance.json');
  let appearance={language:'English',theme:'light'};
  if(fs.existsSync(file))appearance={...appearance,...JSON.parse(fs.readFileSync(file,'utf8'))};
  const appearanceOnly=value=>{
    if(typeof value?.language==='string')appearance.language=value.language;
    if(['light','dark','system'].includes(value?.theme))appearance.theme=value.theme;
    fs.writeFileSync(file,JSON.stringify(appearance));nativeTheme.themeSource=appearance.theme;installMenu(getWindow,appearance.language);
  };
  nativeTheme.themeSource=appearance.theme;
  installMenu(getWindow,appearance.language);
  const config=()=>({config:{...appearance,llmProvider:'local',llmBaseUrl:origin,llmModel:'GPT-6-Astra',llmApiKey:'',runtimeMode:'bundled'},
    llmConnectionStatus:{provider:'local',apiKeyRequired:false,apiKeyConfigured:false,apiKeySource:'not-required',endpointConfigured:true},
    platform:process.platform,isPackaged:app.isPackaged,configPath:file,logsPath:profile,restartRequired:false});
  const handle=(name,fn)=>ipcMain.handle('neuroclaw:'+name,fn);
  handle('get-config',()=>config());
  handle('save-config',(_event,value)=>{appearanceOnly(value);return config();});
  handle('set-language',(_event,language)=>{appearanceOnly({language});return config();});
  handle('set-theme',(_event,theme)=>{appearanceOnly({theme});return config();});
  handle('discover-models',()=>({models:[{id:'GPT-6-Astra',name:'GPT-6-Astra',provider:'local'}],provider:'local'}));
  handle('detect-local-pythons',()=>[]);
  handle('show-application-menu',()=>{Menu.getApplicationMenu()?.popup({window:getWindow()});return {ok:true};});
  handle('restart',()=>{getWindow()?.reload();return {ok:true};});
  handle('reset-application',()=>({canceled:true}));
  handle('select-attachment-files',()=>({canceled:true,files:[]}));
  handle('select-project-folder',async()=>{
    const result=await dialog.showOpenDialog(getWindow(),{properties:['openDirectory'],defaultPath:workspaceRoot()});
    return {canceled:result.canceled,path:result.filePaths[0]||''};
  });
  handle('create-project-folder',()=>({canceled:true}));
  handle('export-user-study-results',()=>({canceled:true}));
  handle('export-chat-session',async(_event,request)=>{
    const name=path.basename(String(request?.defaultFileName||'NeuroDiscovery-chat.md'));
    const result=await dialog.showSaveDialog(getWindow(),{defaultPath:path.join(app.getPath('documents'),name)});
    if(result.canceled||!result.filePath)return {canceled:true};
    const conversationPath=result.filePath;
    const worklogPath=path.join(path.dirname(conversationPath),path.parse(conversationPath).name+'-worklog-'+Date.now()+path.extname(conversationPath));
    fs.writeFileSync(conversationPath,String(request?.conversationContent||''),'utf8');
    fs.writeFileSync(worklogPath,String(request?.worklogContent||''),{encoding:'utf8',flag:'wx'});
    return {canceled:false,path:conversationPath,directory:path.dirname(conversationPath),conversationPath,worklogPath};
  });
}
function installMenu(getWindow,language='English'){
  const action=name=>()=>getWindow()?.webContents.send('neuroclaw:menu-action',name);
  const t=(en,zh)=>language.includes('Chinese')?zh:en;
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    {label:'NeuroDiscovery',submenu:[{label:t('New Chat','新建对话'),accelerator:'CmdOrCtrl+N',click:action('new-chat')},{type:'separator'},
      {label:t('Human Evaluation 1','Human Evaluation 1（专家研究）'),click:action('open-expert-study')},{label:t('Human Evaluation 2','Human Evaluation 2（扩展）'),click:action('open-hypothesis-ranking')},{type:'separator'},
      {label:t('Settings...','设置...'),click:action('open-settings')},{type:'separator'},{label:t('Reload','重新加载'),role:'reload'},{type:'separator'},{label:t('Exit','退出'),role:'quit'}]},
    {label:t('Edit','编辑'),submenu:[{label:t('Undo','撤销'),role:'undo'},{label:t('Redo','重做'),role:'redo'},{type:'separator'},
      {label:t('Cut','剪切'),role:'cut'},{label:t('Copy','复制'),role:'copy'},{label:t('Paste','粘贴'),role:'paste'},{label:t('Select All','全选'),role:'selectAll'}]},
    {label:t('View','视图'),submenu:[{label:t('Reload','重新加载'),role:'reload'},{label:t('Force Reload','强制重新加载'),role:'forceReload'},{type:'separator'},
      {label:t('Actual Size','实际大小'),accelerator:'CmdOrCtrl+0',click:action('text-scale-reset')},
      {label:t('Zoom In','放大'),accelerator:'CmdOrCtrl+Plus',click:action('text-scale-increase')},
      {label:t('Zoom Out','缩小'),accelerator:'CmdOrCtrl+-',click:action('text-scale-decrease')},
      ...[['CmdOrCtrl+=','increase'],['CmdOrCtrl+numadd','increase'],['CmdOrCtrl+numsub','decrease']].map(([accelerator,direction])=>({label:direction,accelerator,visible:false,acceleratorWorksWhenHidden:true,click:action('text-scale-'+direction)})),
      {type:'separator'},{label:t('Toggle Full Screen','切换全屏'),role:'togglefullscreen'}]},
    {label:t('Help','帮助'),submenu:[{label:t('About NeuroDiscovery','关于 NeuroDiscovery'),click:()=>dialog.showMessageBox(getWindow(),{type:'info',title:t('About NeuroDiscovery','关于 NeuroDiscovery'),message:'NeuroDiscovery Desktop',detail:t('Version ','版本 ')+app.getVersion()})}]},
  ]));
}
async function main(){
  const smoke=process.argv.includes('--demo-smoke');
  app.setName('NeuroDiscovery');
  if(process.platform==='win32')app.setAppUserModelId('org.neurodiscovery.offline-demo');
  const profile=smoke?fs.mkdtempSync(path.join(os.tmpdir(),'nd-native-smoke-')):path.join(app.getPath('appData'),'NeuroDiscovery-NativeDemo');
  app.setPath('userData',profile);
  if(smoke)app.disableHardwareAcceleration();
  if(!app.requestSingleInstanceLock()){app.quit();return;}
  let win;
  app.on('second-instance',()=>{if(win){if(win.isMinimized())win.restore();win.show();win.focus();}});
  await app.whenReady();
  const backend=await createDemoServer({staticRoot:staticRoot(),profile,workspace:workspaceRoot()});
  installNetwork(backend.origin);installIPC({origin:backend.origin,profile,getWindow:()=>win});
  const open=()=>{win=createDemoWindow({origin:backend.origin,show:!smoke});win.on('closed',()=>{win=null;});if(smoke)win.webContents.setBackgroundThrottling(false);};
  open();
  app.on('before-quit',()=>{void backend.close();});
  app.on('activate',()=>{if(!win)open();});
  app.on('window-all-closed',()=>{if(process.platform!=='darwin')app.quit();});
}
if(require.main===module)main().catch(error=>{console.error(error);app.exit(1);});
module.exports={createDemoWindow,staticRoot,workspaceRoot,dataRoot,installNetwork,installIPC,installMenu};
