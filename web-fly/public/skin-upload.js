const $=id=>document.getElementById(id);
const skinKey=name=>name.normalize('NFKC').trim().toLowerCase();
let importing=false;
async function savedSkins(db){return new Promise((resolve,reject)=>{const r=db.transaction('skins').objectStore('skins').getAll();r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});}
const images={barLeft:'taiko-bar-left',drumInner:'taiko-drum-inner',drumOuter:'taiko-drum-outer',barRight:'taiko-bar-right',barGlow:'taiko-bar-right-glow',hit:'taikohitcircle',hitOverlay:'taikohitcircleoverlay',hitOverlay1:'taikohitcircleoverlay-1',big:'taikobigcircle',bigOverlay:'taikobigcircleoverlay',bigOverlay1:'taikobigcircleoverlay-1',rollMiddle:'taiko-roll-middle',rollEnd:'taiko-roll-end',spinner:'spinner-circle'};
const sounds={don:'taiko-drum-hitnormal',kat:'taiko-drum-hitclap',finish:'taiko-drum-hitfinish',whistle:'taiko-drum-hitwhistle',roll:'taiko-drum-hitnormal',spinner:'spinnerbonus',miss:'combobreak'};
const database=new Promise((resolve,reject)=>{const r=indexedDB.open('flytaiko-skins',1);r.onupgradeneeded=()=>r.result.createObjectStore('skins',{keyPath:'id'});r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});database.catch(()=>{});
async function player(){for(let i=0;i<200;i++){const p=$('viewer').contentWindow;if(p?.registerRecordedSkin)return p;await new Promise(r=>setTimeout(r,100));}throw Error('Player unavailable');}
async function install(record){
 const urls={images:{},sounds:{}};for(const a of record.assets)urls[a.type][a.key]=URL.createObjectURL(new Blob([a.bytes],{type:a.mime}));
 let report;
 try{report=await(await player()).registerRecordedSkin(record.id,urls);}finally{for(const g of Object.values(urls))for(const url of Object.values(g))URL.revokeObjectURL(url);}
 if(![...$('skin').options].some(o=>o.value===record.id))$('skin').add(new Option(record.name,record.id));
 return report;
}
async function importSkin(files,archive){
 if(importing)return;importing=true;
 $('skin-status').textContent='Reading skin…';
 try{
  const entries=[];
  if(archive){if(archive.size>64*1024**2)throw Error('Archive limit: 64 MB');const zip=await JSZip.loadAsync(archive);const members=Object.values(zip.files);if(members.length>2000)throw Error('Too many skin files');for(const item of members)if(!item.dir)entries.push({name:item.name.split('/').pop().toLowerCase(),path:item.name.toLowerCase(),size:item._data?.uncompressedSize||0,read:()=>item.async('arraybuffer')});}
  else for(const file of files)entries.push({name:file.name.toLowerCase(),path:(file.webkitRelativePath||file.name).toLowerCase(),size:file.size,read:()=>file.arrayBuffer()});
  entries.sort((a,b)=>Number(/(^|\/)taiko\//.test(b.path))-Number(/(^|\/)taiko\//.test(a.path)));
  const assets=[];let total=0;
  for(const [type,map]of Object.entries({images,sounds}))for(const [key,base]of Object.entries(map)){
   const suffix={don:'normal',kat:'clap',finish:'finish',whistle:'whistle',roll:'normal'}[key];
   const aliases=suffix?[base,`taiko-normal-hit${suffix}`,`taiko-soft-hit${suffix}`,`taiko-drum-hit${suffix}`]:[base]; // Taiko-prefixed samples only.
   const names=type==='images'?[base+'@2x.png',base+'.png',base+'-0@2x.png',base+'-0.png']:aliases.flatMap(b=>[b+'.ogg',b+'.wav',b+'.mp3']);const e=names.map(n=>entries.find(f=>f.name===n)).find(Boolean);if(!e)continue;
   if(e.size>8*1024**2)throw Error('Asset limit: 8 MB');const bytes=await e.read();total+=bytes.byteLength;if(bytes.byteLength>8*1024**2||total>40*1024**2)throw Error('Selected assets exceed size limit');
   assets.push({type,key,bytes,mime:{png:'image/png',ogg:'audio/ogg',wav:'audio/wav',mp3:'audio/mpeg'}[e.name.split('.').pop()]});
  }
  if(!assets.some(a=>a.key==='hit'))throw Error('No Taiko hitcircle found');
  const name=archive?archive.name.replace(/\.(osk|zip)$/i,''):(files[0]?.webkitRelativePath.split('/')[0]||'Custom skin');
  const db=await database,duplicates=(await savedSkins(db)).filter(r=>skinKey(r.name)===skinKey(name));
  const record={id:duplicates[0]?.id||'custom-'+crypto.randomUUID(),name,assets};
  const report=await install(record);await new Promise((resolve,reject)=>{const tx=db.transaction('skins','readwrite'),store=tx.objectStore('skins');for(const old of duplicates)store.delete(old.id);store.put(record);tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);tx.onabort=()=>reject(tx.error||Error('Skin save aborted'));});
  for(const option of [...$('skin').options])if(option.value!==record.id && duplicates.some(old=>old.id===option.value)){(await player()).removeRecordedSkin(option.value);option.remove();localStorage.removeItem('flytaiko-hit-volume-'+option.value);}
  [...$('skin').options].find(o=>o.value===record.id).textContent=name;
  $('skin').value=record.id;await $('skin').onchange();
  const missing=['don','kat'].filter(k=>!report.sounds.includes(k));
  $('skin-status').textContent=`${record.name}: ${assets.length} assets saved in this browser. Hitsounds: ${report.sounds.join(', ')||'none'}.${missing.length?' Default fallback: '+missing.join(', ')+'.':''}${report.soundErrors.length?' Could not decode: '+report.soundErrors.join(', ')+'.':''}`;
 }catch(e){$('skin-status').textContent='Skin import failed: '+e.message;}finally{importing=false;$('skin-archive').value='';$('skin-folder').value='';}
}
$('skin-archive').onchange=e=>{if(e.target.files[0])importSkin(null,e.target.files[0]);};
$('skin-folder').onchange=e=>{if(e.target.files.length)importSkin(e.target.files,null);};
database.then(db=>new Promise((resolve,reject)=>{const r=db.transaction('skins').objectStore('skins').getAll();r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);})).then(async rows=>{
 for(const row of rows)await install(row);
 const selected=localStorage.getItem('flytaiko-selected-skin');
 if(rows.some(row=>row.id===selected)){$('skin').value=selected;await $('skin').onchange();}
}).catch(e=>{$('skin-status').textContent='Saved skins unavailable: '+e.message;});
