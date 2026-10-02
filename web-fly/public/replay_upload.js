const get = id => document.getElementById(id);
const endpoint = path => window.FLYTAIKO_BACKEND_URL ? new URL(path, window.FLYTAIKO_BACKEND_URL).href : path;
let uploadId, timer;
get('open-upload').onclick=()=>{get('upload-panel').open=true;};
const status = text => { get('upload-status').textContent = text; };
async function response(res) { const body=await res.json(); if(!res.ok) throw Error(body.error || `HTTP ${res.status}`); return body; }
get('upload-map').onclick = () => {
 const file=get('upload-file').files[0];
 if(!file || !file.name.toLowerCase().endsWith('.osz') || file.size > 128*1024**2) return status('Select an .osz file up to 128 MB.');
 get('upload-map').disabled=true; get('generate-replay').disabled=true; status('Uploading map…');
 const xhr=new XMLHttpRequest(); xhr.open('POST',endpoint('/api/osz')); xhr.setRequestHeader('Content-Type','application/octet-stream'); xhr.timeout=300000;
 xhr.upload.onprogress=e=>{ if(e.lengthComputable) get('upload-progress').value=e.loaded/e.total*100; };
 const fail=()=>{status('Upload failed. Check the backend connection.');get('upload-map').disabled=false;};
 xhr.onerror=fail; xhr.ontimeout=fail;
 xhr.onload=()=>{try {
  const body=JSON.parse(xhr.responseText); if(xhr.status!==200) throw Error(body.error||`HTTP ${xhr.status}`);
  uploadId=body.upload_id; get('upload-chart').replaceChildren();
  body.charts.forEach((c,i)=>get('upload-chart').add(new Option(`${c.artist} - ${c.title} [${c.version}] · ${c.notes} notes`,i)));
  get('generate-replay').disabled=false; status('Select a difficulty and mods, then generate a replay.');
 }catch(e){status(`Error: ${e.message}`);}finally{get('upload-map').disabled=false;}};
 xhr.send(file);
};
get('generate-replay').onclick=async()=>{
 get('generate-replay').disabled=true;
 try {
  const job=await response(await fetch(endpoint('/api/replays'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({upload_id:uploadId,chart_index:Number(get('upload-chart').value),mods:get('upload-mods').value,model:'v25'})}));
  localStorage.setItem('flytaiko-replay-job',job.job_id); watch(job.job_id);
 }catch(e){status(`Error: ${e.message}`);get('generate-replay').disabled=false;}
};
async function poll(id) {
 try {
  const res=await fetch(endpoint(`/api/replays/${id}`),{cache:'no-store'});
  if(res.status===404){clearInterval(timer);localStorage.removeItem('flytaiko-replay-job');status('Previous job is unavailable. Select a saved replay or upload a new map.');return;}
  const job=await response(res);
  status(`Job ${id}: ${job.status}${job.progress_frames ? ` · ${job.progress_frames} frames` : ''}`);
  if(['complete','failed'].includes(job.status)) {
   clearInterval(timer);localStorage.removeItem('flytaiko-replay-job');get('generate-replay').disabled=!uploadId;
   if(job.status==='failed') return status(`Error: ${job.error}`);
   status(`Ready: ${job.metrics.great} Great · ${job.metrics.good} Good · ${job.metrics.miss} Miss`);
   const link=document.createElement('a');link.textContent='Watch replay';link.href=`?dataset=${encodeURIComponent(job.dataset)}`;
   get('upload-result').replaceChildren(link);
   window.dispatchEvent(new CustomEvent('replay-created',{detail:job.dataset}));
  }
 }catch(e){status(`Status unavailable: ${e.message}`);}
}
function watch(id){clearInterval(timer);timer=setInterval(()=>poll(id),5000);poll(id);}
const previous=localStorage.getItem('flytaiko-replay-job'); if(/^[a-f0-9]{12}$/.test(previous||''))watch(previous);
