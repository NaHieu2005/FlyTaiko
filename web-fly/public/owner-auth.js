let token,expires=0;
const $=id=>document.getElementById(id);
function display(admin){$('upload-panel').hidden=!admin;$('open-upload').hidden=!admin;$('owner-action').textContent=admin?'Sign out':'Owner sign in';window.dispatchEvent(new CustomEvent('owner-session',{detail:admin}));}
async function session(){const r=await fetch('/api/admin/session',{cache:'no-store'});if(!r.ok)throw Error('Owner service unavailable');const s=await r.json();token=s.backend_token;expires=Date.now()+5*60*1000;display(s.admin);return s;}
const ready=window.FLYTAIKO_CLOUD_MODE?session().catch(()=>{display(false);}):Promise.resolve();
window.flytaikoOwner={ready,async token(){await ready;if(!token||Date.now()>expires)await session();if(!token)throw Error('Owner sign-in required');return token;}};
$('owner-action').onclick=async()=>{
 if(token){await fetch('/api/admin/logout',{method:'POST'});token=null;expires=0;display(false);return;}
 $('owner-error').textContent='';$('owner-dialog').showModal();$('owner-password').focus();
};
$('owner-cancel').onclick=()=>$('owner-dialog').close();
$('owner-form').onsubmit=async e=>{
 e.preventDefault();$('owner-submit').disabled=true;
 try{const r=await fetch('/api/admin/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:$('owner-password').value})});if(!r.ok)throw Error(r.status===401?'Incorrect owner password.':'Sign-in unavailable.');$('owner-password').value='';await session();$('owner-dialog').close();$('upload-panel').open=true;}
 catch(error){$('owner-error').textContent=error.message;}finally{$('owner-submit').disabled=false;}
};
