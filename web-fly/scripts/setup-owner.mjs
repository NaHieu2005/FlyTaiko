// Generates credentials locally; never prints secrets or commits them.
import {randomBytes,createHash} from 'node:crypto';
import {readFile,writeFile,mkdir,chmod} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import {put} from '@vercel/blob';
const root=new URL('../../runs/',import.meta.url);await mkdir(root,{recursive:true});
const file=new URL('owner-secrets.json',root);
let secrets;
try{secrets=JSON.parse(await readFile(file,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;secrets={password:randomBytes(24).toString('base64url'),auth:randomBytes(32).toString('base64url'),publish:randomBytes(32).toString('base64url')};await writeFile(file,JSON.stringify(secrets),{mode:0o600});}
await chmod(file,0o600);
await writeFile(new URL('owner-password.txt',root),secrets.password+'\n',{mode:0o600});
const bootstrap=await put('system/bootstrap.json','{}',{access:'public',token:process.env.BLOB_READ_WRITE_TOKEN,addRandomSuffix:false,allowOverwrite:true,contentType:'application/json'});
const variables={FLYTAIKO_AUTH_SECRET:secrets.auth,FLYTAIKO_PUBLISH_KEY:secrets.publish,FLYTAIKO_ADMIN_PASSWORD_SHA256:createHash('sha256').update(secrets.password).digest('hex'),FLYTAIKO_ASSET_ORIGIN:new URL(bootstrap.url).origin};
for(const [name,value]of Object.entries(variables)){
 const result=spawnSync('vercel',['env','add',name,'production,preview','--force','--sensitive','--yes','--scope','nahieu2005'],{input:value,encoding:'utf8'});
 if(result.status!==0)throw Error('Cannot configure '+name);
 console.log('Configured secret:',name);
}
await writeFile(new URL('cloud.env',root),Object.entries({...variables,FLYTAIKO_CLOUD_URL:'https://flytaiko.vercel.app',BLOB_READ_WRITE_TOKEN:process.env.BLOB_READ_WRITE_TOKEN||''}).map(([k,v])=>k+'='+JSON.stringify(v)).join('\n')+'\n',{mode:0o600});
console.log('Owner password saved to runs/owner-password.txt (private, excluded from Git).');
