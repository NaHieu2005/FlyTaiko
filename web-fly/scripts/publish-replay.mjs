// Publish every referenced asset before committing a replay to the catalogue.
import {put,head} from '@vercel/blob';
import {readFile,stat} from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve,basename,extname} from 'node:path';
const dataset=process.argv[2];
if(!/^[a-zA-Z0-9-]{1,100}$/.test(dataset||''))throw Error('Provide a replay dataset identifier');
const directory=resolve(process.argv[3]||`public/demos/${dataset}`);
const token=process.env.BLOB_READ_WRITE_TOKEN;
if(!token||!process.env.FLYTAIKO_PUBLISH_KEY)throw Error('Publisher credentials missing');
const cache=new Map();
async function digest(file){const hash=createHash('sha256');for await(const chunk of createReadStream(file))hash.update(chunk);return hash.digest('hex');}
async function upload(name,body,hash){
 const key=`assets/${hash}${extname(name)}`;
 try{return (await head(key,{token})).url;}catch(error){if(error.constructor.name!=='BlobNotFoundError')throw error;}
 const mime={'.json':'application/json','.bin':'application/octet-stream','.mp3':'audio/mpeg','.ogg':'audio/ogg','.wav':'audio/wav','.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg','.webp':'image/webp'}[extname(name)]||'application/octet-stream';
 const blob=await put(key,body,{token,access:'public',addRandomSuffix:false,contentType:mime,cacheControlMaxAge:31536000,multipart:true});return blob.url;
}
async function asset(path){
 if(typeof path!=='string'||!path.startsWith(`demos/${dataset}/`))throw Error('Asset points outside this replay');
 if(cache.has(path))return cache.get(path);
 const name=path.slice(`demos/${dataset}/`.length);
 if(!/^[a-zA-Z0-9_.-]+$/.test(name))throw Error('Invalid asset filename');
 const file=resolve(directory,name);const info=await stat(file);
 if(!info.isFile())throw Error('Missing asset '+name);
 let body,hash;
 if(name.endsWith('.json')){const data=await rewrite(JSON.parse(await readFile(file,'utf8')));body=Buffer.from(JSON.stringify(data));hash=createHash('sha256').update(body).digest('hex');}
 else{hash=await digest(file);body=createReadStream(file);}
 const url=await upload(name,body,hash);cache.set(path,url);console.log('Uploaded asset:',name);return url;
}
async function rewrite(value){
 if(Array.isArray(value))return value.every(v=>typeof v==='number'||v===null)?value:Promise.all(value.map(rewrite));
 if(value&&typeof value==='object'){const result={};for(const [k,v]of Object.entries(value))result[k]=await rewrite(v);return result;}
 if(typeof value==='string'&&value.startsWith(`demos/${dataset}/`))return asset(value);
 return value;
}
const original=JSON.parse(await readFile(resolve(directory,'manifest.json'),'utf8'));
const manifest=await rewrite(original);
const bytes=Buffer.from(JSON.stringify(manifest));const hash=createHash('sha256').update(bytes).digest('hex');
const manifest_url=await upload('manifest.json',bytes,hash);
const response=await fetch(new URL('/api/admin/publish',process.env.FLYTAIKO_CLOUD_URL||'https://flytaiko.vercel.app'),{method:'POST',headers:{'Content-Type':'application/json',Authorization:'Bearer '+process.env.FLYTAIKO_PUBLISH_KEY},body:JSON.stringify({dataset,label:manifest.replays[0].label,manifest_url,manifest,metrics:manifest.replays[0].metrics})});
if(!response.ok)throw Error('Catalogue publication failed: HTTP '+response.status);
console.log(JSON.stringify({published:true,dataset,manifest_url}));
