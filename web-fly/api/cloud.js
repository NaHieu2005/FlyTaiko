import {neon} from '@neondatabase/serverless';
import {createHash} from 'node:crypto';
import {owner,sign,sameOrigin,equal} from '../lib/auth.js';

export default async function handler(req,res){
 res.setHeader('Cache-Control','no-store');res.setHeader('X-Content-Type-Options','nosniff');
 const reply=(code,body)=>res.status(code).json(body);
 try{
  const url=new URL(req.url,'https://flytaiko.vercel.app');
  const route=String(req.query?.route || url.searchParams.get('route') || '').replace(/^\/+|\/+$/g,'');
  if(route==='admin/session'&&req.method==='GET')return reply(200,owner(req)?{admin:true,backend_token:sign('gpu-worker'),backend_url:process.env.FLYTAIKO_BACKEND_URL}:{admin:false});
  if(route==='admin/login'&&req.method==='POST'){
   if(!sameOrigin(req))return reply(403,{error:'Origin not allowed'});
   const password=req.body?.password;
   if(typeof password!=='string'||password.length>256||!process.env.FLYTAIKO_ADMIN_PASSWORD_SHA256||!equal(createHash('sha256').update(password).digest('hex'),process.env.FLYTAIKO_ADMIN_PASSWORD_SHA256))return reply(401,{error:'Invalid owner password'});
   res.setHeader('Set-Cookie',`flytaiko_owner=${sign('web-owner',86400)}; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=86400`);
   return reply(200,{admin:true});
  }
  if(route==='admin/logout'&&req.method==='POST'){
   if(!sameOrigin(req))return reply(403,{error:'Origin not allowed'});
   res.setHeader('Set-Cookie','flytaiko_owner=; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=0');return reply(200,{admin:false});
  }
  if(route==='replays'&&req.method==='GET'){
   const sql=neon(process.env.DATABASE_URL);
   const rows=await sql`SELECT dataset,label,manifest_url,manifest,metrics FROM replays WHERE label NOT LIKE 'Replay Upload Smoke%' ORDER BY created_at DESC,dataset`;
   res.setHeader('Cache-Control','public, max-age=0, s-maxage=10, stale-while-revalidate=30');
   return reply(200,{replays:rows.map(r=>({...r,status:'complete'}))});
  }
  if(route==='admin/publish'&&req.method==='POST'){
   const token=(req.headers.authorization||'').replace(/^Bearer /,'');
   if(!process.env.FLYTAIKO_PUBLISH_KEY||!equal(token,process.env.FLYTAIKO_PUBLISH_KEY))return reply(401,{error:'Publisher authentication required'});
   const {dataset,label,manifest_url,manifest,metrics}=req.body||{};
   if(!/^[a-zA-Z0-9-]{1,100}$/.test(dataset||'')||typeof label!=='string'||label.length>500||!Array.isArray(manifest?.replays))return reply(400,{error:'Invalid replay metadata'});
   const parsed=new URL(manifest_url);if(parsed.protocol!=='https:'||parsed.origin!==process.env.FLYTAIKO_ASSET_ORIGIN)return reply(400,{error:'Untrusted manifest origin'});
   const sql=neon(process.env.DATABASE_URL);
   await sql`INSERT INTO replays(dataset,label,manifest_url,manifest,metrics) VALUES(${dataset},${label},${manifest_url},${JSON.stringify(manifest)}::jsonb,${JSON.stringify(metrics||{})}::jsonb) ON CONFLICT(dataset) DO UPDATE SET label=EXCLUDED.label,manifest_url=EXCLUDED.manifest_url,manifest=EXCLUDED.manifest,metrics=EXCLUDED.metrics`;
   return reply(200,{published:true,dataset});
  }
  return reply(404,{error:'Unknown API route'});
 }catch(error){console.error('Cloud API error:',error.name);return reply(503,{error:'Cloud service is temporarily unavailable'});}
}
