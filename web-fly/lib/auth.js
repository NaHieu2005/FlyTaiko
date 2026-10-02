import {createHmac, timingSafeEqual, createHash} from 'node:crypto';
export function equal(a,b){const digest=x=>createHash('sha256').update(String(x)).digest();return timingSafeEqual(digest(a),digest(b));}
export function sign(audience,ttl=900,secret=process.env.FLYTAIKO_AUTH_SECRET){
 if(!secret)throw Error('Authentication is not configured');
 const payload=Buffer.from(JSON.stringify({sub:'owner',aud:audience,exp:Math.floor(Date.now()/1000)+ttl})).toString('base64url');
 return payload+'.'+createHmac('sha256',secret).update(payload).digest('base64url');
}
export function valid(token,audience,secret=process.env.FLYTAIKO_AUTH_SECRET){
 try{if(!secret || !token || token.length>2048)return false;const [body,signature,...extra]=token.split('.');if(extra.length || !equal(signature,createHmac('sha256',secret).update(body).digest('base64url')))return false;const p=JSON.parse(Buffer.from(body,'base64url'));return p.sub==='owner'&&p.aud===audience&&Number.isFinite(p.exp)&&p.exp>Date.now()/1000;}catch{return false;}
}
export function owner(req){const token=(req.headers.cookie||'').split(';').map(s=>s.trim()).find(s=>s.startsWith('flytaiko_owner='))?.slice('flytaiko_owner='.length);return valid(token,'web-owner');}
export function sameOrigin(req){return req.headers.origin===(process.env.FLYTAIKO_ADMIN_ORIGIN||'https://flytaiko.vercel.app');}
