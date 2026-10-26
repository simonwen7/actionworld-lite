'use strict';
// Every non-reset action calls the real server. No prerecorded lookup/fake fallback.
const $ = (id) => document.getElementById(id);
let token = null, busy = false, mm = 10, latest = null;
const canvas = $('worldView'), ctx = canvas.getContext('2d');
const controlButtons = [...document.querySelectorAll('[data-direction]')];
const sizeButtons = [...document.querySelectorAll('[data-mm]')];
const cacheKey = 'actionworld-m8-current';
function status(message, error=false) { const el=$('status'); el.textContent=message; el.classList.toggle('error',error); }
function lock(v){ busy=v;controlButtons.forEach(b=>b.disabled=v||!token);$('startButton').disabled=v;$('resetButton').disabled=v;sizeButtons.forEach(b=>b.disabled=v); }
function format(n, digits=4){ return Number.isFinite(n)?Number(n).toFixed(digits):'—'; }
async function request(path, body){
  const resp=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),cache:'no-store',credentials:'omit'});
  let obj;
  try{obj=await resp.json();}catch{throw new Error(`HTTP ${resp.status}: invalid JSON response`);}
  if(!resp.ok)throw new Error(obj.detail||`HTTP ${resp.status}`);
  if(!obj.session_token||!obj.real_rgb_png?.startsWith('data:image/png;base64,'))throw new Error('Incomplete real-physics response');
  return obj;
}
function pos(x,y){const h=canvas.width/2;const s=canvas.width/(2*.55);return [h+x*s,h-y*s];}
function blockShape(x,y,yaw,color,dash,fill){
  const boxes=[[.045,.095,.025],[-.020,.025,.060]];
  ctx.strokeStyle=color;ctx.setLineDash(dash);ctx.lineWidth=2.5;ctx.fillStyle=fill;
  for(const [cy,hx,hy] of boxes){
    const corners=[[-hx,cy-hy],[hx,cy-hy],[hx,cy+hy],[-hx,cy+hy]];
    ctx.beginPath();
    corners.forEach(([u,v],i)=>{
      const px=x+Math.cos(yaw)*u-Math.sin(yaw)*v;
      const py=y+Math.sin(yaw)*u+Math.cos(yaw)*v;
      const [sx,sy]=pos(px,py);if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy);
    });ctx.closePath();ctx.fill();ctx.stroke();
  }
  ctx.setLineDash([]);
}
function draw(response){
  if(!ctx)return;
  const w=canvas.width;ctx.fillStyle='#132137';ctx.fillRect(0,0,w,w);
  ctx.strokeStyle='#243a52';ctx.lineWidth=1;
  for(let t=-.5;t<=.5001;t+=.1){let [x]=pos(t,0),[,y]=pos(0,t);ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,w);ctx.stroke();ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
  const a=response?.actual;
  if(!a){ctx.fillStyle='#748dad';ctx.textAlign='center';ctx.font='14px system-ui';ctx.fillText('World coordinates appear after Reset',w/2,w/2);return;}
  blockShape(a.actual_xy_m[0],a.actual_xy_m[1],a.actual_yaw_rad,'#e49a64',[],'rgba(228,154,100,.10)');
  const p=response.prediction;
  if(p){blockShape(p.ic01_xy_m[0],p.ic01_xy_m[1],p.ic01_yaw_rad,'#53a7fa',[8,5],'rgba(83,167,250,.06)');
    blockShape(p.m6_xy_m[0],p.m6_xy_m[1],p.ic01_yaw_rad,'#a993ff',[2,5],'rgba(169,147,255,.04)');}
  const [px,py]=pos(...a.pusher_xy_m);ctx.fillStyle='#62c6f6';ctx.beginPath();ctx.arc(px,py,8,0,2*Math.PI);ctx.fill();
}
function show(response){
  latest=response;token=response.session_token;
  const img=$('realFrame');img.src=response.real_rgb_png;img.style.display='block';$('scenePlaceholder').style.display='none';
  $('stepDisplay').textContent=`STEP ${response.scored_actions} / 64`;
  $('timeDisplay').textContent=`t = ${format(response.physics_time_s,2)} s`;
  const m=response.metrics;
  $('baseError').textContent=m?format(m.ic01_next_xy_error_cm,4):'—';
  $('m6Error').textContent=m?format(m.m6_next_xy_error_cm,4):'—';
  $('correction').textContent=response.prediction?format(Math.hypot(...response.prediction.correction_m)*1000,3):'—';
  $('contact').textContent=m?`${response.prediction.predicted_contact?'Yes':'No'} / ${m.actual_contact?'Yes':'No'}`:'—';
  draw(response);
  // Cache a display snapshot for refresh only. The server will restore the genuine signed physics state.
  try{sessionStorage.setItem(cacheKey,JSON.stringify(response));}catch{}
}
async function reset(){
  lock(true);status('Resetting real MuJoCo simulation…');
  try{const result=await request('/api/session',{});show(result);status('Ready. Each button press runs genuine model inference.');}
  catch(e){status(`Reset failed: ${e.message}`,true);}
  finally{lock(false);}
}
async function action(direction){
  if(busy||!token)return;
  const original=token;lock(true);status(`Computing ${direction} · ${mm} mm…`);
  try{const result=await request('/api/action',{session_token:original,step_size_mm:mm,direction});
    show(result);status(`Step ${result.scored_actions}: real physics + one IC-01 forward + M6 correction.`);}
  catch(e){status(`Action not applied: ${e.message}. Token unchanged. You may Reset.`,true);}
  finally{lock(false);}
}
sizeButtons.forEach(b=>b.addEventListener('click',()=>{if(busy)return;mm=Number(b.dataset.mm);sizeButtons.forEach(x=>x.classList.toggle('selected',x===b));}));
controlButtons.forEach(b=>b.addEventListener('click',()=>action(b.dataset.direction)));
$('resetButton').addEventListener('click',reset);$('startButton').addEventListener('click',reset);
// Serialized foreground actions only. Signed tokens are tamper-evident, not single-use.
try{const saved=JSON.parse(sessionStorage.getItem(cacheKey));
  if(saved?.session_token){show(saved);status('Previous session snapshot restored. The server validates state on your next action; Reset if expired.');}}
catch{}
draw(latest);lock(false);
