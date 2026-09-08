/* Particle lifecycle and color buckets adapted from:
 * cambecc/earth, 5f091f0c3b60aa38a996a886985bacb3673d16c3,
 * public/libs/earth/1.0.0/earth.js, animate/evolve/draw.
 * Copyright (c) 2014 Cameron Beccario. MIT; full license embedded in the HTML.
 * FVCOM sampling, RK2, temporal playback, UI and finite-age WebGL: OMA implementation.
 */
"use strict";
(async function(){
const started=performance.now(),$=id=>document.getElementById(id);
try {
  await new Promise(r=>setTimeout(r,20));
  const meta=JSON.parse($("metadata").textContent),data={};
  const types={"<f8":Float64Array,"<f4":Float32Array,"<i4":Int32Array,"|u1":Uint8Array};
  for(const [name,desc] of Object.entries(meta.arrays)) {
    const el=$("data-"+name),raw=atob(el.textContent.trim()),bytes=new Uint8Array(raw.length);
    for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
    data[name]=new types[desc.dtype](bytes.buffer);el.remove();
  }
  const engine=new TracerEngine(data,meta),shade=$("shade"),coast=$("coast");
  const trails=new TrailRenderer($("trails"),{forceCanvas:new URLSearchParams(location.search).has("no-webgl")}),map=coast.getContext("2d");
  let width=0,height=0,dpr=1,scale=1,ox=0,oy=0;
  let mode="snapshot",playing=true,time=meta.times[0],density=3000,tail=3,duration=60,visualSpeed=1;
  const preferences={snapshot:{tail:3,speed:1},continuous:{tail:1,speed:1}};
  const tick=1/60;let activeTime=0,accumulator=0,stallCount=0,emissionsEnabled=true;
  let particles=[],last=0,drag=null,loopCount=0,frameTimes=[],drawTimes=[],shader=null;
  let resetCount=0,selectionMs=[],lastUI=0,visible=false;
  let exporting=false,exportStatus=null,deferredResize=false;
  const palette=[[10,27,55],[18,68,107],[19,123,133],[79,165,123],[196,185,102],[236,113,70]];
  function color(t){const q=Math.max(0,Math.min(1,t))*5,i=Math.min(4,Math.floor(q)),a=q-i;return palette[i].map((v,k)=>Math.round(v*(1-a)+palette[i+1][k]*a));}
  function project(x,y){return [x*scale+ox,oy-y*scale];}
  function unproject(x,y){return [(x-ox)/scale,(oy-y)/scale];}
  function viewport(){const a=unproject(0,height),b=unproject(width,0);return [...a,...b];}
  function trailView(){return {width,height,dpr,scale,ox,oy};}
  function drawTrails(){trails.draw(activeTime,tail,trailView());}
  function clear(){trails.clear(activeTime);resetCount++;}
  function reseed(){engine.setViewport(viewport());particles=Array.from({length:density},()=>{const p=engine.spawn();p.age=engine.random()*8;return p;});clear();}
  function fit(){const b=engine.box;scale=Math.min((width-90)/(b[2]-b[0]),(height-250)/(b[3]-b[1]));ox=width/2-(b[0]+b[2])/2*scale;oy=(height-100)/2+(b[1]+b[3])/2*scale;}
  function upload(){shader?.upload();}
  function drawShade(){shader?.draw(trailView(),time,$("shading").checked);}
  function drawMap(){map.clearRect(0,0,width,height);map.lineWidth=.8;
    drawMeshBoundary(map,data,trailView());const desired=100/scale,power=Math.pow(10,Math.floor(Math.log10(desired))),distance=[1,2,5,10].map(x=>x*power).filter(x=>x<=desired).pop()||power;
    $("scale").style.width=distance*scale+"px";$("scale").textContent=distance>=1000?(distance/1000)+" km":Math.round(distance)+" m";
  }
  function resize(reset=false){width=innerWidth;height=innerHeight;dpr=Math.min(2,devicePixelRatio||1);for(const c of [shade,coast]){c.width=Math.round(width*dpr);c.height=Math.round(height*dpr);}trails.resize(width,height,dpr);map.setTransform(dpr,0,0,dpr,0,0);if(reset)fit();reseed();drawMap();drawShade();}
  function iso(t){return new Date(t*1000).toISOString().replace("T"," ").replace(/\.\d+Z$/," UTC");}
  function updateUI(){
    $("timestamp").textContent=iso(time);$("play").textContent=playing?"Pause":"Play";$("play").setAttribute("aria-label",playing?"Pause animation":"Play animation");
    const slider=$("time"),ts=meta.times;
    if(mode==="snapshot"){slider.max=ts.length-1;slider.step=1;slider.value=engine.a;$("time-note").textContent="Fixed field · "+(engine.a+1)+" / "+ts.length+" native snapshots";}
    else {slider.max=ts[ts.length-1]-ts[0];slider.step=1;slider.value=time-ts[0];$("time-note").textContent="Interpolated "+iso(ts[engine.a]).slice(11,16)+"–"+iso(ts[engine.b]).slice(11,16)+" · field playback";}
    if(engine.blockedGap)$("notice").textContent="Playback stopped at a missing-data interval. Choose another source snapshot to continue.";
    else if(!shader)$("notice").textContent="WebGL2 shading unavailable. Particle trails and the mesh map remain active.";
    else $("notice").textContent="";
    $("export-gif").disabled=mode!=="snapshot"||exporting;$("export-gif").title=mode!=="snapshot"?"Select Snapshot to export a GIF":"Export the visible snapshot";
    $("speed").disabled=false;$("duration").disabled=mode!=="continuous";
    $("motion-note").textContent=mode==="continuous"&&visualSpeed!==1?"Illustrative particle motion ×"+visualSpeed:"";
    $("tail").value=tail;$("tail-value").textContent=tail+" s";$("speed").value=visualSpeed;$("speed-value").textContent=visualSpeed+"×";
    $("duration").value=duration;$("duration-value").textContent=duration+" s";
    const h=trails.metrics(activeTime,tail),equivalent=tail*(meta.times.at(-1)-meta.times[0])/duration/60;
    $("history-note").textContent=(mode==="continuous"?"Trail window: up to "+equivalent.toFixed(1)+" model min at this playback rate. ":"")+(h.historyLimited?"History capacity reached: retained "+h.effectiveHistorySeconds.toFixed(2)+" s. ":"")+(h.trailBackend==="Canvas2D"?"Canvas trail fallback.":"");
  }
  function select(t){const st=performance.now();time=t;accumulator=0;last=0;const changed=engine.setTime(time,mode==="continuous");if(changed)upload();reseed();drawShade();updateUI();selectionMs.push(performance.now()-st);}
  function setMode(value){if(value===mode)return;mode=value;for(const k of ["snapshot","continuous"])$(k).classList.toggle("active",k===mode);
    tail=preferences[mode].tail;visualSpeed=preferences[mode].speed;
    if(mode==="snapshot")time=meta.times[engine.bracket(time,false)[0]];
    select(time);
  }
  function changeView(){clear();engine.setViewport(viewport());drawMap();drawShade();}
  function zoom(factor,x=width/2,y=(height-120)/2){const before=unproject(x,y);const base=Math.min(width/(engine.box[2]-engine.box[0]),height/(engine.box[3]-engine.box[1]));scale=Math.max(base*.3,Math.min(base*80,scale*factor));ox=x-before[0]*scale;oy=y+before[1]*scale;changeView();reseed();}
  $("snapshot").onclick=()=>setMode("snapshot");$("continuous").onclick=()=>setMode("continuous");
  $("play").onclick=()=>setPlaying(!playing);
  $("time").oninput=e=>select(mode==="snapshot"?meta.times[+e.target.value]:meta.times[0]+ +e.target.value);
  $("prev").onclick=()=>select(meta.times[Math.max(0,engine.a-1)]);$("next").onclick=()=>select(meta.times[Math.min(meta.times.length-1,engine.a+1)]);
  $("density").oninput=e=>{density=+e.target.value;$("density-value").textContent=density;reseed();};
  function setVisualSpeed(value){if(!Number.isFinite(value)||value<.25||value>4)throw Error("Visual speed must be 0.25–4");visualSpeed=value;preferences[mode].speed=value;updateUI();}
  function setTrailDuration(value){if(!Number.isFinite(value)||value<.2||value>8)throw Error("Trail duration must be 0.2–8 seconds");tail=value;preferences[mode].tail=value;drawTrails();updateUI();}
  function setDuration(value){if(!Number.isFinite(value)||value<15||value>300)throw Error("Playback duration must be 15–300 seconds");duration=value;updateUI();}
  function setPlaying(value){playing=!!value;last=0;updateUI();}
  $("tail").oninput=e=>setTrailDuration(+e.target.value);
  $("speed").oninput=e=>setVisualSpeed(+e.target.value);
  $("duration").oninput=e=>setDuration(+e.target.value);
  $("shading").onchange=drawShade;
  $("settings-button").onclick=()=>{$("settings").hidden=!$("settings").hidden;};
  $("about-button").onclick=()=>$("about").showModal();$("close-about").onclick=()=>$("about").close();
  $("zoom-in").onclick=()=>zoom(1.5);$("zoom-out").onclick=()=>zoom(1/1.5);$("reset").onclick=()=>{fit();changeView();reseed();};
  coast.onwheel=e=>{e.preventDefault();zoom(Math.exp(-e.deltaY*.001),e.clientX,e.clientY);};
  coast.onpointerdown=e=>{drag=[e.clientX,e.clientY,ox,oy];coast.setPointerCapture(e.pointerId);};
  coast.onpointerup=()=>{drag=null;reseed();};coast.onpointercancel=()=>{drag=null;};
  coast.onpointermove=e=>{
    if(drag){ox=drag[2]+e.clientX-drag[0];oy=drag[3]+e.clientY-drag[1];changeView();return;}
    const [x,y]=unproject(e.clientX,e.clientY),c=engine.locate(x,y);if(c<0||!engine.mask[c]){$("hover").textContent="Outside active water";return;}
    const display=engine.sample(c,x,y,time),native=engine.native(c,time),weights=engine.bary(c,x,y);let lon=0,lat=0;
    for(let k=0;k<3;k++){const n=data.tri[c*3+k];lon+=data.geo[n*2]*weights[k];lat+=data.geo[n*2+1]*weights[k];}
    $("hover").textContent=lat.toFixed(4)+"°, "+lon.toFixed(4)+"°  ·  element "+(c+1)+"\nNative: "+Math.hypot(...native).toFixed(3)+" m/s  ["+native.map(x=>x.toFixed(3)).join(", ")+"] · "+(meta.vector_basis==="east-north"?"east/north":"grid x/y")+"\nDisplay: "+Math.hypot(...display).toFixed(3)+" m/s  ["+display.map(x=>x.toFixed(3)).join(", ")+"] · grid x/y";
  };
  addEventListener("resize",()=>{if(exporting)deferredResize=true;else resize(true);});
  document.addEventListener("visibilitychange",()=>{last=0;});
  function stepParticles(seconds,t,continuous,wallStart,wallSeconds,tickEnd){
    for(const p of particles){
      if(p.cell<0||p.age>8||!engine.mask[p.cell])engine.spawn(p);
      if(p.cell<0)continue;
      p.path=[];
      if(!engine.integrate(p,seconds,t,continuous,continuous?visualSpeed:1)){engine.spawn(p);continue;}
      const velocity=engine.sample(p.cell,p.x,p.y,continuous?t+seconds:t),index=Math.min(3,Math.floor(Math.sqrt(Math.hypot(...velocity)/meta.vmax)*4));
      if(emissionsEnabled)for(const segment of p.path)trails.append(segment,index,wallStart+segment[4]/seconds*wallSeconds,wallStart+segment[5]/seconds*wallSeconds,tickEnd,tail);
      const [sx,sy]=project(p.x,p.y);if(sx<0||sx>width||sy<0||sy>height)p.age=9;
    }
  }
  function animate(dt){
    const tickEnd=activeTime+dt;trails.history.expire(activeTime-trails.epoch,tail);
    if(mode==="snapshot") {
      // A display gain, explicitly separate from physical time in snapshot mode.
      advanceSnapshot(engine,particles,trails,{...trailView(),visualSpeed,time,tail,vmax:meta.vmax},activeTime,dt,emissionsEnabled);
    } else {
      const rate=(meta.times.at(-1)-meta.times[0])/duration;let remaining=dt*rate,wallOffset=0;
      while(remaining>1e-7) {
        if(time>=meta.times.at(-1)-1e-7){time=meta.times[0];engine.setTime(time,true);upload();reseed();loopCount++;}
        if(engine.blockedGap){playing=false;remaining=0;updateUI();break;}
        const until=meta.times[engine.b]-time,chunk=Math.min(remaining,until);
        if(chunk<=1e-7){engine.setTime(time,true);upload();break;}
        stepParticles(chunk,time,true,activeTime+wallOffset,chunk/rate,tickEnd);time+=chunk;remaining-=chunk;wallOffset+=chunk/rate;
        if(time>=meta.times[engine.b]-1e-7){engine.setTime(time,true);upload();if(engine.maskChanged)clear();}
      }
    }
    if(mode!=="snapshot")particles.forEach(p=>p.age+=dt);
    activeTime=tickEnd;
  }
  function skipStall(elapsed){
    stallCount++;activeTime+=elapsed;accumulator=0;
    if(mode==="continuous"){
      let left=elapsed*(meta.times.at(-1)-meta.times[0])/duration;
      while(left>1e-7){
        if(time>=meta.times.at(-1)-1e-7){time=meta.times[0];loopCount++;}
        if(engine.setTime(time,true))upload();
        if(engine.blockedGap){playing=false;break;}
        const chunk=Math.min(left,meta.times[engine.b]-time);if(chunk<=1e-7)break;
        time+=chunk;left-=chunk;
      }
      if(engine.setTime(time,true))upload();if(engine.blockedGap)playing=false;
    }
    reseed();
  }
  function advance(elapsed){
    if(!Number.isFinite(elapsed)||elapsed<0)throw Error("Invalid elapsed time");
    if(elapsed>.25)skipStall(elapsed);
    else {accumulator+=elapsed;while(accumulator>=tick-1e-10){animate(tick);accumulator=Math.max(0,accumulator-tick);if(engine.blockedGap&&mode==="continuous"){playing=false;accumulator=0;break;}}}
    if(mode==="continuous")drawShade();drawTrails();
  }
  function frame(now){if(last&&playing&&!exporting&&!drag&&!document.hidden){const elapsed=(now-last)/1000,before=performance.now();advance(elapsed);drawTimes.push(performance.now()-before);frameTimes.push(elapsed*1000);if(frameTimes.length>1800){frameTimes.shift();drawTimes.shift();}}
    last=now;if(now-lastUI>150){updateUI();lastUI=now;}requestAnimationFrame(frame);
  }
  $("subtitle").textContent=meta.layer.replaceAll("_"," ")+" currents · "+meta.nodes.toLocaleString()+" nodes · "+meta.elements.toLocaleString()+" triangles";
  $("about-data").textContent=meta.timestamps[0]+" through "+meta.timestamps.at(-1)+". "+meta.timestamps.length+" native records. "+meta.boundary_label+". CRS: "+meta.crs+". Original vector basis: "+meta.vector_basis+". Color scale: square root, 0–"+meta.vmax+" m/s. "+(meta.invalid_geographic_arrays?"Invalid geographic arrays were replaced by a projection of native x/y. ":"")+"No wind field is included.";
  const legend=$("legend-bar").getContext("2d");for(let x=0;x<260;x++){legend.fillStyle="rgb("+color(x/259).join(",")+")";legend.fillRect(x,0,1,14);}$("legend-ticks").replaceChildren(...[0,.25,.5,.75,1].map(t=>{const e=document.createElement("span");e.textContent=(t*t*meta.vmax).toFixed(t?2:0);return e;}));
  try{shader=createFieldRenderer(shade,engine,data,meta,new URLSearchParams(location.search).has("no-webgl"));if(!shader.available)shader=null;}catch(e){console.warn("Shading fallback:",e.message);shader=null;}
  shade.addEventListener("webglcontextlost",e=>{e.preventDefault();shader=null;updateUI();});
  if(!shader){$("shading").checked=false;$("shading").disabled=true;}
  upload();resize(true);updateUI();$("loading").remove();visible=true;
  const loadMs=performance.now()-started;
  function metrics(){const sorted=frameTimes.slice().sort((a,b)=>a-b),p=i=>sorted.length?sorted[Math.floor((sorted.length-1)*i)]:null;return {ready:visible,gifExport:{busy:exporting,...exportStatus},loadMs,renderer:shader?shader.backend:"Canvas fallback",webgl:!!shader,mode,time,playing,visualSpeed,trailDuration:tail,playbackDuration:duration,activeTime,stallCount,...trails.metrics(activeTime,tail),particles:particles.length,frames:frameTimes.length,medianFPS:p(.5)?1000/p(.5):null,p95FrameMs:p(.95),meanDrawMs:drawTimes.reduce((a,b)=>a+b,0)/(drawTimes.length||1),selectionMs:selectionMs.slice(),resetCount,loopCount,stats:{...engine.stats},memory:performance.memory?{used:performance.memory.usedJSHeapSize,total:performance.memory.totalJSHeapSize}:null};}
  async function exportGif(options={}){
    if(mode!=="snapshot")throw Error("Select Snapshot to export a GIF");
    if(exporting)throw Error("A GIF export is already running");
    const state={time,density,tail,visualSpeed,shading:!!shader&&$("shading").checked,forceCanvas:trails.backend!=="WebGL2 instanced",view:trailView(),bounds:viewport()};
    const wasPlaying=playing;exporting=true;setPlaying(false);
    try{const result=await renderSnapshotGif(data,meta,state,{...options,onProgress:p=>{exportStatus=p;options.onProgress?.(p);}});
      exportStatus={phase:"complete",bytes:result.blob.size,elapsedMs:result.report.elapsed_ms,report:result.report};return result;
    }catch(error){exportStatus={phase:error.name==="AbortError"?"cancelled":"failed",error:error.message};throw error;}
    finally{exporting=false;if(deferredResize){deferredResize=false;resize(false);}setPlaying(wasPlaying);}
  }
  let gifAbort=null,gifURLs=[];
  function releaseGif(){for(const url of gifURLs)URL.revokeObjectURL(url);gifURLs=[];$("gif-preview").removeAttribute("src");$("gif-result").hidden=true;}
  function gifDimensions(){const w=+$("gif-width").value;$("gif-dimensions").textContent=w+" \u00d7 "+Math.round(w*height/width)+" pixels \u00b7 "+(w/300).toFixed(1)+" inches wide at 300 pixels/inch. GIF has no physical DPI metadata.";}
  for(const id of ["gif-width","gif-duration","gif-fps"])$(id).oninput=gifDimensions;
  $("export-gif").onclick=()=>{$("gif-field").textContent=meta.title+" \u00b7 "+iso(time)+" \u00b7 "+meta.layer.replaceAll("_"," ");gifDimensions();$("gif-dialog").showModal();};
  $("gif-close").onclick=()=>{$("gif-dialog").close();};
  $("gif-dialog").addEventListener("cancel",()=>gifAbort?.abort());
  $("gif-dialog").addEventListener("close",()=>{gifAbort?.abort();releaseGif();});
  $("gif-cancel").onclick=()=>gifAbort?.abort();
  $("gif-start").onclick=async()=>{
    releaseGif();gifAbort=new AbortController();$("gif-start").disabled=true;$("gif-cancel").hidden=false;$("gif-progress").hidden=false;
    for(const id of ["gif-width","gif-duration","gif-fps"])$(id).disabled=true;
    try{
      const {blob,report}=await exportGif({width:+$("gif-width").value,duration:+$("gif-duration").value,fps:+$("gif-fps").value,signal:gifAbort.signal,
        onProgress:p=>{$("gif-progress").value=p.fraction;$("gif-status").textContent=({preparing:"Preparing map",warming:"Preparing particle tails",encoding:"Encoding frames",complete:"Complete"})[p.phase]+" \u00b7 "+Math.round(p.fraction*100)+"%";}});
      if(!$("gif-dialog").open)return;
      const url=URL.createObjectURL(blob),reportURL=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:"application/json"}));gifURLs=[url,reportURL];
      const name=meta.title.replace(/[^a-z0-9]+/gi,"_").replace(/^_|_$/g,"")+"_"+meta.layer+"_"+report.timestamp_utc.replace(/[-:]/g,"").replace(".000","")+"_"+report.width+"x"+report.height;
      $("gif-preview").src=url;$("gif-download").href=url;$("gif-download").download=name+".gif";$("gif-report").href=reportURL;$("gif-report").download=name+".json";$("gif-result").hidden=false;
      $("gif-status").textContent=(blob.size/1024**2).toFixed(1)+" MiB \u00b7 "+(report.elapsed_ms/1000).toFixed(1)+" seconds to generate"+(report.dropped_segments?" \u00b7 History capacity reached; effective trails "+report.effective_history_seconds.toFixed(2)+" s":"");
    }catch(error){$("gif-status").textContent=error.name==="AbortError"?"Export cancelled. Your snapshot is unchanged.":"Export failed: "+error.message;}
    finally{gifAbort=null;$("gif-start").disabled=false;$("gif-cancel").hidden=true;$("gif-progress").hidden=true;for(const id of ["gif-width","gif-duration","gif-fps"])$(id).disabled=false;}
  };
  window.fvcomTracer={engine,data,meta,trails,metrics,exportGif,select,setMode,project,unproject,setPlaying,setDuration,setVisualSpeed,setTrailDuration,
    resetMetrics:()=>{frameTimes=[];drawTimes=[];selectionMs=[];},
    testAdvance:dt=>{advance(dt);updateUI();},
    setEmissionEnabled:value=>{emissionsEnabled=!!value;},
    setView:bounds=>{scale=Math.min((width-100)/(bounds[2]-bounds[0]),(height-220)/(bounds[3]-bounds[1]));ox=width/2-(bounds[0]+bounds[2])/2*scale;oy=(height-100)/2+(bounds[1]+bounds[3])/2*scale;changeView();reseed();},
    snapshot:()=>particles.slice(0,20).map(({x,y,cell,age})=>({x,y,cell,age}))};
  requestAnimationFrame(frame);
}catch(error){$("loading").textContent="Unable to display this field: "+error.message;console.error(error);window.fvcomTracerError=String(error.stack||error);}
})();
