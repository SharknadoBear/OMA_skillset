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
  function compile(gl,type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
  function initGL(){
    if(new URLSearchParams(location.search).has("no-webgl"))return null;
    const gl=shade.getContext("webgl2",{alpha:true,antialias:true,preserveDrawingBuffer:true});if(!gl)return null;
    const vs=`#version 300 es
      in vec2 pos; in vec2 velA; in vec2 velB; in float wet;
      uniform vec2 size; uniform vec3 view; uniform float alpha;
      out vec2 velocity; flat out float valid;
      void main(){vec2 p=vec2(pos.x*view.x+view.y,view.z-pos.y*view.x);gl_Position=vec4(p.x/size.x*2.-1.,1.-p.y/size.y*2.,0.,1.);velocity=mix(velA,velB,alpha);valid=wet;}`;
    const fs=`#version 300 es
      precision highp float; in vec2 velocity; flat in float valid; uniform float vmax; out vec4 pixel;
      vec3 colors(float x){float q=clamp(x,0.,1.)*5.;
      vec3 a=vec3(10,27,55),b=vec3(18,68,107),c=vec3(19,123,133),d=vec3(79,165,123),e=vec3(196,185,102),f=vec3(236,113,70);
      if(q<1.)return mix(a,b,q)/255.;if(q<2.)return mix(b,c,q-1.)/255.;if(q<3.)return mix(c,d,q-2.)/255.;if(q<4.)return mix(d,e,q-3.)/255.;return mix(e,f,q-4.)/255.;}
      void main(){if(valid<.5)discard;pixel=vec4(colors(sqrt(length(velocity)/vmax)),.83);}`;
    const program=gl.createProgram();gl.attachShader(program,compile(gl,gl.VERTEX_SHADER,vs));gl.attachShader(program,compile(gl,gl.FRAGMENT_SHADER,fs));gl.linkProgram(program);
    if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
    gl.useProgram(program);const buffers={};
    function buffer(name,size,values,dynamic=false){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,values,dynamic?gl.DYNAMIC_DRAW:gl.STATIC_DRAW);const loc=gl.getAttribLocation(program,name);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,gl.FLOAT,false,0,0);buffers[name]=b;}
    const positions=new Float32Array(meta.elements*6);
    for(let i=0;i<data.tri.length;i++){const n=data.tri[i];positions[2*i]=data.xy[2*n];positions[2*i+1]=data.xy[2*n+1];}
    buffer("pos",2,positions);buffer("velA",2,engine.A,true);buffer("velB",2,engine.B,true);buffer("wet",1,new Float32Array(meta.elements*3),true);
    const uniforms={};for(const name of ["size","view","alpha","vmax"])uniforms[name]=gl.getUniformLocation(program,name);
    const ext=gl.getExtension("WEBGL_debug_renderer_info");
    return {gl,program,buffers,uniforms,renderer:ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER)};
  }
  function upload(){if(!shader)return;const {gl,buffers}=shader;for(const [name,a] of [["velA",engine.A],["velB",engine.B]]){gl.bindBuffer(gl.ARRAY_BUFFER,buffers[name]);gl.bufferData(gl.ARRAY_BUFFER,a,gl.DYNAMIC_DRAW);}
    const wet=new Float32Array(meta.elements*3);for(let c=0;c<meta.elements;c++)wet.fill(engine.mask[c],c*3,c*3+3);
    gl.bindBuffer(gl.ARRAY_BUFFER,buffers.wet);gl.bufferData(gl.ARRAY_BUFFER,wet,gl.DYNAMIC_DRAW);
  }
  function drawShade(){if(!shader)return;const {gl,uniforms:u}=shader;gl.viewport(0,0,shade.width,shade.height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);if(!$("shading").checked)return;
    gl.uniform2f(u.size,width,height);gl.uniform3f(u.view,scale,ox,oy);gl.uniform1f(u.alpha,engine.alpha(time));gl.uniform1f(u.vmax,meta.vmax);gl.drawArrays(gl.TRIANGLES,0,meta.elements*3);
  }
  function drawMap(){map.clearRect(0,0,width,height);map.lineWidth=.8;
    for(let type=0;type<2;type++){map.beginPath();map.strokeStyle=type?"#a5c1cc66":"#a9c3cf99";map.setLineDash(type?[4,5]:[]);
      for(let i=0;i<data.boundaryType.length;i++)if(data.boundaryType[i]===type){const a=data.boundary[i*2],b=data.boundary[i*2+1];map.moveTo(...project(data.xy[a*2],data.xy[a*2+1]));map.lineTo(...project(data.xy[b*2],data.xy[b*2+1]));}map.stroke();}
    map.setLineDash([]);const desired=100/scale,power=Math.pow(10,Math.floor(Math.log10(desired))),distance=[1,2,5,10].map(x=>x*power).filter(x=>x<=desired).pop()||power;
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
  addEventListener("resize",()=>resize(true));
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
      const gain=12/Math.max(scale*meta.vmax*.125,1e-9)*visualSpeed;
      stepParticles(dt*gain,time,false,activeTime,dt,tickEnd);
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
    particles.forEach(p=>p.age+=dt);
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
  function frame(now){if(last&&playing&&!drag&&!document.hidden){const elapsed=(now-last)/1000,before=performance.now();advance(elapsed);drawTimes.push(performance.now()-before);frameTimes.push(elapsed*1000);if(frameTimes.length>1800){frameTimes.shift();drawTimes.shift();}}
    last=now;if(now-lastUI>150){updateUI();lastUI=now;}requestAnimationFrame(frame);
  }
  $("subtitle").textContent=meta.layer.replaceAll("_"," ")+" currents · "+meta.nodes.toLocaleString()+" nodes · "+meta.elements.toLocaleString()+" triangles";
  $("about-data").textContent=meta.timestamps[0]+" through "+meta.timestamps.at(-1)+". "+meta.timestamps.length+" native records. "+meta.boundary_label+". CRS: "+meta.crs+". Original vector basis: "+meta.vector_basis+". Color scale: square root, 0–"+meta.vmax+" m/s. "+(meta.invalid_geographic_arrays?"Invalid geographic arrays were replaced by a projection of native x/y. ":"")+"No wind field is included.";
  const legend=$("legend-bar").getContext("2d");for(let x=0;x<260;x++){legend.fillStyle="rgb("+color(x/259).join(",")+")";legend.fillRect(x,0,1,14);}$("legend-ticks").replaceChildren(...[0,.25,.5,.75,1].map(t=>{const e=document.createElement("span");e.textContent=(t*t*meta.vmax).toFixed(t?2:0);return e;}));
  try{shader=initGL();}catch(e){console.warn("Shading fallback:",e.message);shader=null;}
  shade.addEventListener("webglcontextlost",e=>{e.preventDefault();shader=null;updateUI();});
  if(!shader){$("shading").checked=false;$("shading").disabled=true;}
  upload();resize(true);updateUI();$("loading").remove();visible=true;
  const loadMs=performance.now()-started;
  function metrics(){const sorted=frameTimes.slice().sort((a,b)=>a-b),p=i=>sorted.length?sorted[Math.floor((sorted.length-1)*i)]:null;return {ready:visible,loadMs,renderer:shader?shader.renderer:"Canvas fallback",webgl:!!shader,mode,time,playing,visualSpeed,trailDuration:tail,playbackDuration:duration,activeTime,stallCount,...trails.metrics(activeTime,tail),particles:particles.length,frames:frameTimes.length,medianFPS:p(.5)?1000/p(.5):null,p95FrameMs:p(.95),meanDrawMs:drawTimes.reduce((a,b)=>a+b,0)/(drawTimes.length||1),selectionMs:selectionMs.slice(),resetCount,loopCount,stats:{...engine.stats},memory:performance.memory?{used:performance.memory.usedJSHeapSize,total:performance.memory.totalJSHeapSize}:null};}
  window.fvcomTracer={engine,data,meta,trails,metrics,select,setMode,project,unproject,setPlaying,setDuration,setVisualSpeed,setTrailDuration,
    resetMetrics:()=>{frameTimes=[];drawTimes=[];selectionMs=[];},
    testAdvance:dt=>{advance(dt);updateUI();},
    setEmissionEnabled:value=>{emissionsEnabled=!!value;},
    setView:bounds=>{scale=Math.min((width-100)/(bounds[2]-bounds[0]),(height-220)/(bounds[3]-bounds[1]));ox=width/2-(bounds[0]+bounds[2])/2*scale;oy=(height-100)/2+(bounds[1]+bounds[3])/2*scale;changeView();reseed();},
    snapshot:()=>particles.slice(0,20).map(({x,y,cell,age})=>({x,y,cell,age}))};
  requestAnimationFrame(frame);
}catch(error){$("loading").textContent="Unable to display this field: "+error.message;console.error(error);window.fvcomTracerError=String(error.stack||error);}
})();
