/* Deterministic, offline snapshot GIF export. Scientific arrays stay unscaled. */
"use strict";
class SnapshotGifSession {
  constructor(data,meta,state,width){
    this.data=data;this.meta=meta;this.state=state;this.time=0;
    this.view={...state.view,dpr:width/state.view.width};
    this.width=width;this.height=Math.round(state.view.height*this.view.dpr);
    this.engine=new TracerEngine(data,meta);this.engine.seed=19460907;
    this.engine.setTime(state.time,false);this.engine.setViewport(state.bounds);
    this.particles=Array.from({length:state.density},()=>{const p=this.engine.spawn();p.age=this.engine.random()*8;return p;});
    this.base=document.createElement("canvas");this.base.width=width;this.base.height=this.height;
    this.output=this.base.cloneNode(false);this.ctx=this.output.getContext("2d",{willReadFrequently:true});
    this.trails=new TrailRenderer(document.createElement("canvas"),{forceCanvas:state.forceCanvas});
    this.trails.resize(this.view.width,this.view.height,this.view.dpr);
    try{this.drawBase();}catch(error){this.dispose();throw error;}
    this.maxHistoryBytes=0;this.droppedSegments=0;
  }
  drawBase(){
    const c=this.base.getContext("2d"),v=this.view,m=this.meta;
    c.fillStyle="#0c1721";c.fillRect(0,0,this.width,this.height);
    if(this.state.shading){
      const canvas=this.base.cloneNode(false);let field;
      try{field=createFieldRenderer(canvas,this.engine,this.data,m);if(!field.available)throw Error("Export shading unavailable; disable shading and retry");
        field.draw(v,this.state.time,true);c.drawImage(canvas,0,0);
      }finally{field?.dispose();canvas.width=canvas.height=1;}
    }
    c.setTransform(v.dpr,0,0,v.dpr,0,0);drawMeshBoundary(c,this.data,v);
  }
  labels(c){
    const v=this.view,m=this.meta,pad=24,legendWidth=Math.min(260,v.width*.36),font=Math.max(10,Math.min(14,v.width/70));
    c.save();c.setTransform(v.dpr,0,0,v.dpr,0,0);
    c.fillStyle="#0c1721";c.fillRect(0,0,v.width,92);
    c.fillStyle="#eff6f7";c.font=`500 ${font*1.6}px Segoe UI, Arial, sans-serif`;
    c.fillText(m.title,pad,32,v.width-pad*2);
    c.font=`${font}px Segoe UI, Arial, sans-serif`;c.fillStyle="#c5d8e1";
    c.fillText(new Date(this.state.time*1000).toISOString().replace("T"," ").replace(".000Z"," UTC")+"  ·  "+m.layer.replaceAll("_"," ")+" currents",pad,56,v.width-pad*2);
    c.font=`${font*.85}px Segoe UI, Arial, sans-serif`;c.fillStyle="#9ebac9";
    c.fillText(`Fixed snapshot · illustrative visual speed ×${this.state.visualSpeed}`+(this.state.shading?"":" · speed shading off"),pad,77,v.width-pad*2);
    const x=v.width-pad-legendWidth,y=v.height-62;
    c.fillStyle="#0c1721";c.fillRect(x-12,y-30,legendWidth+24,86);
    c.fillStyle="#d9e9ed";c.font=`${font*.85}px Segoe UI, Arial, sans-serif`;c.fillText("CURRENT SPEED  m/s",x,y-9);
    const palette=[[10,27,55],[18,68,107],[19,123,133],[79,165,123],[196,185,102],[236,113,70]],gradient=c.createLinearGradient(x,0,x+legendWidth,0);
    palette.forEach((rgb,i)=>gradient.addColorStop(i/5,`rgb(${rgb})`));c.fillStyle=gradient;c.fillRect(x,y,legendWidth,10);
    c.fillStyle="#c5d8e1";
    for(const t of [0,.25,.5,.75,1]){c.textAlign=t===0?"left":t===1?"right":"center";c.fillText((t*t*m.vmax).toFixed(t?2:0),x+t*legendWidth,y+28);}
    c.textAlign="center";
    const desired=Math.min(120,v.width*.2)/v.scale,power=10**Math.floor(Math.log10(desired)),distance=[1,2,5,10].map(n=>n*power).filter(n=>n<=desired).pop()||power;
    const len=distance*v.scale;c.fillStyle="#0c1721";c.fillRect(pad-8,v.height-75,len+16,55);
    c.fillStyle="#c5d8e1";c.fillText(distance>=1000?distance/1000+" km":distance+" m",pad+len/2,v.height-50);
    c.strokeStyle="#c5d8e1";c.lineWidth=1.5;c.beginPath();c.moveTo(pad,v.height-43);c.lineTo(pad+len,v.height-43);c.stroke();c.restore();
  }
  step(){
    this.time=advanceSnapshot(this.engine,this.particles,this.trails,{...this.view,...this.state,vmax:this.meta.vmax},this.time,1/60);
    this.maxHistoryBytes=Math.max(this.maxHistoryBytes,this.trails.history.records.byteLength);
    this.droppedSegments=Math.max(this.droppedSegments,this.trails.history.dropped);
  }
  frame(){
    if(this.state.forceCanvas===false&&this.trails.backend!=="WebGL2 instanced")throw Error("Export trail graphics context lost; retry the export");
    this.trails.draw(this.time,this.state.tail,this.view);
    this.ctx.drawImage(this.base,0,0);this.ctx.drawImage(this.trails.canvas,0,0);this.labels(this.ctx);
    return this.ctx.getImageData(0,0,this.width,this.height).data;
  }
  dispose(){this.trails?.dispose();for(const c of [this.base,this.output])if(c)c.width=c.height=1;}
}
function createGifWorker(){
  const code='(()=>{const module={exports:{}},exports=module.exports;\n'+document.getElementById("gifenc-source").textContent+'\nself.gifenc=module.exports;})();\n'+document.getElementById("gif-worker-source").textContent;
  const url=URL.createObjectURL(new Blob([code],{type:"text/javascript"}));
  try{return new Worker(url);}finally{URL.revokeObjectURL(url);}
}
async function renderSnapshotGif(data,meta,state,{width=2400,duration=5,fps=20,onProgress=()=>{},signal}={}){
  const height=Math.round(width*state.view.height/state.view.width);
  if(![1200,1800,2400].includes(width))throw Error("GIF width must be 1200, 1800, or 2400 pixels");
  if(!Number.isInteger(duration)||duration<3||duration>10)throw Error("GIF duration must be 3–10 whole seconds");
  if(![10,20].includes(fps))throw Error("GIF frame rate must be 10 or 20 FPS");
  if(height<1||width*height>8000000)throw Error("GIF exceeds eight megapixels; reduce width or resize the viewer");
  const check=()=>{if(signal?.aborted)throw new DOMException("GIF export cancelled","AbortError");};check();
  const start=performance.now();let session,worker,pending,workerFailure,peakHeap=0;
  const progress=(phase,done,total)=>{peakHeap=Math.max(peakHeap,performance.memory?.usedJSHeapSize||0);onProgress({phase,done,total,fraction:done/total});};
  const cancel=()=>{pending?.reject(new DOMException("GIF export cancelled","AbortError"));pending=null;worker?.terminate();};
  const request=(message,transfer=[])=>new Promise((resolve,reject)=>{check();if(workerFailure)throw workerFailure;pending={resolve,reject};worker.postMessage(message,transfer);});
  try{
    worker=createGifWorker();worker.onmessage=({data:r})=>{const p=pending;pending=null;if(r.error)p?.reject(Error(r.error));else p?.resolve(r);};
    worker.onerror=e=>{e.preventDefault();workerFailure=Error(e.message||"GIF worker failed");pending?.reject(workerFailure);pending=null;};
    signal?.addEventListener("abort",cancel,{once:true});
    progress("preparing",0,1);await new Promise(r=>setTimeout(r,0));check();
    session=new SnapshotGifSession(data,meta,state,width);
    const warm=Math.ceil(state.tail*60);
    for(let i=0;i<warm;i++){check();session.step();if(i%12===11){progress("warming",i+1,warm);await new Promise(r=>setTimeout(r,0));}}
    const count=duration*fps;let encodedBytes=0;
    for(let f=0;f<count;f++){
      check();if(f)for(let i=0;i<60/fps;i++)session.step();
      const rgba=session.frame();const r=await request({kind:"frame",rgba:rgba.buffer,width,height,fps},[rgba.buffer]);
      encodedBytes=r.bytes;progress("encoding",f+1,count);
    }
    check();const r=await request({kind:"finish"});check();
    const blob=new Blob([r.bytes],{type:"image/gif"}),origin=meta.origin||[0,0];
    const report={schema:"fvcom_snapshot_gif_v1",case:meta.title,layer:meta.layer,timestamp_utc:new Date(state.time*1000).toISOString(),
      sources:meta.sources,scientific_arrays:meta.arrays,crs:meta.crs,vector_basis:meta.vector_basis,
      projected_extent:state.bounds.map((x,i)=>x+origin[i%2]),local_extent:state.bounds,logical_view:state.view,
      width,height,duration_seconds:duration,fps,frames:count,loop:"forward, infinite; restart seam allowed",intended_ppi:300,intended_width_inches:width/300,
      appearance:{particles:state.density,trail_seconds:state.tail,visual_speed:state.visualSpeed,shading:state.shading,vmax_mps:meta.vmax,color_normalization:"square root"},
      seed:19460907,warmup_seconds:warm/60,tick_hz:60,trail_backend:session.trails.backend,
      trail_history_peak_bytes:session.maxHistoryBytes,dropped_segments:session.droppedSegments,
      effective_history_seconds:session.trails.metrics(session.time,state.tail).effectiveHistorySeconds,
      encoder:meta.gif_export.encoder,bytes:blob.size,elapsed_ms:performance.now()-start,peak_js_heap_bytes:peakHeap||null,
      frame_buffers_in_flight:1,worker_encoding:true,palette_mapping:"exact_rgb24_clip_cache",palette_cache_bytes:16777216,particle_checks:session.engine.stats,
      final_particles:session.particles.slice(0,20).map(({x,y,cell,age})=>({x,y,cell,age}))};
    progress("complete",count,count);return {blob,report};
  }finally{signal?.removeEventListener("abort",cancel);worker?.terminate();session?.dispose();}
}
window.SnapshotGifSession=SnapshotGifSession;window.createGifWorker=createGifWorker;window.renderSnapshotGif=renderSnapshotGif;
