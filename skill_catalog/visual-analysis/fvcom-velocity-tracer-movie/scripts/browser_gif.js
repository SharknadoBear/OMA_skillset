/* Run in an initialized offline viewer. Exercises the actual export pipeline. */
(async()=>{
  const f=fvcomTracer,checks=[];
  const check=(name,passed)=>{checks.push({name,passed:!!passed});if(!passed)throw Error(name);};
  f.setPlaying(false);f.setMode("continuous");
  let rejected=false;try{await f.exportGif();}catch(e){rejected=e.message.includes("Snapshot");}
  check("continuous export rejected",rejected&&document.getElementById("export-gif").disabled);
  f.setMode("snapshot");
  for(const options of [{width:1234},{duration:2},{duration:3.5},{fps:30}]){
    rejected=false;try{await f.exportGif(options);}catch(e){rejected=true;}
    check("invalid export option "+JSON.stringify(options),rejected&&!f.metrics().gifExport.busy&&!f.metrics().playing);
  }
  const v={width:1400,height:1000,dpr:1,scale:f.project(1,0)[0]-f.project(0,0)[0],ox:f.project(0,0)[0],oy:f.project(0,0)[1]};
  const state={time:f.meta.times[0],density:30,tail:.2,visualSpeed:2,shading:false,forceCanvas:true,view:v,bounds:[...f.unproject(0,1000),...f.unproject(1400,0)]};
  const outcomes=[];
  for(const [width,fps] of [[1200,10],[2400,20]]){
    const s=new SnapshotGifSession(f.data,f.meta,state,width);
    try{for(let frame=0;frame<fps;frame++){for(let tick=0;tick<60/fps;tick++)s.step();s.frame();}
      outcomes.push(JSON.stringify(s.particles.map(({x,y,cell,age})=>({x,y,cell,age}))));
      check("snapshot source time fixed "+fps,s.engine.a===0&&s.engine.b===0);
    }finally{s.dispose();}
  }
  check("resolution and capture FPS preserve travel",outcomes[0]===outcomes[1]);
  for(const phase of ["warming","encoding"]){
    const before=JSON.stringify({p:f.snapshot(),time:f.metrics().time,active:f.metrics().activeTime,reset:f.metrics().resetCount});
    const controller=new AbortController();let aborted=false;
    try{await f.exportGif({width:1200,duration:3,fps:10,signal:controller.signal,onProgress:p=>{if(p.phase===phase)controller.abort();}});}catch(e){aborted=e.name==="AbortError";}
    check("cancel during "+phase,aborted&&!f.metrics().gifExport.busy);
    check("cancel preserves live state "+phase,before===JSON.stringify({p:f.snapshot(),time:f.metrics().time,active:f.metrics().activeTime,reset:f.metrics().resetCount}));
  }
  // Worker syntax failures must be surfaced even if they occur during warm-up.
  const source=document.getElementById("gif-worker-source"),saved=source.textContent;
  try{source.textContent="this is invalid javascript !!!";let error=false;
    try{await f.exportGif({width:1200,duration:3,fps:10});}catch(e){error=true;}
    check("worker startup failure recovers",error&&!f.metrics().gifExport.busy&&!f.metrics().playing);
  }finally{source.textContent=saved;}
  // Decoding is checked independently in Python: black background must return
  // exactly after every segment and antialiased edge has expired.
  window.gifExpiryFixtures=[];
  for(const forceCanvas of [false,true]){
    const trails=new TrailRenderer(document.createElement("canvas"),{forceCanvas});
    const view={width:160,height:100,dpr:1,scale:1,ox:0,oy:100};trails.resize(160,100,1);
    trails.append([20,50,135,50,0,0],3,0,0,0,.5);
    const c=document.createElement("canvas");c.width=160;c.height=100;const ctx=c.getContext("2d");
    const worker=createGifWorker();
    const request=(m,transfers=[])=>new Promise((resolve,reject)=>{worker.onmessage=({data:r})=>r.error?reject(Error(r.error)):resolve(r);worker.onerror=e=>reject(Error(e.message));worker.postMessage(m,transfers);});
    try{for(let i=0;i<21;i++){
      trails.draw(i/20,.5,view);ctx.fillStyle="#000";ctx.fillRect(0,0,160,100);ctx.drawImage(trails.canvas,0,0);
      const rgba=ctx.getImageData(0,0,160,100).data;await request({kind:"frame",width:160,height:100,fps:20,rgba:rgba.buffer},[rgba.buffer]);
    }
      check("source trail expiry "+trails.backend,trails.alphaPixels().nonzero===0);
      const result=await request({kind:"finish"}),url=URL.createObjectURL(new Blob([result.bytes],{type:"image/gif"}));
      window.gifExpiryFixtures.push({backend:trails.backend,url});
    }finally{worker.terminate();trails.dispose();}
  }
  return {passed:true,checks};
})()
