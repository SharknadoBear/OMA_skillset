/* Finite-age renderer and bounded history tests in the actual browser backend. */
(() => {
  const checks=[];function check(name,ok,detail={}){checks.push({name,passed:!!ok,...detail});if(!ok)throw Error(name+": "+JSON.stringify(detail));}
  const view={width:96,height:64,dpr:1,scale:1,ox:0,oy:64};
  for(const forceCanvas of [false,true]){
    const canvas=document.createElement("canvas");document.body.append(canvas);const r=new TrailRenderer(canvas,{forceCanvas});r.resize(96,64,1);
    const backend=r.backend;
    for(const fps of [30,60,120]){
      r.clear(0);r.append([10,32,80,32],3,0,0,0,1);r.draw(0,1,view);const start=r.alphaPixels();check(backend+" visible start "+fps,start.nonzero>0);
      let halfway=0;for(let i=1;i<=fps+1;i++){r.draw(i/fps,1,view);if(i===fps/2)halfway=r.alphaPixels().sum;}
      const end=r.alphaPixels();check(backend+" zero expiry "+fps,end.nonzero===0,{start,halfway,end});
    }
    r.clear(0);r.append([10,32,80,32],3,0,.1,.1,3);r.draw(.7,3,view);check(backend+" longer history visible",r.alphaPixels().nonzero>0);
    r.draw(.7,.2,view);check(backend+" shortening expires immediately",r.alphaPixels().nonzero===0);
    r.clear(0);r.append([10,32,80,32],3,0,0,0,1);for(const t of [.011,.057,.201,.249,.381,.601,.997,1.03])r.draw(t,1,view);
    check(backend+" irregular interval expiry",r.alphaPixels().nonzero===0);
    r.clear(0);r.append([10,32,80,32],3,0,0,0,1);r.draw(.3,1,view);const a=r.alphaPixels();r.draw(.3,1,view);check(backend+" redraw cannot accumulate",JSON.stringify(a)===JSON.stringify(r.alphaPixels()));
    if(r.gl)r.gl.getExtension("WEBGL_lose_context")?.loseContext();r.canvas.remove();
  }
  const h=new TrailHistory(128);for(let i=0;i<8;i++)h.push([0,0,1,1,i,i,0,i],8);
  check("history capacity bounded",h.count===4&&h.records.byteLength===128&&h.dropped===4);
  check("capacity disclosure",h.metrics(8,8).historyLimited&&h.metrics(8,8).effectiveHistorySeconds===4);
  h.expire(20,8);check("history fully expires",h.count===0);h.clear();check("history reset clears capacity notice",!h.metrics(20,8).historyLimited);
  return {passed:checks.every(x=>x.passed),checks};
})()
