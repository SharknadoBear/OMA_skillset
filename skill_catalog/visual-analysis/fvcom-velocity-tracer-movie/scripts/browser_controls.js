/* Live-viewer speed, clock, finite history, and reset contracts. Runs paused. */
(() => {
  const f=fvcomTracer,checks=[];function check(name,ok,detail={}){checks.push({name,passed:!!ok,...detail});if(!ok)throw Error(name+": "+JSON.stringify(detail));}
  const first=f.meta.times[0],span=f.meta.times.at(-1)-first,oldGaps=f.meta.gap_after_indices.slice();
  f.setPlaying(false);f.setMode("snapshot");f.setVisualSpeed(1.5);f.setTrailDuration(3.5);
  f.setMode("continuous");check("continuous preset",f.metrics().trailDuration===1&&f.metrics().visualSpeed===1);
  check("continuous speed slider enabled",!document.getElementById("speed").disabled);
  f.setVisualSpeed(2);f.setTrailDuration(.6);f.setMode("snapshot");check("snapshot preferences remembered",f.metrics().visualSpeed===1.5&&f.metrics().trailDuration===3.5);
  f.setMode("continuous");check("continuous preferences remembered",f.metrics().visualSpeed===2&&f.metrics().trailDuration===.6);
  f.setDuration(60);f.setTrailDuration(1);const outcomes=[];
  for(const gain of [.25,1,2,4]){
    f.engine.seed=19460907;f.select(first);f.setVisualSpeed(gain);
    for(let i=0;i<60;i++)f.testAdvance(1/60);
    const m=f.metrics();outcomes.push({gain,time:m.time,position:f.snapshot()[0]});
    check("field clock independent of speed "+gain,Math.abs(m.time-first-span/60)<1e-4,{time:m.time});
    check("motion annotation "+gain,document.getElementById("motion-note").textContent.includes("Illustrative")===(gain!==1));
  }
  check("speed changes real particle motion",Math.abs(outcomes[0].position.x-outcomes.at(-1).position.x)>1e-4,{outcomes});
  const saved=f.metrics(),particles=JSON.stringify(f.snapshot()),native=JSON.stringify(f.engine.native(0,f.metrics().time));
  f.setVisualSpeed(1.25);f.setDuration(300);
  check("live settings preserve particles and time",JSON.stringify(f.snapshot())===particles&&f.metrics().time===saved.time&&f.metrics().resetCount===saved.resetCount);
  check("native velocity remains unscaled",JSON.stringify(f.engine.native(0,f.metrics().time))===native);
  f.setTrailDuration(.2);check("shorter setting removes old history",f.metrics().retainedSegments<saved.retainedSegments&&f.metrics().resetCount===saved.resetCount);
  f.setDuration(60);f.setVisualSpeed(1);f.setTrailDuration(1);
  const invariance=[];
  for(const fps of [30,60,120]){
    f.engine.seed=19460907;f.select(first);f.setEmissionEnabled(true);
    for(let i=0;i<fps;i++)f.testAdvance(1/fps);
    const before=f.trails.alphaPixels(),positions=f.snapshot(),resets=f.metrics().resetCount;
    f.setEmissionEnabled(false);for(let i=0;i<=fps;i++)f.testAdvance(1/fps);
    const after=f.trails.alphaPixels();check("live field exact expiry "+fps,before.nonzero>0&&after.nonzero===0&&f.metrics().resetCount===resets,{before,after});
    invariance.push({fps,positions,alphaSum:before.sum});
  }
  check("fixed ticks independent of display frequency",invariance.every(row=>row.positions.every((p,i)=>Math.hypot(p.x-invariance[0].positions[i].x,p.y-invariance[0].positions[i].y)<1e-7)),{alphaSums:invariance.map(x=>x.alphaSum)});
  check("frame-rate brightness agreement",Math.max(...invariance.map(x=>x.alphaSum))/Math.min(...invariance.map(x=>x.alphaSum))<1.01);
  f.setEmissionEnabled(true);f.select(first);const m=f.metrics();f.testAdvance(.5);
  check("visible stall advances clock and clears",Math.abs(f.metrics().time-first-span/120)<1e-4&&f.metrics().stallCount===m.stallCount+1&&f.metrics().retainedSegments===0);
  f.meta.gap_after_indices=[0];f.select(first);f.testAdvance(.5);check("stall cannot bridge missing records",f.metrics().time===first&&!f.metrics().playing&&f.engine.blockedGap);
  f.meta.gap_after_indices=oldGaps;f.select(first);
  f.setMode("snapshot");f.setVisualSpeed(1);f.setTrailDuration(3);f.setMode("continuous");f.setVisualSpeed(1);f.setTrailDuration(1);f.setDuration(60);f.setMode("snapshot");f.select(first);
  return {passed:checks.every(x=>x.passed),checks};
})()
