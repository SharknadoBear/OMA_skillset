/* Run with a browser that has tracer_engine.js loaded. Returns measured checks. */
(() => {
  const results=[];
  function check(name,condition,detail={}){results.push({name,passed:!!condition,...detail});if(!condition)throw Error(name+": "+JSON.stringify(detail));}
  function mesh(points,triangles,fn=(t,x,y)=>[1,.5],times=[0,1,2],wetFn=()=>1){
    const n=triangles.length,xy=new Float64Array(points.flat()),tri=new Int32Array(triangles.flat()),neighbors=new Int32Array(n*3).fill(-1),area=new Float64Array(n),height=new Float32Array(n),uv=new Float32Array(times.length*n*2),wet=new Uint8Array(times.length*n),edges=new Map();
    triangles.forEach((ids,c)=>{
      const [a,b,d]=ids.map(i=>points[i]),det=(b[0]-a[0])*(d[1]-a[1])-(b[1]-a[1])*(d[0]-a[0]);area[c]=det/2;
      height[c]=det/Math.max(...ids.map((id,k)=>Math.hypot(points[id][0]-points[ids[(k+1)%3]][0],points[id][1]-points[ids[(k+1)%3]][1])));
      times.forEach((t,f)=>{uv.set(fn(t,(a[0]+b[0]+d[0])/3,(a[1]+b[1]+d[1])/3,c),(f*n+c)*2);wet[f*n+c]=wetFn(f,c);});
      for(let k=0;k<3;k++){const e=[ids[(k+1)%3],ids[(k+2)%3]].sort((a,b)=>a-b).join(":");if(edges.has(e)){const [p,s]=edges.get(e);neighbors[c*3+k]=p;neighbors[p*3+s]=c;}else edges.set(e,[c,k]);}
    });
    const data={xy,tri,neighbors,area,height,uv,wet},meta={nodes:points.length,elements:n,times,gap_after_indices:[]};
    return new TracerEngine(data,meta);
  }
  function regular(n=10,keep=()=>true,fn){
    const points=[],triangles=[];for(let j=0;j<=n;j++)for(let i=0;i<=n;i++)points.push([-10+20*i/n,-10+20*j/n]);
    for(let j=0;j<n;j++)for(let i=0;i<n;i++){if(!keep(-10+20*(i+.5)/n,-10+20*(j+.5)/n))continue;const a=j*(n+1)+i,b=a+1,c=a+n+1,d=c+1;triangles.push([a,b,d],[a,d,c]);}
    return mesh(points,triangles,fn);
  }
  const uniform=regular(),p={x:-2,y:-1,cell:uniform.locate(-2,-1)};
  check("uniform translation",uniform.integrate(p,2,0,false)&&Math.abs(p.x)<1e-8&&Math.abs(p.y)<1e-8,{position:[p.x,p.y]});
  const zero=regular(10,()=>true,()=>[0,0]),z={x:1,y:1,cell:zero.locate(1,1)};
  check("zero flow remains stationary",zero.integrate(z,10000,0,false)&&z.x===1&&z.y===1);
  const reversal=regular(10,()=>true,t=>[1-t,0]),r={x:0,y:.1,cell:reversal.locate(0,.1)};
  reversal.setTime(0,true);const mid=reversal.sample(r.cell,r.x,r.y,.5);check("linear component midpoint",Math.abs(mid[0]-.5)<1e-7);
  reversal.integrate(r,1,0,true);check("first reversal half distance",Math.abs(r.x-.5)<1e-7);
  reversal.setTime(1,true);reversal.integrate(r,1,1,true);check("reversal returns to origin",Math.abs(r.x)<1e-7);
  function rotation(dt){const e=regular();e.sample=(c,x,y)=>[-.2*y,.2*x];const p={x:4,y:0,cell:e.locate(4,0)};for(let t=0;t<5-1e-9;t+=dt){if(!e.integrate(p,dt,t,false))throw Error("rotation integration failed");}return Math.hypot(p.x-4*Math.cos(1),p.y-4*Math.sin(1));}
  const coarse=rotation(.2),fine=rotation(.1);check("RK2 rotation convergence",coarse/fine>3.5&&fine<.001,{coarse,fine,ratio:coarse/fine});
  const island=regular(20,(x,y)=>!(Math.abs(x)<2&&Math.abs(y)<2));
  check("island endpoints individually wet",island.locate(-8,.3)>=0&&island.locate(8,.3)>=0);
  check("full segment cannot jump island",island.trace(island.locate(-8,.3),-8,.3,8,.3)===-1);
  check("path beside island remains valid",island.trace(island.locate(-8,3.3),-8,3.3,8,3.3)>=0);
  const channel=regular(20,(x,y)=>Math.abs(y)<1);
  check("narrow channel remains traversable",channel.trace(channel.locate(-8,.2),-8,.2,8,.2)>=0);
  check("narrow channel bank blocks escape",channel.trace(channel.locate(-8,.2),-8,.2,-8,2)===-1);
  const sectors=mesh([[0,0],[1,0],[0,1],[-1,0],[0,-1]],[[0,1,2],[0,2,3],[0,3,4],[0,4,1]],(t,x,y,c)=>[c===0?1:9,0],[0,1],(f,c)=>f===0?1:(c%2===0?1:0));
  check("snapshot honors current wet mask",sectors.mask.every(v=>v===1));sectors.setTime(.5,true);
  check("common wet mask across time",sectors.mask[0]===1&&sectors.mask[1]===0);
  check("disconnected vertex fans stay separate",Math.abs(sectors.A[0]-1)<1e-7&&Math.abs(sectors.A[12]-9)<1e-7,{first:sectors.A[0],opposite:sectors.A[12]});
  check("dry edge terminates path",sectors.trace(0,.2,.2,-.2,.2)===-1);
  uniform.m.gap_after_indices=[0];uniform.setTime(.5,true);check("missing interval blocked",uniform.blockedGap&&uniform.a===uniform.b);
  let outside=false;try{uniform.setTime(-1,true);}catch(e){outside=true;}check("no time extrapolation",outside);
  const seedA=regular(),seedB=regular(),aa=seedA.spawn(),bb=seedB.spawn();check("repeatable wet seeds",aa.x===bb.x&&aa.y===bb.y&&seedA.mask[aa.cell]===1);
  for(const gain of [.25,1,2,4]){
    const e=regular(),p={x:-2,y:-1,cell:e.locate(-2,-1),path:[]};e.setTime(0,true);
    check("independent speed "+gain,e.integrate(p,1,0,true,gain)&&Math.abs(p.x-(-2+gain))<1e-7&&Math.abs(p.y-(-1+.5*gain))<1e-7,{position:[p.x,p.y]});
    check("unscaled sampled velocity "+gain,e.sample(p.cell,p.x,p.y,1)[0]===1&&Math.abs(p.path.at(-1)[5]-1)<1e-8);
    const rev=regular(10,()=>true,t=>[1-t,0]),r={x:0,y:.1,cell:rev.locate(0,.1)};rev.setTime(0,true);rev.integrate(r,1,0,true,gain);
    check("scaled reversal intermediate time "+gain,Math.abs(r.x-.5*gain)<1e-7);rev.setTime(1,true);rev.integrate(r,1,1,true,gain);
    check("scaled reversal returns "+gain,Math.abs(r.x)<1e-7);
  }
  function scaledRotation(dt){const e=regular();e.sample=(c,x,y)=>[-.2*y,.2*x];const p={x:4,y:0,cell:e.locate(4,0)};for(let t=0;t<1-1e-9;t+=dt)e.integrate(p,dt,t,false,4);return Math.hypot(p.x-4*Math.cos(.8),p.y-4*Math.sin(.8));}
  const rc=scaledRotation(.1),rf=scaledRotation(.05);check("scaled RK2 rotation convergence",rc/rf>3.5&&rf<.002,{coarse:rc,fine:rf});
  island.sample=()=>[1,0];const ip={x:-8,y:.3,cell:island.locate(-8,.3)};check("accelerated integration stops at island",!island.integrate(ip,4,0,false,4));
  channel.sample=()=>[0,1];const cp={x:-8,y:.2,cell:channel.locate(-8,.2)};check("accelerated integration stops at channel bank",!channel.integrate(cp,1,0,false,4));
  sectors.sample=(c)=>sectors.mask[c]?[-1,0]:null;const dp={x:.2,y:.2,cell:0};check("accelerated integration stops at dry cell",!sectors.integrate(dp,.2,0,true,4));
  return {passed:results.every(r=>r.passed),checks:results};
})()
