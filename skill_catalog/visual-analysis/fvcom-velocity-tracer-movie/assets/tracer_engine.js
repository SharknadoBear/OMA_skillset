/* Native triangular field sampler and bounded midpoint particle integrator.
 * Renderer lifecycle adaptations are in viewer.js; this numerical core is OMA code.
 */
"use strict";
class TracerEngine {
  constructor(data, metadata) {
    this.d = data; this.m = metadata; this.n = metadata.elements;
    this.seed = 19460907; this.pairKey = "";
    this.stats = {segments:0, boundaryStops:0, dryStops:0, substepLimits:0, reseeds:0};
    this.fans = Array.from({length: metadata.nodes}, () => []);
    this.coeff = new Float64Array(this.n*6);
    this.box = [Infinity,Infinity,-Infinity,-Infinity];
    for(let i=0;i<metadata.nodes;i++) {
      const x=data.xy[2*i], y=data.xy[2*i+1];
      this.box[0]=Math.min(this.box[0],x); this.box[1]=Math.min(this.box[1],y);
      this.box[2]=Math.max(this.box[2],x); this.box[3]=Math.max(this.box[3],y);
    }
    this.spatial = Array.from({length:128*128},()=>[]);
    this.cellBox = new Float64Array(this.n*4);
    for(let c=0;c<this.n;c++) {
      const ids=[data.tri[c*3],data.tri[c*3+1],data.tri[c*3+2]];
      ids.forEach((n,k)=>this.fans[n].push(c*3+k));
      const [a,b,e]=ids, ax=data.xy[a*2], ay=data.xy[a*2+1];
      const bx=data.xy[b*2]-ax, by=data.xy[b*2+1]-ay;
      const cx=data.xy[e*2]-ax, cy=data.xy[e*2+1]-ay, det=bx*cy-by*cx;
      this.coeff.set([ax,ay,cy/det,-cx/det,-by/det,bx/det],c*6);
      const xs=ids.map(n=>data.xy[n*2]),ys=ids.map(n=>data.xy[n*2+1]);
      const bounds=[Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)];
      this.cellBox.set(bounds,c*4);
      const l=this.bin(bounds[0],bounds[1]),r=this.bin(bounds[2],bounds[3]);
      for(let j=l[1];j<=r[1];j++) for(let i=l[0];i<=r[0];i++) this.spatial[j*128+i].push(c);
    }
    this.splitNodes=[];
    for(let n=0;n<this.fans.length;n++) {
      if(this.sectors(this.fans[n],null).length>1) this.splitNodes.push(n);
    }
    this.setTime(metadata.times[0],false);
    this.setViewport(this.box);
  }
  bin(x,y) {
    return [Math.max(0,Math.min(127,Math.floor((x-this.box[0])/(this.box[2]-this.box[0])*128))),
      Math.max(0,Math.min(127,Math.floor((y-this.box[1])/(this.box[3]-this.box[1])*128)))];
  }
  random() { let s=this.seed; s^=s<<13; s^=s>>>17; s^=s<<5; this.seed=s>>>0; return this.seed/4294967296; }
  bary(c,x,y) {
    const q=c*6,dx=x-this.coeff[q],dy=y-this.coeff[q+1];
    const b=this.coeff[q+2]*dx+this.coeff[q+3]*dy,d=this.coeff[q+4]*dx+this.coeff[q+5]*dy;
    return [1-b-d,b,d];
  }
  locate(x,y) {
    if(x<this.box[0]||x>this.box[2]||y<this.box[1]||y>this.box[3]) return -1;
    const [i,j]=this.bin(x,y);
    for(const c of this.spatial[j*128+i]) if(this.bary(c,x,y).every(v=>v>=-1e-9)) return c;
    return -1;
  }
  sectors(fan,mask) {
    const todo=new Set(fan.filter(k=>!mask||mask[Math.floor(k/3)])),groups=[];
    while(todo.size) {
      const stack=[todo.values().next().value],group=[];
      while(stack.length) {
        const corner=stack.pop(); if(!todo.delete(corner)) continue;
        group.push(corner);
        const cell=Math.floor(corner/3),k=corner%3;
        for(const side of [(k+1)%3,(k+2)%3]) {
          const neighbor=this.d.neighbors[cell*3+side];
          for(const candidate of todo) if(Math.floor(candidate/3)===neighbor) stack.push(candidate);
        }
      }
      groups.push(group);
    }
    return groups;
  }
  reconstruct(frame,mask) {
    const sums=new Float64Array(this.m.nodes*3),out=new Float32Array(this.n*6),offset=frame*this.n*2;
    const affected=new Set(this.splitNodes);
    for(let c=0;c<this.n;c++) {
      const area=this.d.area[c];
      for(let k=0;k<3;k++) {
        const node=this.d.tri[c*3+k];
        if(!mask[c]) { affected.add(node); continue; }
        sums[node*3]+=this.d.uv[offset+c*2]*area;
        sums[node*3+1]+=this.d.uv[offset+c*2+1]*area;
        sums[node*3+2]+=area;
      }
    }
    for(let c=0;c<this.n;c++) if(mask[c]) for(let k=0;k<3;k++) {
      const node=this.d.tri[c*3+k],w=sums[node*3+2];
      out[c*6+k*2]=sums[node*3]/w; out[c*6+k*2+1]=sums[node*3+1]/w;
    }
    for(const node of affected) for(const group of this.sectors(this.fans[node],mask)) {
      let u=0,v=0,w=0;
      for(const corner of group) { const c=Math.floor(corner/3),a=this.d.area[c]; u+=this.d.uv[offset+c*2]*a;v+=this.d.uv[offset+c*2+1]*a;w+=a; }
      for(const corner of group) {out[corner*2]=u/w;out[corner*2+1]=v/w;}
    }
    return out;
  }
  bracket(t,continuous) {
    const ts=this.m.times;
    if(t<ts[0]-1e-6||t>ts[ts.length-1]+1e-6) throw Error("Time outside data coverage");
    let lo=0,hi=ts.length-1;
    while(lo<hi) {const mid=Math.ceil((lo+hi)/2); if(ts[mid]<=t)lo=mid;else hi=mid-1;}
    return [lo,continuous&&lo<ts.length-1 ? lo+1:lo];
  }
  setTime(t,continuous) {
    const [a,b]=this.bracket(t,continuous); this.time=t;
    this.blockedGap=continuous&&a!==b&&this.m.gap_after_indices.includes(a);
    const actualB=this.blockedGap?a:b,key=a+":"+actualB;
    if(key===this.pairKey) return false;
    this.a=a;this.b=actualB;this.pairKey=key;
    const mask=new Uint8Array(this.n);this.maskChanged=false;
    for(let c=0;c<this.n;c++) {
      mask[c]=this.d.wet[a*this.n+c]&&this.d.wet[actualB*this.n+c]?1:0;
      if(this.mask&&mask[c]!==this.mask[c])this.maskChanged=true;
    }
    this.mask=mask;this.A=this.reconstruct(a,mask);this.B=a===actualB?this.A:this.reconstruct(actualB,mask);
    if(this.viewport)this.setViewport(this.viewport);
    return true;
  }
  alpha(t) {return this.a===this.b?0:Math.max(0,Math.min(1,(t-this.m.times[this.a])/(this.m.times[this.b]-this.m.times[this.a])));}
  sample(c,x,y,t) {
    if(c<0||!this.mask[c])return null;
    const w=this.bary(c,x,y),a=this.alpha(t);let u=0,v=0;
    for(let k=0;k<3;k++) {const q=c*6+k*2;u+=w[k]*(this.A[q]*(1-a)+this.B[q]*a);v+=w[k]*(this.A[q+1]*(1-a)+this.B[q+1]*a);}
    return [u,v];
  }
  native(c,t) {
    const a=this.alpha(t),i=(this.a*this.n+c)*2,j=(this.b*this.n+c)*2;
    const values=this.d.sourceUv||this.d.uv;
    return [values[i]*(1-a)+values[j]*a,values[i+1]*(1-a)+values[j+1]*a];
  }
  trace(cell,x,y,tx,ty) {
    // Walk every crossed edge, even when the endpoint lies in another wet cell.
    let fraction=0;
    for(let walk=0;walk<256;walk++) {
      if(cell<0||!this.mask[cell]) {this.stats.dryStops++;return -1;}
      const end=this.bary(cell,tx,ty);
      if(end.every(v=>v>=-1e-10)) {this.stats.segments++;return cell;}
      const start=this.bary(cell,x+(tx-x)*fraction,y+(ty-y)*fraction);
      let hit=Infinity,side=-1;
      for(let k=0;k<3;k++) if(end[k]<-1e-10) {
        const ratio=Math.max(0,start[k])/(Math.max(0,start[k])-end[k]);
        if(ratio<hit) {hit=ratio;side=k;}
      }
      if(side<0)return -1;
      const next=this.d.neighbors[cell*3+side];
      if(next<0) {this.stats.boundaryStops++;return -1;}
      if(!this.mask[next]) {this.stats.dryStops++;return -1;}
      fraction=Math.min(1,fraction+(1-fraction)*hit+1e-12);
      cell=next;
    }
    this.stats.substepLimits++;return -1;
  }
  integrate(p,seconds,time,continuous,velocityScale=1) {
    if(!Number.isFinite(velocityScale)||velocityScale<=0)throw Error("Invalid particle velocity scale");
    let left=seconds,t=time,steps=0;
    while(left>1e-8) {
      if(++steps>256){this.stats.substepLimits++;return false;}
      const first=this.sample(p.cell,p.x,p.y,continuous?t:time);if(!first)return false;
      const speed=Math.hypot(...first)*velocityScale;
      let dt=Math.min(left, .35*this.d.height[p.cell]/Math.max(speed,1e-6));
      if(continuous&&this.a!==this.b)dt=Math.min(dt,this.m.times[this.b]-t);
      if(dt<=1e-8)return false;
      const mx=p.x+first[0]*velocityScale*dt/2,my=p.y+first[1]*velocityScale*dt/2;
      const midCell=this.trace(p.cell,p.x,p.y,mx,my);if(midCell<0)return false;
      const mid=this.sample(midCell,mx,my,continuous?t+dt/2:time);if(!mid)return false;
      // The midpoint may be faster; shrink the next trial rather than accepting an oversized step.
      const safe=.35*Math.min(this.d.height[p.cell],this.d.height[midCell])/Math.max(Math.hypot(...mid)*velocityScale,1e-6);
      if(safe<dt*.8) {
        dt=safe;
        const hx=p.x+first[0]*velocityScale*dt/2,hy=p.y+first[1]*velocityScale*dt/2,hc=this.trace(p.cell,p.x,p.y,hx,hy);
        if(hc<0)return false;
        const hm=this.sample(hc,hx,hy,continuous?t+dt/2:time);
        mid[0]=hm[0];mid[1]=hm[1];
      }
      const tx=p.x+mid[0]*velocityScale*dt,ty=p.y+mid[1]*velocityScale*dt;
      const next=this.trace(p.cell,p.x,p.y,tx,ty);if(next<0)return false;
      if(p.path)p.path.push([p.x,p.y,tx,ty,seconds-left,seconds-left+dt]);
      p.x=tx;p.y=ty;p.cell=next;left-=dt;t+=dt;
    }
    return true;
  }
  setViewport(bounds) {
    this.viewport=bounds.slice();this.seedCells=[];this.seedCumulative=[];let area=0;
    for(let c=0;c<this.n;c++) if(this.mask[c]) {
      const q=c*4,b=this.cellBox;
      if(b[q+2]<bounds[0]||b[q]>bounds[2]||b[q+3]<bounds[1]||b[q+1]>bounds[3])continue;
      this.seedCells.push(c);area+=this.d.area[c];this.seedCumulative.push(area);
    }
    this.seedArea=area;
  }
  spawn(p={}) {
    if(!this.seedCells.length) {p.cell=-1;return p;}
    for(let retry=0;retry<64;retry++) {
      const target=this.random()*this.seedArea;let lo=0,hi=this.seedCells.length-1;
      while(lo<hi) {const m=(lo+hi)>>1;if(this.seedCumulative[m]<target)lo=m+1;else hi=m;}
      const c=this.seedCells[lo],r=Math.sqrt(this.random()),s=this.random(),w=[1-r,r*(1-s),r*s];let x=0,y=0;
      for(let k=0;k<3;k++){const n=this.d.tri[c*3+k];x+=this.d.xy[n*2]*w[k];y+=this.d.xy[n*2+1]*w[k];}
      if(x<this.viewport[0]||x>this.viewport[2]||y<this.viewport[1]||y>this.viewport[3])continue;
      p.x=x;p.y=y;p.cell=c;p.age=0;this.stats.reseeds++;return p;
    }
    p.cell=-1;return p;
  }
}
window.TracerEngine=TracerEngine;
