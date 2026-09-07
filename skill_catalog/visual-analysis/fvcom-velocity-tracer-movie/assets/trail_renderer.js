/* Finite-age particle segments. OMA implementation; no accumulated bitmap fade.
 * One 32-byte record: endpoints, start/end birth time, palette bucket, tick end.
 * tick end orders expiry even when particle substeps within a tick interleave.
 */
"use strict";
class TrailHistory {
  constructor(maxBytes=64*1024*1024) {
    this.limit=Math.floor(maxBytes/32);this.capacity=Math.min(16384,this.limit);
    this.records=new Float32Array(this.capacity*8);this.head=0;this.count=0;
    this.dirty=[];this.version=0;this.dropped=0;this.limitedUntil=-Infinity;
  }
  clear(){this.head=0;this.count=0;this.dirty=[];this.dropped=0;this.limitedUntil=-Infinity;}
  grow(){
    const next=Math.min(this.limit,this.capacity*2);if(next===this.capacity)return;
    const a=new Float32Array(next*8);
    for(let i=0;i<this.count;i++)a.set(this.records.subarray(((this.head+i)%this.capacity)*8,((this.head+i)%this.capacity)*8+8),i*8);
    this.records=a;this.capacity=next;this.head=0;this.version++;this.dirty=[[0,this.count]];
  }
  push(values,tail){
    if(this.count===this.capacity)this.grow();
    if(this.count===this.capacity){this.limitedUntil=Math.max(this.limitedUntil,this.records[this.head*8+7]+tail);this.head=(this.head+1)%this.capacity;this.count--;this.dropped++;}
    const slot=(this.head+this.count)%this.capacity;this.records.set(values,slot*8);this.count++;
    const last=this.dirty.at(-1);if(last&&last[1]===slot)last[1]++;else this.dirty.push([slot,slot+1]);
  }
  expire(now,tail){while(this.count&&this.records[this.head*8+7]<=now-tail+1e-7){this.head=(this.head+1)%this.capacity;this.count--;}}
  ranges(){const first=Math.min(this.count,this.capacity-this.head);return first?[[this.head,first],...(first<this.count?[[0,this.count-first]]:[])]:[];}
  metrics(now,tail){return {retainedSegments:this.count,historyBytes:this.records.byteLength,historyLimitBytes:this.limit*32,droppedSegments:this.dropped,
    historyLimited:now<this.limitedUntil,effectiveHistorySeconds:this.count?Math.min(tail,Math.max(0,now-this.records[this.head*8+4])):0};}
}
class TrailRenderer {
  constructor(canvas,{forceCanvas=false,maxBytes}={}) {
    this.canvas=canvas;this.history=new TrailHistory(maxBytes);this.epoch=0;this.gpuVersion=-1;this.lastDraw=null;
    this.palette=[[164,218,225,.55],[203,239,238,.7],[229,248,244,.85],[251,255,248,.95]];
    this.backend="Canvas2D";this.reason="";
    if(!forceCanvas)try{this.initGL();}catch(error){this.reason=String(error.message);this.toCanvas();}
    else this.toCanvas();
    if(this.gl)this.canvas.addEventListener("webglcontextlost",e=>{e.preventDefault();this.reason="Trail graphics context lost";this.toCanvas();if(this.lastDraw)this.draw(...this.lastDraw);});
  }
  toCanvas(){
    if(this.gl||!this.canvas.getContext("2d")){const replacement=this.canvas.cloneNode(false);this.canvas.replaceWith(replacement);this.canvas=replacement;}
    this.gl=null;this.ctx=this.canvas.getContext("2d");this.backend="Canvas2D";
    if(this.view)this.resize(this.view.width,this.view.height,this.view.dpr);
  }
  initGL(){
    const gl=this.canvas.getContext("webgl2",{alpha:true,antialias:false,premultipliedAlpha:true,preserveDrawingBuffer:true});
    if(!gl)throw Error("WebGL2 unavailable");this.gl=gl;
    const vs=`#version 300 es
      precision highp float;
      in vec4 endpoints;in vec2 born;in float bucket;
      uniform vec2 size;uniform vec3 view;uniform float radius;
      out vec2 local;flat out float lengthPx;flat out vec2 birth;flat out int colorIndex;
      void main(){
        vec2 a=vec2(endpoints.x*view.x+view.y,view.z-endpoints.y*view.x);
        vec2 b=vec2(endpoints.z*view.x+view.y,view.z-endpoints.w*view.x);
        vec2 delta=b-a;float len=length(delta);vec2 along=len>1e-7?delta/len:vec2(1,0);
        float x=(gl_VertexID==1||gl_VertexID==3)?len+radius:-radius;
        float y=gl_VertexID>=2?radius:-radius;vec2 p=a+x*along+y*vec2(-along.y,along.x);
        gl_Position=vec4(p.x/size.x*2.-1.,1.-p.y/size.y*2.,0.,1.);
        local=vec2(x,y);lengthPx=len;birth=born;colorIndex=int(bucket+.5);
      }`;
    const fs=`#version 300 es
      precision highp float;
      in vec2 local;flat in float lengthPx;flat in vec2 birth;flat in int colorIndex;
      uniform float now;uniform float duration;uniform float aa;out vec4 pixel;
      void main(){
        float fraction=lengthPx>1e-7?clamp(local.x/lengthPx,0.,1.):1.;
        float age=max(0.,now-mix(birth.x,birth.y,fraction));if(age>=duration)discard;
        float distance=length(vec2(local.x-clamp(local.x,0.,lengthPx),local.y));
        float coverage=1.-smoothstep(.525-aa,.525+aa,distance);
        float weight=max(0.,(exp(-3.*age/duration)-exp(-3.))/(1.-exp(-3.)));
        vec4 color=colorIndex==0?vec4(164.,218.,225.,.55):colorIndex==1?vec4(203.,239.,238.,.7):colorIndex==2?vec4(229.,248.,244.,.85):vec4(251.,255.,248.,.95);
        float alpha=color.a*coverage*weight;pixel=vec4(color.rgb/255.*alpha,alpha);
      }`;
    const compile=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
    const p=gl.createProgram();gl.attachShader(p,compile(gl.VERTEX_SHADER,vs));gl.attachShader(p,compile(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);
    if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));this.program=p;gl.useProgram(p);
    this.buffer=gl.createBuffer();this.locations={};for(const name of ["endpoints","born","bucket"])this.locations[name]=gl.getAttribLocation(p,name);
    this.uniforms={};for(const name of ["size","view","radius","now","duration","aa"])this.uniforms[name]=gl.getUniformLocation(p,name);
    gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);this.backend="WebGL2 instanced";
  }
  resize(width,height,dpr){this.view={width,height,dpr};this.canvas.width=Math.round(width*dpr);this.canvas.height=Math.round(height*dpr);if(this.ctx)this.ctx.setTransform(dpr,0,0,dpr,0,0);}
  clear(now=0){this.history.clear();this.epoch=now;if(this.gl){this.gl.clearColor(0,0,0,0);this.gl.clear(this.gl.COLOR_BUFFER_BIT);}else if(this.ctx){this.ctx.clearRect(0,0,this.canvas.width,this.canvas.height);}}
  append(segment,bucket,start,end,tickEnd,tail){if(Math.hypot(segment[2]-segment[0],segment[3]-segment[1])<1e-9)return;
    this.history.push([segment[0],segment[1],segment[2],segment[3],start-this.epoch,end-this.epoch,bucket,tickEnd-this.epoch],tail);
  }
  draw(now,tail,view){
    this.lastDraw=[now,tail,view];const clock=now-this.epoch,h=this.history;h.expire(clock,tail);
    if(this.gl)this.drawGL(clock,tail,view);else this.drawCanvas(clock,tail,view);
    h.dirty=[];
  }
  drawGL(now,tail,v){
    const gl=this.gl,h=this.history,u=this.uniforms;gl.viewport(0,0,this.canvas.width,this.canvas.height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.program);gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);
    if(this.gpuVersion!==h.version){gl.bufferData(gl.ARRAY_BUFFER,h.records,gl.DYNAMIC_DRAW);this.gpuVersion=h.version;}
    else for(const [a,b] of h.dirty)if(b>a)gl.bufferSubData(gl.ARRAY_BUFFER,a*32,h.records.subarray(a*8,b*8));
    gl.uniform2f(u.size,v.width,v.height);gl.uniform3f(u.view,v.scale,v.ox,v.oy);gl.uniform1f(u.radius,.525+1/v.dpr);gl.uniform1f(u.aa,.5/v.dpr);
    gl.uniform1f(u.now,now);gl.uniform1f(u.duration,tail);
    for(const [start,count] of h.ranges()){
      for(const [name,size,offset] of [["endpoints",4,0],["born",2,16],["bucket",1,24]]){const loc=this.locations[name];gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,gl.FLOAT,false,32,start*32+offset);gl.vertexAttribDivisor(loc,1);}
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP,0,4,count);
    }
  }
  drawCanvas(now,tail,v){
    const ctx=this.ctx,h=this.history,a=h.records;ctx.clearRect(0,0,v.width,v.height);ctx.lineWidth=1.05;ctx.lineCap="round";
    // Eight age buckets bound Canvas calls. Expiry remains exact; clips trim the
    // portion of a segment older than T rather than preserving a rounded edge.
    const paths=Array.from({length:32},()=>new Path2D()),count=new Uint32Array(32);
    for(const [start,n] of h.ranges())for(let i=start;i<start+n;i++){
      const q=i*8,b0=a[q+4],b1=a[q+5];if(now-b1>=tail)continue;
      const f=b1>b0?Math.max(0,Math.min(1,(now-tail-b0)/(b1-b0))):0;
      const birth0=b0+(b1-b0)*f,age=Math.max(0,now-(birth0+b1)/2),bin=Math.min(7,Math.floor(age/tail*8)),key=bin*4+a[q+6];
      const x0=a[q]+(a[q+2]-a[q])*f,y0=a[q+1]+(a[q+3]-a[q+1])*f;
      paths[key].moveTo(x0*v.scale+v.ox,v.oy-y0*v.scale);paths[key].lineTo(a[q+2]*v.scale+v.ox,v.oy-a[q+3]*v.scale);count[key]++;
    }
    for(let i=31;i>=0;i--)if(count[i]){const age=(Math.floor(i/4)+.5)/8,w=(Math.exp(-3*age)-Math.exp(-3))/(1-Math.exp(-3)),c=this.palette[i%4];ctx.strokeStyle=`rgba(${c[0]},${c[1]},${c[2]},${c[3]*w})`;ctx.stroke(paths[i]);}
  }
  metrics(now,tail){return {trailBackend:this.backend,trailFallbackReason:this.reason,...this.history.metrics(now-this.epoch,tail)};}
  alphaPixels(){const n=this.canvas.width*this.canvas.height,a=new Uint8Array(n*4);
    if(this.gl)this.gl.readPixels(0,0,this.canvas.width,this.canvas.height,this.gl.RGBA,this.gl.UNSIGNED_BYTE,a);else a.set(this.ctx.getImageData(0,0,this.canvas.width,this.canvas.height).data);
    let nonzero=0,max=0,sum=0;for(let i=3;i<a.length;i+=4){if(a[i])nonzero++;max=Math.max(max,a[i]);sum+=a[i];}return {nonzero,max,sum};
  }
}
window.TrailHistory=TrailHistory;window.TrailRenderer=TrailRenderer;
