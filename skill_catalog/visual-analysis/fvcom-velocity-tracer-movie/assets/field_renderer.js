/* Shared native-triangle shading, boundaries and snapshot integration. */
"use strict";
function createFieldRenderer(shade,engine,data,meta,forceCanvas=false){
  let shader=null,width=0,height=0,scale=1,ox=0,oy=0,time=meta.times[0],enabled=true;
  function compile(gl,type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
  function initGL(){
    if(forceCanvas)return null;
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
  function drawShade(){if(!shader)return;const {gl,uniforms:u}=shader;gl.viewport(0,0,shade.width,shade.height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);if(!enabled)return;
    gl.uniform2f(u.size,width,height);gl.uniform3f(u.view,scale,ox,oy);gl.uniform1f(u.alpha,engine.alpha(time));gl.uniform1f(u.vmax,meta.vmax);gl.drawArrays(gl.TRIANGLES,0,meta.elements*3);
  }

  try {shader=initGL();if(shader)upload();}catch(error){if(shader)shader.gl.getExtension("WEBGL_lose_context")?.loseContext();throw error;}
  return {get backend(){return shader?.renderer||"Canvas fallback";},get available(){return !!shader;},upload,
    draw(v,t,on=true){({width,height,scale,ox,oy}=v);time=t;enabled=on;drawShade();},
    dispose(){if(!shader)return;const {gl,program,buffers}=shader;for(const b of Object.values(buffers))gl.deleteBuffer(b);gl.deleteProgram(program);gl.getExtension("WEBGL_lose_context")?.loseContext();shader=null;}
  };
}
function drawMeshBoundary(ctx,data,v){
  const project=(x,y)=>[x*v.scale+v.ox,v.oy-y*v.scale];ctx.lineWidth=.8;
  for(let type=0;type<2;type++){ctx.beginPath();ctx.strokeStyle=type?"#a5c1cc66":"#a9c3cf99";ctx.setLineDash(type?[4,5]:[]);
    for(let i=0;i<data.boundaryType.length;i++)if(data.boundaryType[i]===type){const a=data.boundary[i*2],b=data.boundary[i*2+1];ctx.moveTo(...project(data.xy[a*2],data.xy[a*2+1]));ctx.lineTo(...project(data.xy[b*2],data.xy[b*2+1]));}ctx.stroke();}
  ctx.setLineDash([]);
}
function advanceSnapshot(engine,particles,trails,settings,activeTime,dt,emit=true){
  const {scale,visualSpeed,time,tail,width,height,ox,oy,vmax}=settings;
  const seconds=dt*12/Math.max(scale*vmax*.125,1e-9)*visualSpeed,tickEnd=activeTime+dt;
  trails.history.expire(activeTime-trails.epoch,tail);
  for(const p of particles){
    if(p.cell<0||p.age>8||!engine.mask[p.cell])engine.spawn(p);
    if(p.cell>=0){p.path=[];
      if(!engine.integrate(p,seconds,time,false)){engine.spawn(p);}
      else {const velocity=engine.sample(p.cell,p.x,p.y,time),bucket=Math.min(3,Math.floor(Math.sqrt(Math.hypot(...velocity)/vmax)*4));
        if(emit)for(const segment of p.path)trails.append(segment,bucket,activeTime+segment[4]/seconds*dt,activeTime+segment[5]/seconds*dt,tickEnd,tail);
        const x=p.x*scale+ox,y=oy-p.y*scale;if(x<0||x>width||y<0||y>height)p.age=9;
      }
    }
    p.age+=dt;
  }
  return tickEnd;
}
window.createFieldRenderer=createFieldRenderer;window.drawMeshBoundary=drawMeshBoundary;window.advanceSnapshot=advanceSnapshot;
