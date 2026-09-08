/* Worker body, preceded by the pinned, unmodified gifenc CommonJS bundle. */
"use strict";
const {GIFEncoder,quantize,nearestColorIndex}=self.gifenc;
let encoder,colors,previous,delta,width,height,fps,frames=0,colorLookup;
// gifenc.applyPalette creates a fresh RGB565 cache per call, assigning each bin
// from its first encountered pixel. Particle colors can change that assignment
// for stationary background pixels. Cache exact RGB24 colors across the clip.
function stablePalette(rgba){
  if(!colorLookup)colorLookup=new Uint8Array(1<<24); // bounded 16 MiB, zero = unset
  const indices=new Uint8Array(rgba.length/4);
  for(let i=0,j=0;i<indices.length;i++,j+=4){
    const r=rgba[j],g=rgba[j+1],b=rgba[j+2],key=(r<<16)|(g<<8)|b;
    let index=colorLookup[key];
    if(!index){index=nearestColorIndex(colors,[r,g,b])+1;colorLookup[key]=index;}
    indices[i]=index;
  }
  return indices;
}
self.onmessage=({data:m})=>{
  try {
    if(m.kind==="frame"){
      if(!encoder){width=m.width;height=m.height;fps=m.fps;encoder=GIFEncoder();}
      if(m.width!==width||m.height!==height)throw Error("GIF frame dimensions changed");
      const rgba=new Uint8Array(m.rgba);
      if(!colors)colors=quantize(rgba,255);
      const current=stablePalette(rgba);
      if(!previous){
        encoder.writeFrame(current,width,height,{palette:[[0,0,0],...colors],delay:1000/fps,repeat:0,dispose:1});
        delta=new Uint8Array(current.length);
      }else{
        for(let i=0;i<current.length;i++)delta[i]=current[i]===previous[i]?0:current[i];
        // Changed pixels include the restored background under expired tails.
        encoder.writeFrame(delta,width,height,{delay:1000/fps,transparent:true,transparentIndex:0,dispose:1});
      }
      previous=current;frames++;
      if(encoder.bytesView().length>128*1024*1024)throw Error("GIF exceeds 128 MiB; choose a shorter clip or smaller width");
      self.postMessage({kind:"frame",frames,bytes:encoder.bytesView().length});
    }else if(m.kind==="finish"){
      if(!encoder)throw Error("No GIF frames");encoder.finish();const bytes=encoder.bytesView();
      self.postMessage({kind:"finish",bytes,frames},[bytes.buffer]);
    }else throw Error("Unknown GIF worker request");
  }catch(error){self.postMessage({error:error.message});}
};
