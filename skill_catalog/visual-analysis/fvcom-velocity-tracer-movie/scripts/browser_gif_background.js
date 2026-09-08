/* Returns GIF/mask Blob URLs and report for independent pixel verification.
 * Download URLs before revoking them. Run in Snapshot with shading enabled.
 */
(async()=>{
  const f=fvcomTracer,original=SnapshotGifSession.prototype.frame;
  let first,mask,frames=0;
  SnapshotGifSession.prototype.frame=function(){
    const rgba=original.call(this),pixels=new Uint32Array(rgba.buffer);
    if(!first){first=pixels.slice();mask=new Uint8Array(pixels.length).fill(1);
      for(let i=0;i<pixels.length;i++){
        const j=i*4,r=rgba[j],g=rgba[j+1],b=rgba[j+2];
        if(b>r&&g>r&&r<180&&!(r===12&&g===23&&b===33)&&i/this.width>160&&i/this.width<this.height-160)mask[i]=3;
      }
    }else for(let i=0;i<pixels.length;i++)if(pixels[i]!==first[i])mask[i]=0;
    frames++;return rgba;
  };
  try{
    const {blob,report}=await f.exportGif({width:1200,duration:3,fps:10});
    if(frames!==report.frames)throw Error("Source frame count mismatch");
    return {gif_url:URL.createObjectURL(blob),mask_url:URL.createObjectURL(new Blob([mask])),report};
  }finally{SnapshotGifSession.prototype.frame=original;}
})()
