////////////////////////////////////////////////////////////////////////////////
// 
// Alan Reiner
// 05 Jan, 2010
// 
//
// Some very simple code for generating fractals using CUDA
// Some code is intentionally left inefficient because simplicity was more
// important.  This might be a good playground for testing various optimization
// techniques
//
////////////////////////////////////////////////////////////////////////////////
#include "fractal_kernel.h.cu"


__device__ inline void IFS(VALUE & znplus1Re,
                           VALUE & znplus1Im,
                           VALUE const & znRe,
                           VALUE const & znIm,
                           VALUE const & cRe,
                           VALUE const & cIm)
                                    
{
   VALUE re = znRe*znRe - znIm*znIm + cRe; 
   VALUE im = 2*znRe*znIm + cIm;
   znplus1Re = re;
   znplus1Im = im;
}


// Separating out into a separate function really isn't necessary...
__device__ inline int MandlebrotIterator( VALUE const & cRe, 
                                          VALUE const & cIm,
                                          int   const & iterMaxEsc)
                                     
{
   // First iteration
   VALUE znRe = cRe;
   VALUE znIm = cIm;
   
   int t;
   for(t=1; t<iterMaxEsc; t++)
   {
      if(znRe*znRe + znIm*znIm < 4.0)
         IFS(znRe, znIm, znRe, znIm, cRe, cIm);
      else
         break; // TODO: this will create divergent branches, try ghost code
   }

   return t;
}


__global__ void GenerateFractalTile( VALUE *devOutPtr,
                                     // TODO:  a lot of params, maybe const mem?
                                     int    nTileRows,
                                     int    nTileCols,
                                     VALUE  tileMinRe,
                                     VALUE  tileMinIm,
                                     VALUE  tileStepRe,
                                     VALUE  tileStepIm,
                                     int    iterMaxEsc)
{
   const int tileRow = blockDim.x*blockIdx.x + threadIdx.x;
   const int tileCol = blockDim.y*blockIdx.y + threadIdx.y;
   const int tileIdx = IDX_1D(tileRow, tileCol, nTileCols);   


   // This thread corresponds to a single point in the complex plane
   VALUE cRe = tileMinRe + tileCol*tileStepRe;
   VALUE cIm = tileMinIm + tileRow*tileStepIm;

   // First iteration
   VALUE znRe = cRe;
   VALUE znIm = cIm;
   VALUE tempRe,tempIm;

   int time;
   for(time=1; time<iterMaxEsc; time++)
   {
      if(znRe*znRe + znIm*znIm < 4.0)
      {
         tempRe = znRe*znRe - znIm*znIm + cRe; 
         tempIm = 2*znRe*znIm + cIm;
         znRe = tempRe;
         znIm = tempIm;
      }
      else
         break; // TODO: this will create divergent branches, try ghost code
   }

   // Not sure if I need/should sync threads.  I think coalesced global mem 
   // accesses work better if all the threads are ready to do it at once...
   __syncthreads();

   devOutPtr[tileIdx] = __logf(time);

}
                                    


