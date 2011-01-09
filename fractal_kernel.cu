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
#include "cudaComplex.h.cu"


//__device__ void IFS_Mandlebrot(Complex 


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
   cudaComplex<VALUE> c( tileMinRe + tileCol*tileStepRe,
                     tileMinIm + tileRow*tileStepIm);

   cudaComplex<VALUE> temp(0,0);

   // First iteration
   cudaComplex<VALUE> zn = c;


   int time;
   for(time=1; time<iterMaxEsc; time++)
   {
      if(zn.conj_sq() < 4.0)
      {
         temp = zn*zn + c;
         zn = temp;
      }
      else
         break; // TODO: this will create divergent branches, try ghost code
   }

   // Not sure if I need/should sync threads.  I think coalesced global mem 
   // accesses work better if all the threads are ready to do it at once...
   __syncthreads();

   devOutPtr[tileIdx] = log2f((float)time);



   // Original code without complex numbers
   /*
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

   devOutPtr[tileIdx] = log2f((float)time);
   */

}
                                    


