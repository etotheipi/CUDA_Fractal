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


__global__ void GenerateMandlebrotTile( 
                                     VALUE *devOutPtr,
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

   // The rest of the iterations
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
}
                                    
__global__ void GenerateJuliaTile( 
                                     VALUE *devOutPtr,
                                     VALUE cRe,
                                     VALUE cIm,
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
   cudaComplex<VALUE> zn( tileMinRe + tileCol*tileStepRe,
                         tileMinIm + tileRow*tileStepIm);
   cudaComplex<VALUE> temp(0,0);

   // First iteration
   cudaComplex<VALUE> c(cRe, cIm);

   // The rest of the iterations
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
}


// This is a completely failed attempt at producing cool 3D mandlebrot
// This should probably be ignored
__global__ void GenerateFractalTile3D( VALUE *devOutPtr,
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
   //const int tilePixels = blockDim.x * blockDim.y;

   const int localIdx = IDX_1D(threadIdx.x, threadIdx.y, blockDim.y);

   __shared__ char shmMem[8192];
   float* shmAccum = (float*)shmMem;

   shmAccum[localIdx] = 0.0f;

   VALUE j = 0.5;
   VALUE jMin   = 0.0;
   VALUE jMax   = 1.0;
   int   jRes   = 16;
   VALUE jWidth = jMax - jMin;
   VALUE jStep = jWidth / (VALUE)jRes;

   for( j=jMin; j<jMax; j+=jStep)
   {
      // This thread corresponds to a single point in the complex plane
      cudaQuaternion<VALUE> c( tileMinRe + tileCol*tileStepRe,
                               tileMinIm + tileRow*tileStepIm, 
                               j,
                               0);
   
      cudaQuaternion<VALUE> temp(0,0);
   
      // First iteration
      cudaQuaternion<VALUE> zn = c;
   
   
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
   
      VALUE jScale = (jWidth - (j-jMin)) / jWidth;
      shmAccum[localIdx] += jScale*jScale * (float)time;
   }
   
   __syncthreads();
   // Not sure if I need/should sync threads.  I think coalesced global mem 
   // accesses work better if all the threads are ready to do it at once...
   devOutPtr[tileIdx] = log2f((float)(shmAccum[localIdx]));

   // Not sure if I need/should sync threads.  I think coalesced global mem 
   // accesses work better if all the threads are ready to do it at once...

   /*
   const int tileRow = blockDim.x*blockIdx.x + threadIdx.x;
   const int tileCol = blockDim.y*blockIdx.y + threadIdx.y;
   const int tileIdx = IDX_1D(tileRow, tileCol, nTileCols);   
   const int tilePixels = blockDim.x * blockDim.y;

   __shared__ char shmMem[8192];
   float* shmAccum = (float*)shmMem;

   VALUE jMin   = 0.0;
   VALUE jMax   = 1.0;
   int   jRes   = 2048;
   VALUE jWidth = jMax - jMin;
   VALUE jStep = jWidth / (VALUE)jRes;

   shmAccum[tileIdx] = 0.0;
   
   //for(VALUE j=jMin; j<=jMin+0.0001; j+=jStep)
   //{
   VALUE j = 0.0;

      // This thread corresponds to a single point in the complex plane
      cudaQuaternion<VALUE> c( tileMinRe + tileCol*tileStepRe,
                               tileMinIm + tileRow*tileStepIm,
                               j,
                               0);
   
      cudaQuaternion<VALUE> temp;
   
      // First iteration
      cudaQuaternion<VALUE> zn = c;
   
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

      //VALUE jScale = (jWidth - (j-jMin)) / jWidth;
      //shmAccum[tileIdx] += jScale*jScale * (float)time;
   //}

   // Not sure if I need/should sync threads.  I think coalesced global mem 
   // accesses work better if all the threads are ready to do it at once...
   __syncthreads();

   //devOutPtr[tileIdx] = log2f((float)(shmAccum[tileIdx]));
   devOutPtr[tileIdx] = log2f((float)time);
   */

}





