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
#ifndef _FRACTAL_KERNEL_H_CU_
#define _FRACTAL_KERNEL_H_CU_


#include <iostream>
#include "cudaConvUtilities.h.cu"
#include "cudaComplex.h.cu"


//__constant__ unsigned char cmapDevConst[768];
texture<unsigned int, 3, cudaReadModeElementType> tex_uint;
textureReference const * tex_uint_ref;


__device__ inline unsigned int getGrey(int time, int iterMaxEsc)
{
   float val0to1 = log2f((float)(time-1)) / log2f((float)iterMaxEsc); 
   unsigned int grey256 = (unsigned int)(256.0 * val0to1);
   return (grey256 << 16) | (grey256 << 8) | (grey256 << 0);
}

__device__ inline unsigned int getColor(int time, int iterMaxEsc)
{
   
   float val0to1 = log2f((float)(time-1)) / log2f((float)iterMaxEsc); 
   unsigned int grey256 = (unsigned int)(256.0 * val0to1);

   return (tex3D(tex_uint, grey256, 0, 0) <<  0 |
           tex3D(tex_uint, grey256, 1, 0) <<  8 |
           tex3D(tex_uint, grey256, 2, 0) << 16  );
}

////////////////////////////////////////////////////////////////////////////////
template<typename FLOAT_TYPE>
__global__ void GenerateMandlebrotTile_SM10( 
                                     float* devOutPtr,
                                     // TODO:  a lot of params, maybe const mem?
                                     int    nTileRows,
                                     int    nTileCols,
                                     FLOAT_TYPE  tileMinRe,
                                     FLOAT_TYPE  tileMinIm,
                                     FLOAT_TYPE  tileStepRe,
                                     FLOAT_TYPE  tileStepIm,
                                     int    iterMaxEsc)
{
   const int tileRow = blockDim.x*blockIdx.x + threadIdx.x;
   const int tileCol = blockDim.y*blockIdx.y + threadIdx.y;
   const int tileIdx = IDX_1D(tileRow, tileCol, nTileCols);   

   // This thread corresponds to a single point in the complex plane
   FLOAT_TYPE cReal = tileMinRe + tileCol*tileStepRe;
   FLOAT_TYPE cImag = tileMinIm + tileRow*tileStepIm;

   FLOAT_TYPE tempReal = 0;
   FLOAT_TYPE tempImag = 0;

   // First iteration
   FLOAT_TYPE znReal = 0;
   FLOAT_TYPE znImag = 0;

   // The rest of the iterations
   int time;
   for(time=1; time<iterMaxEsc; time++)
   {
      if(znReal*znReal + znImag*znImag < 4.0)
      {
         tempReal = znReal*znReal - znImag*znImag + cReal;
         tempImag = 2*znReal*znImag + cImag;

         znReal = tempReal;
         znImag = tempImag;
      }
      else
         break; // TODO: this will create divergent branches, try ghost code
   }

   // Not sure if I need/should sync threads.  I think coalesced global mem 
   // accesses work better if all the threads are ready to do it at once...
   __syncthreads();

   devOutPtr[tileIdx] = log2f((float)time) / log2f((float)iterMaxEsc);
}



////////////////////////////////////////////////////////////////////////////////
template<typename FLOAT_TYPE>
__global__ void GenerateJuliaTile_z2plusc_SM10( 
                                     unsigned int* devOutPtr,
                                     FLOAT_TYPE cReal,
                                     FLOAT_TYPE cImag,
                                     // TODO:  a lot of params, maybe const mem?
                                     int    nTileRows,
                                     int    nTileCols,
                                     FLOAT_TYPE  tileMinRe,
                                     FLOAT_TYPE  tileMinIm,
                                     FLOAT_TYPE  tileStepRe,
                                     FLOAT_TYPE  tileStepIm,
                                     int    iterMaxEsc,
                                     unsigned char* devColormapData=NULL)
{
   const int tileRow = blockDim.x*blockIdx.x + threadIdx.x;
   const int tileCol = blockDim.y*blockIdx.y + threadIdx.y;
   const int tileIdx = IDX_1D(tileRow, tileCol, nTileCols);   

   // This thread corresponds to a single point in the complex plane
   FLOAT_TYPE znReal = tileMinRe + tileCol*tileStepRe;
   FLOAT_TYPE znImag = tileMinIm + tileRow*tileStepIm;

   FLOAT_TYPE tempReal = 0;
   FLOAT_TYPE tempImag = 0;

   // The rest of the iterations
   int time;
   for(time=1; time<iterMaxEsc; time++)
   {
      if(znReal*znReal + znImag*znImag < 4.0)
      {
         tempReal = znReal*znReal - znImag*znImag + cReal;
         tempImag = 2*znReal*znImag + cImag;

         znReal = tempReal;
         znImag = tempImag;
      }
      else
         break; // TODO: this will create divergent branches, try ghost code
   }

   // Not sure if I need/should sync threads.  I think coalesced global mem 
   // accesses work better if all the threads are ready to do it at once...
   __syncthreads();

   devOutPtr[tileIdx] = getColor(time, iterMaxEsc);
}
                                    


template<typename FLOAT_TYPE>
__global__ void GenerateJuliaTile_cexpz_SM20( 
                                     unsigned int* devOutPtr,
                                     FLOAT_TYPE cReal,
                                     FLOAT_TYPE cImag,
                                     // TODO:  a lot of params, maybe const mem?
                                     int    nTileRows,
                                     int    nTileCols,
                                     FLOAT_TYPE  tileMinRe,
                                     FLOAT_TYPE  tileMinIm,
                                     FLOAT_TYPE  tileStepRe,
                                     FLOAT_TYPE  tileStepIm,
                                     int    iterMaxEsc)
{
   const int tileRow = blockDim.x*blockIdx.x + threadIdx.x;
   const int tileCol = blockDim.y*blockIdx.y + threadIdx.y;
   const int tileIdx = IDX_1D(tileRow, tileCol, nTileCols);   

   // This thread corresponds to a single point in the complex plane
   //FLOAT_TYPE znReal = tileMinRe + tileCol*tileStepRe;
   //FLOAT_TYPE znImag = tileMinIm + tileRow*tileStepIm;
   cudaComplex<FLOAT_TYPE> zn( tileMinRe + tileCol*tileStepRe,
                               tileMinIm + tileRow*tileStepIm);

   //FLOAT_TYPE tempReal = 0;
   //FLOAT_TYPE tempImag = 0;
   cudaComplex<FLOAT_TYPE> temp(0,0);

   cudaComplex<FLOAT_TYPE> c(cReal, cImag);

   // The rest of the iterations
   int time;
   for(time=1; time<iterMaxEsc; time++)
   {
      if(zn.conj_sq() < 1000.0)
      {
         temp = c * zn.zexp();
         zn = temp;
      }
      else
         break; // TODO: this will create divergent branches, try ghost code
   }

   // Not sure if I need/should sync threads.  I think coalesced global mem 
   // accesses work better if all the threads are ready to do it at once...
   __syncthreads();

   devOutPtr[tileIdx] = getColor(time, iterMaxEsc);
}


template<typename FLOAT_TYPE>
__global__ void GenerateJuliaTile_other_SM20( 
                                     unsigned int* devOutPtr,
                                     FLOAT_TYPE cReal,
                                     FLOAT_TYPE cImag,
                                     // TODO:  a lot of params, maybe const mem?
                                     int    nTileRows,
                                     int    nTileCols,
                                     FLOAT_TYPE  tileMinRe,
                                     FLOAT_TYPE  tileMinIm,
                                     FLOAT_TYPE  tileStepRe,
                                     FLOAT_TYPE  tileStepIm,
                                     int    iterMaxEsc)
{
   const int tileRow = blockDim.x*blockIdx.x + threadIdx.x;
   const int tileCol = blockDim.y*blockIdx.y + threadIdx.y;
   const int tileIdx = IDX_1D(tileRow, tileCol, nTileCols);   

   // This thread corresponds to a single point in the complex plane
   //FLOAT_TYPE znReal = tileMinRe + tileCol*tileStepRe;
   //FLOAT_TYPE znImag = tileMinIm + tileRow*tileStepIm;
   cudaComplex<FLOAT_TYPE> zn( tileMinRe + tileCol*tileStepRe,
                               tileMinIm + tileRow*tileStepIm);

   //FLOAT_TYPE tempReal = 0;
   //FLOAT_TYPE tempImag = 0;
   cudaComplex<FLOAT_TYPE> temp(0,0);

   cudaComplex<FLOAT_TYPE> c(cReal, cImag);

   // The rest of the iterations
   int time;
   for(time=1; time<iterMaxEsc; time++)
   {
      if(zn.conj_sq() < 1000.0)
      {
         temp = (3*zn*zn*zn - 2*c*zn + 3) / (-4*zn*zn + 1);
         zn = temp;
      }
      else
         break; // TODO: this will create divergent branches, try ghost code
   }

   // Not sure if I need/should sync threads.  I think coalesced global mem 
   // accesses work better if all the threads are ready to do it at once...
   __syncthreads();

   devOutPtr[tileIdx] = getColor(time, iterMaxEsc);
}

#endif
