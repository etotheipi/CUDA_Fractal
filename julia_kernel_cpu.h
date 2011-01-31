#ifndef _JULIA_KERNEL_CPU_H_
#define _JULIA_KERNEL_CPU_H_

#include <cmath>

template<typename VALUE>
void GenerateJuliaTile_z2plusc_CPU( float* outputPtr,
                                    VALUE  cReal,
                                    VALUE  cImag,
                                    int    nTileRows,
                                    int    nTileCols,
                                    VALUE  tileMinRe,
                                    VALUE  tileMinIm,
                                    VALUE  tileStepRe,
                                    VALUE  tileStepIm,
                                    int    iterMaxEsc)
{
   VALUE znReal, znImag, tempReal, tempImag;
   for(int r=0; r<nTileRows; r++)
   {
      for(int c=0; c<nTileCols; c++)
      {
         znReal = tileMinRe + r*tileStepRe;
         znImag = tileMinIm + c*tileStepIm;

         int time;
         for(time=1; time<iterMaxEsc; time++)
         {
            if(znReal*znReal + znImag*znImag > 4.0)
               break;

            tempReal = znReal*znReal - znImag*znImag + cReal;
            tempImag = 2*znReal*znImag + cImag;
            
            znReal = tempReal;
            znImag = tempImag;
         }

         // Gotta flip the rows
         int index = (nTileCols-c-1) * nTileRows + r;
         outputPtr[index] = logf((float)time) / logf((float)iterMaxEsc);
      }

   }

}

#endif
