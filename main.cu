/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>
#include <math.h>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <cutil_inline.h>
#include <stopwatch.h>
#include <cmath>
#include <assert.h>


#include "cudaImageHost.h"
#include "cudaImageDevice.h.cu"
#include "ComplexNumber.h"
#include "cudaComplex.h.cu"
#include "fractal_kernel.h.cu"
#include "writePNG.h"

using namespace std;

unsigned int timer;

#define SIZE_TILE 4096

int  runDevicePropertiesQuery(void);
void writePngFile(cudaImageHost<VALUE> img);

////////////////////////////////////////////////////////////////////////////////
//
// Program main
//
// TODO:  Remove the CUTIL calls so libcutil is not required to compile/run
//
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
   VALUE cReal, cImag;
   
   
   if(argc < 3)
   {
      cReal = -0.8;
      cImag = -0.156;
      cout << "USAGE:  " << argv[0] << " [cReal cImag]" << endl;
      cout << "         using    " << cReal << " and " << cImag << endl;
      
   }
   else
   {
      cReal = strtod(argv[1], NULL);
      cImag = strtod(argv[2], NULL);
   }

   cout << "\n********************************************************************************" << endl;
   cout << "***Starting CUDA-accelerated fractal generation" << endl;
   cout << "***NOTE:  This uses CUDA OOP which relies on CUDA Capability 2.0+" << endl << endl;

   // Always do this first, as a sanity check:
   runDevicePropertiesQuery();

   int resRe = SIZE_TILE*2;
   int resIm = resRe;

   VALUE minRe = -2;
   VALUE minIm = -2;
   VALUE maxRe =  2;
   VALUE maxIm =  2;


   int nTilesRe = (resRe-1)/SIZE_TILE + 1;
   int nTilesIm = (resIm-1)/SIZE_TILE + 1;
   int nTiles   = nTilesRe * nTilesIm;

   VALUE pixelStepRe = (maxRe - minRe) / (VALUE)resRe;
   VALUE pixelStepIm = (maxIm - minIm) / (VALUE)resIm;

   VALUE tileStepRe = (maxRe - minRe) / nTilesRe;
   VALUE tileStepIm = (maxIm - minIm) / nTilesIm;

   const int tileSizeX = (resRe < SIZE_TILE ? resRe : SIZE_TILE);
   const int tileSizeY = (resIm < SIZE_TILE ? resIm : SIZE_TILE);

   // TODO:  Need to figure out how to handle remarkably large images
   //        Probably need to not only process separately, but also
   //        write out separate files for each tile
   //        For now, only going to do one tile, so I don't have to
   //        worry about it
   cudaImageHost<VALUE>    wholeFractal(resRe, resIm);
   cudaImageHost<VALUE>    hostTile(tileSizeX, tileSizeY);
   cudaImageDevice<VALUE>  devTile(tileSizeX, tileSizeY);

   int bx = 16;
   int by = 16;
   int gx = tileSizeX / bx;
   int gy = tileSizeY / by;
   dim3 BLOCK(bx, by, 1);
   dim3 GRID( gx, gy, 1);


   cout << "Getting ready to generate the fractal, in tiles:" << endl;

   cout << "\tTotal fractal size:  (" << resRe << " x " << resIm << ") pixels\n"
        << "\tEach tile is size:   (" << SIZE_TILE << " x " << SIZE_TILE << ") pixels\n"
        << "\tTotal num tiles:     (" << nTilesRe << " x " << nTilesIm << ") = " << nTiles << " tile(s)\n"
        << "\tCUDA block size...   (" << bx << "," << by << ",1) threads\n"
        << "\t... in a grid....    (" << gx << "," << gy << ",1) blocks\n\n";
   
   cout << "Starting actual fractal calculation..." << endl;
   cout << "\t Creating julia set with c="<<cReal<<" + " <<cImag<<"i"<< endl;


   // Create a colormap for the PNG file to saved with
   Colormap cmap("cmap_green_red.txt");

   // Now actually start the rendering
   float accumFractalTime = 0.0f;
   float accumFileWriteTime = 0.0f;
   char* fn = new char[256];
   for(int tileRe=0; tileRe<nTilesRe; tileRe++)
   {
      for(int tileIm=0; tileIm<nTilesIm; tileIm++)
      {
         VALUE tileMinRe = minRe + tileRe*tileStepRe;
         VALUE tileMinIm = minIm + tileIm*tileStepIm;

         cout << "Generating tile (" << tileRe << "," << tileIm << ")..." << endl;
         gpuStartTimer();
         GenerateJuliaTile<<<GRID, BLOCK>>>(   devTile,
                                               // -0.8, -0.156 is my favorite, so far
                                               cReal,
                                               cImag,
                                               SIZE_TILE,
                                               SIZE_TILE,
                                               tileMinRe,
                                               tileMinIm,
                                               pixelStepRe,
                                               pixelStepIm,
                                               1024) ;

         devTile.copyToHost(hostTile);
         accumFractalTime += gpuStopTimer();

         // Copy this tile into the master fractal 
         int startRow = tileIm*SIZE_TILE;
         int startCol = tileRe*SIZE_TILE;
         for(int r=0; r<SIZE_TILE; r++)
            for(int c=0; c<SIZE_TILE; c++)
               wholeFractal(startRow+r,startCol+c) = hostTile(r,c);


         
      }
   }

   // Write the host tile to png file
   cpuStartTimer();
   sprintf(fn, "fractal_%dx%d.png", resRe, resIm);
   cout << "writing to file " << string(fn) << "..." << endl;
   writePngFile( wholeFractal.getDataPtr(),
                 wholeFractal.numRows(), 
                 wholeFractal.numCols(), 
                 fn,
                 &cmap);
   accumFileWriteTime += cpuStopTimer();

   //wholeFractal.writeFile("fractal.txt");

   cout << endl << endl;
   cout << "\tTime to render entire fractal: "
        << accumFractalTime/1000. << "s" << endl;

   cout << "\tTime to write image to file:   " 
        << accumFileWriteTime/1000. << "s" << endl;

   cudaThreadExit();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Query the devices on the system and select the fastest (or override the
// selectedDevice variable to choose your own
int runDevicePropertiesQuery(void)
{
   cout << endl;
   cout << "****************************************";
   cout << "***************************************" << endl;
   cout << "***Device query and selection:" << endl;
   int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
   {
		cout << "cudaGetDeviceCount() FAILED." << endl;
      cout << "CUDA Driver and Runtime version may be mismatched.\n";
      return -1;
	}

   // Check to make sure we have at least on CUDA-capable device
   if( deviceCount == 0)
   {
      cout << "No CUDA devices available." << endl;
      return -1;
	}

   // Fastest device automatically selected.  Can override below
   int selectedDevice = cutGetMaxGflopsDeviceId() ;
   //selectedDevice = 0;
   cudaSetDevice(selectedDevice);

   cudaDeviceProp gpuProp;
   cout << "CUDA-enabled devices on this system:  " << deviceCount <<  endl;
   for(int dev=0; dev<deviceCount; dev++)
   {
      cudaGetDeviceProperties(&gpuProp, dev); 
      char* devName = gpuProp.name;
      int mjr = gpuProp.major;
      int mnr = gpuProp.minor;
      int memMB = gpuProp.totalGlobalMem / (1024*1024) + 1;
      if( dev==selectedDevice )
         cout << "\t* ";
      else
         cout << "\t  ";

      printf("(%d) %20s (%d MB): \tCUDA Capability %d.%d \n", dev, devName, memMB, mjr, mnr);
   }

   cout << "****************************************";
   cout << "***************************************" << endl;
   return selectedDevice;

}












