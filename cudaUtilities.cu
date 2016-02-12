#include <iostream>
#include <fstream>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>
#include <helper_timer.h>

#include "cudaUtilities.cuh"


// Timer variables
StopWatchInterface *gpuTimerObj=NULL;
cudaEvent_t eventTimerStart;
cudaEvent_t eventTimerStop;




////////////////////////////////////////////////////////////////////////////////
// Simple Timing Calls
void cpuStartTimer(void)
{
   // GPU Timer Functions
   sdkCreateTimer( &gpuTimerObj );
   sdkStartTimer(  &gpuTimerObj );
}

////////////////////////////////////////////////////////////////////////////////
// Stopping also resets the timer
// returns milliseconds
float cpuStopTimer(void)
{
   sdkStopTimer( &gpuTimerObj  );
   float cpuTime = sdkGetTimerValue( &gpuTimerObj );
   sdkDeleteTimer( &gpuTimerObj );
   return cpuTime;
}

////////////////////////////////////////////////////////////////////////////////
// Timing Calls for GPU -- this only counts GPU clock cycles, which will be 
// more precise for measuring GFLOPS and xfer rates, but shorter than wall time
void gpuStartTimer(void)
{
   cudaEventCreate(&eventTimerStart);
   cudaEventCreate(&eventTimerStop);
   cudaEventRecord(eventTimerStart);
}

////////////////////////////////////////////////////////////////////////////////
// Stopping also resets the timer
float gpuStopTimer(void)
{
   cudaEventRecord(eventTimerStop);
   cudaEventSynchronize(eventTimerStop);
   float gpuTime;
   cudaEventElapsedTime(&gpuTime, eventTimerStart, eventTimerStop);
   return gpuTime;
}

////////////////////////////////////////////////////////////////////////////////
// Read/Write images from/to files
void ReadFile(string fn, int* targPtr, int nRows, int nCols)
{
   ifstream in(fn.c_str(), ios::in);
   // We work with Row-Col format, but files written in Col-Row, so switch loop
   for(int r=0; r<nRows; r++)
      for(int c=0; c<nCols; c++)
         in >> targPtr[r*nRows+c];
   in.close();
}

////////////////////////////////////////////////////////////////////////////////
// Writing file in space-separated format
void WriteFile(string fn, int* srcPtr, int nRows, int nCols)
{
   ofstream out(fn.c_str(), ios::out);
   // We work with Row-Col format, but files written in Col-Row, so switch loop
   for(int r=0; r<nRows; r++)
   {
      for(int c=0; c<nCols; c++)
      {
         out << srcPtr[r*nCols+c] << " ";
      }
      out << endl;
   }
   out.close();
}

////////////////////////////////////////////////////////////////////////////////
// Writing image to stdout
void PrintArray(int* srcPtr, int nRows, int nCols)
{
   // We work with Row-Col format, but files written in Col-Row, so switch loop
   for(int r=0; r<nRows; r++)
   {
      cout << "\t";
      for(int c=0; c<nCols; c++)
      {
         cout << srcPtr[r*nCols+c] << " ";
      }
      cout << endl;
   }
}
