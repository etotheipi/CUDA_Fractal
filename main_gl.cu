////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// Title:    Julia Set Explorer
// Author:   Alan Reiner
// Date:     15 Jan, 2011
//
// This program is my own independent Julia set calculator in CUDA, with chunks
// of NVIDIA SDK code to display the sets in interactive OpenGL windows
//
// The CUDA-OpenGL interop functionality is a bitch to implement (to put it
// lightly).  However, I found the following excellent document which describes
// it quite nicely (great for those of us who understand CUDA but not OpenGL):
//
//    http://www.nvidia.com/content/GTC/documents/1055_GTC09.pdf
//
// From that pdf:
//
// ----- Steps to setup OpenGL w/ CUDA -----------------------------------------
//    (1) Create a window (OS specific)
//          glutInit
//          glutInitDisplayMode
//          glutInitWindowSize
//          glutInitWindowPosition
//          glutCreateWindow
//
//    (2) Create a GL context (OS specific)
//          <not sure about this step...maybe not relevant for linux>
//
//    (3) Setup the GL viewport and coord system
//          glViewport
//          glMatrixMode
//          glLoadIdentity
//          glOrtho
//          glMatrixMode
//          glLoadIdentity
//          glEnable(GL_DEPTH_TEST)
//          glClearColor
//          glClear
//    
//    (4) Create the CUDA context 
//          cudaGLSetGLDevice(devID)
//
//    (5) Generate GL buffers to be shared with CUDA
//          glGenBuffers
//          glBindBuffer
//          glBufferData
//
//    (6) Register buffers with CUDA (tell GL and CUDA this buffer is shared)
//          //DEPRECATED: cudaGLRegisterBufferObject( bufObj )
//          //DEPRECATED: cudaGLUnregisterBufferObject( bufObj )
//          cudaGraphicsGLRegisterBuffer( pbo_resource, pbo, flag)
//          cudaGraphicsGLUnregisterBuffer( bufObj )  ... necessary?
//
// ----- Steps to draw an image from CUDA -----------------------------------------
//    (1) Allocate a GL buffer 
//          glGenBuffers 
//          glBindBuffer
//          glBufferData
//          cudaGLRegisterBufferObject
//
//    (2) Create a GL texture
//          glEnable
//          glGenTextures
//          glBindTexture
//          glTexImage2D
//          glTexParamateri (GL_LINEAR filters)
//
//    (3) Map the GL buffer to CUDA
//          cudaGLMapBufferObject(void**, bufObj)
//
//    (4) Write to the image 
//          CUDA can now use the mapped memory same as global mem
//
//    (5) Unmap the GL Buffer
//          cudaGLUnmapBufferObject(bufObj)
//
//    (6) Create a texture from the buffer
//          glBindBuffer
//          glBindTexture
//          glTexSubImage2D
//
//    (7) Draw the image
//
//    (8) Swap Buffers
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <GL/glew.h>

// For whatever reason, I can't find macros for these, so I made my own
#define ACR_MOUSE_WHEEL_ZOOM_IN  3
#define ACR_MOUSE_WHEEL_ZOOM_OUT 4

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#ifdef _WIN32
#include <GL/wglew.h>
#endif 

#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>


// Now, everything else I had originally
#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>
#include <math.h>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <stopwatch.h>
#include <assert.h>


#include "cudaImageHost.h"
#include "cudaImageDevice.h.cu"
#include "julia_kernel_cpu.h"
#include "julia_kernel.h.cu"
#include "writePNG.h"
#include "cudaUtilities.h.cu"

using namespace std;

// We ultimately want to render fractal in CUDA, and pass to OpenGL
// via pixel-buffer-objects (PBOs) and textures
GLuint gl_PBO = 0;
GLuint gl_TEX = 0;
GLuint gl_SHADER = 0;
cudaGraphicsResource* cuda_pbo_resource; // handles OpenGL-CUDA exchange

// TODO:  Using static window size at the moment
int imgWidth  = 1024;
int imgHeight = 1024;

unsigned int timer;


////////////////////////////////////////////////////////////////////////////////
//
// Program main
//
// TODO:  Remove the CUTIL calls so libcutil is not required to compile/run
//
////////////////////////////////////////////////////////////////////////////////

bool useCuda = false;

// Drive image resolution based on initial pixel-width and REAL-range
double realMin;
double realMax;
double imagMin;
double imagMax;

double realCent = 0.0;
double imagCent = 0.0;

double juliaC_real = -0.8;
double juliaC_imag = -0.156;

double basePixelSize1D = 4.0 / (double)imgWidth;
double pxSize;

double scale    =  1.0;
double scaleMin =  0.5;
double scaleMax = 50.0;

int fractalMaxIter = 256;
int logMaxIter = -1;

// Some mouse-interface params
int startPanX = 0;
int startPanY = 0;
double tempPanX = 0;
double tempPanY = 0;
bool leftClicked = false;
bool middleClicked = false;
bool rightClicked = false;

// Grid&Block Sizes for CUDA
dim3 BLOCK;
dim3 GRID;


// Use these for CPU version
GLubyte* h_Src = NULL;  
GLubyte* d_Dst = NULL;  
cudaImageHost<float> hostFractal;
cudaImageHost<unsigned int>  hostUIntFractal;
cudaImageDevice<unsigned int> devUIntFractal;

cudaImageHost<unsigned char>   hostColormap;
cudaImageDevice<unsigned char>  devColormap;

// Use these for GPU version
GLubyte* gpu_h_img;
GLubyte* gpu_d_img;


colormap<GLfloat, 256> currentCMAP;
string colormapFile;

int  runDevicePropertiesQuery(int argc, char **argv);
void writePngFile(unsigned int*, int, int, string, colormap<float,256>*);

bool cpuRender = true;
int  gpuRenderSM = 0;

////////////////////////////////////////////////////////////////////////////////
// Shader code taken directly from bilateralFilter.cpp in CUDA SDK 3.2
   static const char *shader_code = 
      "!!ARBfp1.0\n"
      "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
      "END";
   
   GLuint compileASMShader(GLenum program_type, const char *code)
   {
      GLuint program_id;
      glGenProgramsARB(1, &program_id);
      glBindProgramARB(program_type, program_id);
      glProgramStringARB(program_type, 
                         GL_PROGRAM_FORMAT_ASCII_ARB, 
                         (GLsizei) strlen(code), 
                         (GLubyte *) code);
   
      GLint error_pos;
      glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);
      if (error_pos != -1) 
      {
         const GLubyte *error_string;
         error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
         printf("Program error at position: %d\n%s\n", (int)error_pos, error_string);
         return 0;
      }
      return program_id;
   }

////////////////////////////////////////////////////////////////////////////////
// Need to initialize the GL buffers/textures and PBO resources
void myInitCudaAndOpenGL(int w, int h, int* argc=NULL, char** argv=NULL)
{

   // Select the CUDA device
   static int deviceId;
   static bool firstRun = true;
   if(firstRun)
   {
      firstRun = false;
      // Selects the device and returns the ID
      deviceId = runDevicePropertiesQuery(*argc, argv);
   }

   // Inform CUDA and GL they will be working together on the same device
   cudaGLSetGLDevice(deviceId); 

   ///// INIT CUDA /////
   // [Re-]allocate device memory
   //if(gpu_d_img)
      //cudaFree(gpu_d_img);
   //cutilSafeCall( cudaMalloc( (void**)&gpu_d_img, w*h*sizeof(unsigned int) ));

   // [Re-]allocate some host memory and texture memory on device
   //if(gpu_h_img)
      //delete gpu_h_img;

   //gpu_h_img = new GLubyte[w*h*4];
   //for(int e=0; e<w*h*4; e++)
      //gpu_h_img[e] = 0;

   // Not sure this is necessary for my app
   //initTexture(w, h, gpu_h_img);


   ///// INIT OPENGL /////
   // Create the PBO
   glGenBuffers(1, &gl_PBO);
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
   glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w*h*sizeof(GLubyte)*4,
                                            NULL, GL_DYNAMIC_COPY);
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

   
   // Register the cuda resource with the PBO
   cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, 
                                gl_PBO, 
                                cudaGraphicsMapFlagsWriteDiscard));
   //cutilSafeCall(cudaGraphicsGLUnregisterBuffer(&cuda_pbo_resource, 
                                //gl_PBO, 
                                //cudaGraphicsMapFlagsWriteDiscard));

   // Create the Texture
   glEnable(GL_TEXTURE_2D);
   glGenTextures(1, &gl_TEX);
   glBindTexture(GL_TEXTURE_2D, gl_TEX);
   glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glBindTexture(GL_TEXTURE_2D, 0);


   gl_SHADER = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);

   // Create a CUDA stream for mapping graphics resources 
   // --Not sure why we need this in the init func; thought we do this every draw
   //size_t imgSize;
   //cudaStream_t cuda_stream;
   //cudaStreamCreate(&cuda_stream);
   //cudaGraphicsMapResources(1, &cuda_pbo_resource, cuda_stream);
   //cudaGraphicsResourceGetMappedPointer((void**)(&d_Dst), &imgSize, cuda_pbo_resource);
   //cudaGraphicsUnmapResources(1, &cuda_pbo_resource, cuda_stream);
   //cudaStreamDestroy(cuda_stream);

   hostFractal = cudaImageHost<float>(w, h);
   h_Src = (GLubyte*)malloc(w * h * 4 * sizeof(GLubyte));
}

////////////////////////////////////////////////////////////////////////////////
//
// Initialize colormap and textures
//
void myInitData(void)
{
   int blockSizeX = 16;
   int blockSizeY = 16;
   int gridSizeX = (imgWidth-1)  / blockSizeX + 1;
   int gridSizeY = (imgHeight-1) / blockSizeY + 1;
   BLOCK = dim3(blockSizeX, blockSizeY, 1);
   GRID  = dim3( gridSizeX,  gridSizeY, 1);

   // Create a colormap for display file to saved with
   currentCMAP = colormap<GLfloat, 256>(colormapFile, 3);
   unsigned int cmapUint[768];
   currentCMAP.copyToLinearArrayColMajor(cmapUint);

   
   // Put the colormap in CUDA texture memory
   // (yes, textures are also quite complicated)
   cudaArray* caCmap;
   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned int>();
   cudaExtent texExt = make_cudaExtent(256,3,1);
   cutilSafeCall( cudaMalloc3DArray(&caCmap, &channelDesc, texExt, 0) );
   prepareCudaTexture(cmapUint, caCmap, texExt);
   cudaGetTextureReference(&tex_uint_ref, "tex_uint");
   tex_uint.addressMode[0] = cudaAddressModeClamp;
   tex_uint.addressMode[0] = cudaAddressModeClamp;
   tex_uint.addressMode[0] = cudaAddressModeClamp;
   tex_uint.filterMode     = cudaFilterModePoint;
   tex_uint.normalized     = false;
   cudaBindTextureToArray(tex_uint_ref, caCmap, &channelDesc);

}

void drawText(string s, 
              void* font,
              int nRowPx,
              int nColPx,
              float r=-1.0f, 
              float g=-1.0f, 
              float b=-1.0f)
{
   float pxWidth  = 1.0 / (float)imgWidth;
   float pxHeight = 1.0 / (float)imgHeight;
   float worldXpos = (float)nColPx * pxWidth;
   float worldYpos = (float)(imgHeight-nRowPx+1) * pxHeight;
   if(r>0.0f)
      glColor3f(r,g,b);

   glRasterPos2f(worldXpos, worldYpos);
   for(string::iterator i=s.begin(); i!=s.end(); i++)
   {
      char c = *i;
      glutBitmapCharacter(font, c);
   }
}

////////////////////////////////////////////////////////////////////////////////
void displayFunction(void)
{
   glClear(GL_COLOR_BUFFER_BIT);

   // Enforce the min/max scaling
   scale = (scale < scaleMin ? scaleMin : scale);
   scale = (scale > scaleMax ? scaleMax : scale);

   // Set rendering parameters
   pxSize = basePixelSize1D / scale;
   realMin = realCent - ((double)imgWidth  * pxSize / 2.0) + tempPanX;
   realMax = realCent + ((double)imgWidth  * pxSize / 2.0) + tempPanX;
   imagMin = imagCent - ((double)imgHeight * pxSize / 2.0) + tempPanY;
   imagMax = imagCent + ((double)imgHeight * pxSize / 2.0) + tempPanY;


   // Let's time this thing, too
   cpuStartTimer();


   //if(true)
   if(cpuRender | gpuRenderSM == 0)
   {
      GenerateJuliaTile_z2plusc_CPU( hostFractal.getDataPtr(),
                                     juliaC_real,
                                     juliaC_imag,
                                     imgWidth,
                                     imgHeight,
                                     realMin,
                                     imagMin,
                                     pxSize,
                                     pxSize,
                                     fractalMaxIter) ;

      for(int e=0; e<imgWidth*imgHeight; e++)
      {
         int temp = (int)(256.0f*hostFractal[e]);
         temp = (temp > 255 ? 255 : temp);
         temp = (temp <   0 ?   0 : temp);
         h_Src[4*e+0] = (GLubyte)(currentCMAP[IDX_RED  ][temp]);
         h_Src[4*e+1] = (GLubyte)(currentCMAP[IDX_GREEN][temp]);
         h_Src[4*e+2] = (GLubyte)(currentCMAP[IDX_BLUE ][temp]);
         h_Src[4*e+3] = (GLubyte)(255);
      }
   
      glDrawPixels(imgWidth, imgHeight, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);

   }
   else  // For simple Julia sets, all CUDA architectures run SM10 alg well
   {

      unsigned int* d_result;
      cutilSafeCall(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
      size_t num_bytes; 
      cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&d_result, 
                                                      &num_bytes,  
						                                    cuda_pbo_resource));

      //glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
      //glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, h_Src, GL_STREAM_COPY);

      GenerateJuliaTile_z2plusc_SM10<<<GRID, BLOCK>>>( 
      //GenerateJuliaTile_cexpz_SM20<<<GRID, BLOCK>>>( 
      //GenerateJuliaTile_other_SM20<<<GRID, BLOCK>>>( 
                                       d_result,
                                       juliaC_real,
                                       juliaC_imag,
                                       imgWidth,
                                       imgHeight,
                                       realMin,
                                       imagMin,
                                       pxSize,
                                       pxSize,
                                       fractalMaxIter) ;


      // ******************************************************************** //
      // **** DEBUG CUDA CODE **** //
       
      /*
      cudaImageHost<unsigned int> outImg(imgWidth, imgHeight);
      cudaMemcpy(outImg.getDataPtr(), 
                 d_result, 
                 imgWidth*imgHeight*sizeof(unsigned int), 
                 cudaMemcpyDeviceToHost);
      ofstream testout("test_fract.txt", ios::out);
      for(int row=0; row<imgHeight; row++)
      {
         for(int col=0; col<imgWidth; col++)
            testout << outImg(row,col) << " ";
         testout << endl;
      }
      testout.close();
      exit(0);
      */
      // **** DEBUG CUDA CODE **** //
      // ******************************************************************** //
            

      cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

      
      glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
      glBindTexture(GL_TEXTURE_2D, gl_TEX);
      glTexSubImage2D(GL_TEXTURE_2D, 0,0,0, 
                      imgWidth, imgHeight,
                      GL_RGBA, GL_UNSIGNED_BYTE, 0);
      glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

      // fragment program is required to display floating point texture
      glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_SHADER);
      glEnable(GL_FRAGMENT_PROGRAM_ARB);
      glDisable(GL_DEPTH_TEST);

      glBegin(GL_QUADS);
      {
         glTexCoord2f(0, 0);          
         glVertex2f(  0, 0);
         glTexCoord2f(1, 0);          
         glVertex2f(  1, 0);
         glTexCoord2f(1, 1);          
         glVertex2f(  1, 1);
         glTexCoord2f(0, 1);          
         glVertex2f(  0, 1);
      }
      glEnd();
      glBindTexture(GL_TEXTURE_2D, 0);
      glDisable(GL_FRAGMENT_PROGRAM_ARB);

   }


   // Draw menu in upper-left corner, use pixel coords
   vector<string> dispMenu(10);
   dispMenu[0] = "Use mouse to adjust fractal view";
   dispMenu[1] = "   Left mouse button to pan the display";
   dispMenu[2] = "   Right mouse button to change the Julia c-param";
   dispMenu[3] = "   Scroll wheel to zoom";
   dispMenu[4] = "Use the arrow keys for fine tuning";
   dispMenu[5] = "   No modifiers - pan display";
   dispMenu[6] = "   Ctrl key     - pan slower";
   dispMenu[7] = "   Shift key    - change Julia c-param";
   dispMenu[8] = "   Shift & Ctrl - change Julia c-param slower";
   dispMenu[9] = "+/- for zooming";

   int startRow = 40;
   int startCol = 20;
   int rowStep = 16;
   glColor3f(1.0f, 1.0f, 1.0f);
   for(int i=0; i<10; i++)
   {
      int row = startRow + i*rowStep; 
      int col = startCol;
      drawText( dispMenu[i], GLUT_BITMAP_9_BY_15, row, col);
   }
          
            


   glutSwapBuffers();

   //cout << 1000./cpuStopTimer() << " frames per second." << endl;


   // Debugging code
   //std::ofstream os("test_h_Src.txt", ios::out);
   //for(int row=0; row<imgHeight; row++)
   //{
      //for(int col=0; col<imgWidth; col++)
         //os << (int)(h_Src[row*imgWidth+col]) << " ";
      //os << endl;
   //}
   //exit(0);
         
}


////////////////////////////////////////////////////////////////////////////////
void idleFunc()
{
   glutPostRedisplay();
}


////////////////////////////////////////////////////////////////////////////////
// Define what happens when an ASCII key is pressed
void keyboardFunc(unsigned char k, int, int)
{
   // TODO:  Why doesn't the shift-detection work??
   // UPDATE:  Shift doesn't seem to work w/ number pad
   //int modifiers = glutGetModifiers(); 
   //bool shiftIsDown = (modifiers & GLUT_ACTIVE_SHIFT > 0);
   //bool  ctrlIsDown = (modifiers & GLUT_ACTIVE_CTRL  > 0);
   //bool   altIsDown = (modifiers & GLUT_ACTIVE_ALT   > 0);
   switch (k)
   {
      case '\033':
      case 'q':
      case 'Q':
         printf("Shutting down...\n");
         exit(EXIT_SUCCESS);
         break;

      case '-':
         scale /= 1.1f; break;
      case '+':
         scale *= 1.1f; break;
      case 'p':
      case 'P':
         cout << "C parameter = " << juliaC_real << " + " << juliaC_imag << "i" << endl;
         break;
      case 'c':
         colormapFile = "cmap_blue_green.txt";
         myInitData();
         break;
      default:
          break;
   }

} // keyboardFunc

void keyboardUpFunc(unsigned char k, int, int)
{
   // Do nothing for now
}

void nonAsciiFunc(int k, int, int)
{
   int  modifiers = glutGetModifiers(); 
   bool shiftIsDown = ((modifiers & GLUT_ACTIVE_SHIFT) > 0);
   bool ctrlIsDown  = ((modifiers & GLUT_ACTIVE_CTRL)  > 0);
   bool altIsDown   = ((modifiers & GLUT_ACTIVE_ALT)   > 0);
   double cvalShift = pxSize/(3.0*scale);
   double ctrShift = pxSize/scale;

   if( !shiftIsDown && !ctrlIsDown && !altIsDown)           // No modifiers
   {
      if(k == GLUT_KEY_LEFT ) { realCent -= 10*ctrShift; }
      if(k == GLUT_KEY_RIGHT) { realCent += 10*ctrShift; }
      if(k == GLUT_KEY_DOWN ) { imagCent -= 10*ctrShift; }
      if(k == GLUT_KEY_UP   ) { imagCent += 10*ctrShift; }
   }
   else if( !shiftIsDown && ((ctrlIsDown)) && !altIsDown )  // Ctrl only
   {
      if(k == GLUT_KEY_LEFT ) { realCent -= ctrShift; }
      if(k == GLUT_KEY_RIGHT) { realCent += ctrShift; }
      if(k == GLUT_KEY_DOWN ) { imagCent -= ctrShift; }
      if(k == GLUT_KEY_UP   ) { imagCent += ctrShift; }
   }
   else if( ((shiftIsDown)) && !ctrlIsDown && !altIsDown )  // Shift only
   {
      if(k == GLUT_KEY_LEFT ) { juliaC_real -= 10*cvalShift; }
      if(k == GLUT_KEY_RIGHT) { juliaC_real += 10*cvalShift; }
      if(k == GLUT_KEY_DOWN ) { juliaC_imag -= 10*cvalShift; }
      if(k == GLUT_KEY_UP   ) { juliaC_imag += 10*cvalShift; }
   }
   else if( ((shiftIsDown)) && ((ctrlIsDown)) && !altIsDown )  // Shift+Ctrl
   {
      if(k == GLUT_KEY_LEFT ) { juliaC_real -= cvalShift; }
      if(k == GLUT_KEY_RIGHT) { juliaC_real += cvalShift; }
      if(k == GLUT_KEY_DOWN ) { juliaC_imag -= cvalShift; }
      if(k == GLUT_KEY_UP   ) { juliaC_imag += cvalShift; }
   }
   else
   {
      // ehhh... nothing goes here
   }

}
   

void nonAsciiUpFunc(int k, int, int)
{
   //cout << "C parameter = " << juliaC_real << " + " << juliaC_imag << "i" << endl;
}


////////////////////////////////////////////////////////////////////////////////
void motionFunc(int x, int y)
{
    // This was the original SDK code for fx/fy
    //double fx = (double)(x - lastx) / 50.0 / (double)(imageW);        
    //double fy = (double)(lasty - y) / 50.0 / (double)(imageH);

   // Want to calculate the part of the complex plane clicked
   //double fx = (double)x / (double)(imgWidth);        
   //double fy = (double)y / (double)(imgHeight);

   if (leftClicked) 
   {
      int distX = startPanX - x; 
      int distY = startPanY - y; 
      
      tempPanX =    (float)distX * pxSize;
      tempPanY = -1*(float)distY * pxSize;
   } 
   else if(rightClicked)
   {
      juliaC_real = realMin + x*pxSize/scale;
      juliaC_imag = imagMin + (imgHeight-(y+1))*pxSize/scale;
   }
        

} // motionFunc



////////////////////////////////////////////////////////////////////////////////
void clickFunc(int button, int state, int x, int y)
{
   if (button == GLUT_LEFT_BUTTON)
   {
      if (state == GLUT_DOWN)
      {
         leftClicked = true;
         startPanX = x;
         startPanY = y;
      }
      if ( state == GLUT_UP)
      {
         leftClicked = false;
         realCent += tempPanX;
         imagCent += tempPanY;
         tempPanX = 0;
         tempPanY = 0;
      }
   }

   if (button == GLUT_RIGHT_BUTTON)
   {
      if (state == GLUT_DOWN)
         rightClicked = true;
      else
         rightClicked = false;
      motionFunc(x,y);
   }

   if(button == ACR_MOUSE_WHEEL_ZOOM_IN)
      scale *= 1.05f;
   if(button == ACR_MOUSE_WHEEL_ZOOM_OUT)
      scale /= 1.05f;

} // clickFunc



////////////////////////////////////////////////////////////////////////////////
// TODO:  I haven't really planned on allowing reshaping the OpenGL window, but
//        this should probably run for setting OpenGL params
void reshapeFunc(int w, int h)
{
   glViewport(0, 0, imgWidth, imgHeight);

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(0.0, 1.0, 0.0, 1.0, -1, 1);

   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();

   //glEnable(GL_DEPTH_TEST);
   //glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
   //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   //
} // reshapeFunc


////////////////////////////////////////////////////////////////////////////////
void initGL(int *argc, char **argv)
{
   printf("Initializing GLUT...\n");
   glutInit(argc, argv);
   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
   glutInitWindowSize(imgWidth, imgHeight);
   glutInitWindowPosition(0, 0);
   glutCreateWindow(argv[0]);

   printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));
   if (!glewIsSupported( "GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object" )) 
   {
      fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
      fprintf(stderr, "This sample requires:\n");
      fprintf(stderr, "  OpenGL version 1.5\n");
      fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
      fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
      //cutilExit(*argc, argv);
   }
   printf("OpenGL window created.\n");

}

////////////////////////////////////////////////////////////////////////////////
/*
void initData(int argc, char **argv)
{
   // check for hardware double precision support
   int dev = 0;
   cutGetCmdLineArgumenti(argc, (const char **) argv, "device", &dev);

   cudaDeviceProp deviceProp;
   cutilSafeCall(cudaGetDeviceProperties(&deviceProp, dev));
   int version = deviceProp.major*10 + deviceProp.minor;
   if (version < 11) {
      printf("GPU compute capability is too low (1.0)\n Press Enter to exit the program\n") ;  
      std::getchar();
      return ; 
   }
   //bool haveDoubles = (version >= 13);

   //int numSMs = deviceProp.multiProcessorCount;

   // initialize some of the arguments    
   float x;
   if (cutGetCmdLineArgumentf(argc, (const char **)argv, "xOff", &x)) 
   {
      //
   }
   if (cutGetCmdLineArgumentf(argc, (const char **)argv, "yOff", &x)) 
   {
      //
   }
   if (cutGetCmdLineArgumentf(argc, (const char **)argv, "scale", &x)) 
   {
      //
   }
}
*/



void cleanup()
{

   //double elapsedTime = gpuStopTimer();
    
   //cudaGraphicsUnregisterResource(cuda_pbo_resource);
   //glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

   cudaGraphicsUnregisterResource(cuda_pbo_resource);
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
   glDeleteBuffers(1, &gl_PBO);
   glDeleteTextures(1, &gl_TEX);
   glDeleteProgramsARB(1, &gl_SHADER);

   //if (g_FrameBufferObject) 
   //{
       //delete g_FrameBufferObject; 
       //g_FrameBufferObject = NULL;
   //}

   //if (g_CheckRender) 
   //{
      //delete g_CheckRender; 
      //g_CheckRender = NULL;
   //}
}


int main( int argc, char** argv) 
{

   cout << "\n***************************************"
        << "*****************************************" << endl;
   cout << "***Starting CUDA-accelerated fractal generation" << endl << endl;

   colormapFile = "cmap_blue_gray_yellow.txt";

   // Always do this first, as a sanity check:
   initGL(  &argc, argv);
   myInitCudaAndOpenGL(imgWidth, imgHeight, &argc, argv);
   myInitData();
  
   glutDisplayFunc(displayFunction);
   glutIdleFunc(idleFunc);
   // Ascii-character key presses
   glutKeyboardFunc(keyboardFunc);
   glutKeyboardUpFunc(keyboardUpFunc);
   // Non-ascii presses, like arrows and F1-12 keys
   glutSpecialFunc(nonAsciiFunc);
   glutSpecialUpFunc(nonAsciiUpFunc);
   glutMouseFunc(clickFunc);
   glutMotionFunc(motionFunc);
   glutReshapeFunc(reshapeFunc);

   atexit(cleanup);

#ifdef _WIN32
   setVSync(0) ; 
#endif 


   glutMainLoop();
   cudaThreadExit();
   cutilExit(argc, argv);
   exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Query the devices on the system and select the fastest (or override the
// selectedDevice variable to choose your own
int runDevicePropertiesQuery(int argc, char **argv)
{
   int selectedDevice = -1;
   cutGetCmdLineArgumenti(argc, (const char **) argv, "device", &selectedDevice);

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
   if(selectedDevice == -1)
   {
      selectedDevice = cutGetMaxGflopsDeviceId() ;
      cudaSetDevice(selectedDevice);
   }
   

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
      {
         cout << "\t* ";
         gpuRenderSM = 10*mjr + mnr;
         cpuRender = false;
      }
      else
         cout << "\t  ";

      printf("(%d) %20s (%d MB): \tCUDA Capability %d.%d \n", dev, devName, memMB, mjr, mnr);
   }

   cout << "****************************************";
   cout << "***************************************" << endl;
   return selectedDevice;

}












