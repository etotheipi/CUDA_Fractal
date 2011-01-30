#ifndef _WRITE_PNG_H_
#define _WRITE_PNG_H_

#include <string.h>
#include <png.h>
#include "colormap.h"



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// Write a matrix of data to a png file.  Default is grayscale.  Can create
// and supply pointer to a colormap to add color
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<typename T>
void writePngFile(T const * data, 
                  int NROWS, 
                  int NCOLS, 
                  string filename,
                  colormap<float, 256>* cmap=NULL)
{

   // Pull "raw" data out of the img
   int const NELTS = NROWS*NCOLS;

   // Get min and max values in image, so we know how to scale them
   double minVal = data[0];
   double maxVal = data[0];
   for(int e=1; e<NELTS; e++)
   {
      double val = (double)(data[e]);
      if(val < minVal)
         minVal = val;
      if(val > maxVal)
         maxVal = val;
   }
   int dynRng = maxVal - minVal;


   // Now create the png_byte** that the png_info struct will need
   png_bytepp row_pointers = (png_bytep *)malloc( sizeof(png_bytep) * NROWS);
   for(int row=0; row<NROWS; row++)
      row_pointers[row] =  (png_byte *)malloc(sizeof(png_byte) * NCOLS * 3);

   unsigned int tempGrey;
   double val;
   for(int row=0; row<NROWS; row++)
   {
      for(int col=0; col<NCOLS; col++)
      {
         val = (double)(data[row*NCOLS + col]);
     
         tempGrey = (int)(256.0 * (val - minVal) / dynRng);
         tempGrey = (tempGrey > 255 ? 255 : tempGrey);
       
         if(cmap == NULL)
         {
            // For now, just write out red images, to verify byte-order/endianness
            row_pointers[row][3*col+0] = tempGrey;
            row_pointers[row][3*col+1] = tempGrey;
            row_pointers[row][3*col+2] = tempGrey;
         }
         else
         {
            row_pointers[row][3*col+0] = (unsigned int)cmap->cmapData_[IDX_RED  ][tempGrey];
            row_pointers[row][3*col+1] = (unsigned int)cmap->cmapData_[IDX_GREEN][tempGrey];
            row_pointers[row][3*col+2] = (unsigned int)cmap->cmapData_[IDX_BLUE ][tempGrey];
         }
      }
   }




   FILE *fp = fopen(filename.c_str(), "wb");
   assert(fp);
   
   // Create PNG struct
   png_structp png_ptr = png_create_write_struct( PNG_LIBPNG_VER_STRING, 
                                                  (png_voidp)NULL, 
                                                  NULL, 
                                                  NULL);
   assert(png_ptr);

   // Create PNG info struct
   png_infop info_ptr = png_create_info_struct(png_ptr);
   if(!info_ptr)
   {
      png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
      assert(0);
   }

   // Initialize I/O for PNG on the file
   png_init_io(png_ptr, fp);


   // Is this good for anything??
   //png_color_8 sig_bit;
   //sig_bit.red   = 8;
   //sig_bit.green = 8;
   //sig_bit.blue  = 8;
   //sig_bit.alpha = 8;
   //png_set_sBIT(png_ptr, info_ptr, &sig_bit);

   // Can control PNG compression/speed ratio
   // png_set_filter( png_ptr, 0,
   //                 PNG_FILTER_NONE | PNG_FILTER_VALUE_NONE |
   //                 PNG_FILTER_SUB | PNG_FILTER_VALUE_SUB |
   //                 PNG_FILTER_UP | PNG_FILTER_VALUE_UP |
   //                 PNG_FILTER_AVG | PNG_FILTER_VALUE_AVG |
   //                 PNG_FILTER_PAETH | PNG_FILTER_VALUE_PAETH !
   //                 PNG_ALL_FILTERS);

   // set the zlib compression level //
   // png_set_compression_level(png_ptr, Z_BEST_COMPRESSION);
   // png_set_compression_mem_level(png_ptr, 8);
   // png_set_compression_strategy(png_ptr, Z_DEFAULT_STRATEGY);
   // png_set_compression_window_bits(png_ptr, 15);
   // png_set_compression_method(png_ptr, 8);
   // png_set_compression_buffer_size(png_ptr, 8192)

   // Setting options in info obj
   // width         - holds the width of the image in pixels (up to 2ˆ31).
   // height        - holds the height of the image in pixels (up to 2ˆ31).
   // bit_depth     - holds the bit depth of each image channel.  (valid values 
   //                are 1, 2, 4, 8, 16 and depend also on the color_type. See 
   //                also significant bits (sBIT) below).
   // color_type    - describes which color/alpha channels are present.
   //                 PNG_COLOR_TYPE_GRAY (bit depths 1, 2, 4, 8, 16)
   //                 PNG_COLOR_TYPE_GRAY_ALPHA (bit depths 8, 16)
   //                 PNG_COLOR_TYPE_PALETTE (bit depths 1, 2, 4, 8)
   //                 PNG_COLOR_TYPE_RGB (bit_depths 8, 16)
   //                 PNG_COLOR_TYPE_RGB_ALPHA (bit_depths 8, 16)
   //                 PNG_COLOR_MASK_PALETTE
   //                 PNG_COLOR_MASK_COLOR
   //                 PNG_COLOR_MASK_ALPHA
   // interlace_type - PNG_INTERLACE_NONE or
   //                 PNG_INTERLACE_ADAM7
   // compression_type - (must be PNG_COMPRESSION_TYPE_DEFAULT)
   // filter_method - (must be PNG_FILTER_TYPE_DEFAULT)
   png_set_IHDR( png_ptr, 
                 info_ptr, 
                 NCOLS,
                 NROWS,
                 8, 
                 PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, 
                 PNG_FILTER_TYPE_DEFAULT);


   // PNG transformations:
   //
   //   PNG_TRANSFORM_IDENTITY;     No transformation
   //   PNG_TRANSFORM_PACKING;      Pack 1, 2 and 4-bit samples 
   //   PNG_TRANSFORM_PACKSWAP;     Change order of packed pixels to LSB first
   //   PNG_TRANSFORM_INVERT_MONO   Invert monochrome images
   //   PNG_TRANSFORM_SHIFT         Normalize pixels to the sBIT depth
   //   PNG_TRANSFORM_BGR           Flip RGB to BGR, RGBA to BGRA
   //   PNG_TRANSFORM_SWAP_ALPHA    Flip RGBA to ARGB or GA to AG
   //   PNG_TRANSFORM_INVERT_ALPHA  Change alpha from opacity to transparency
   //   PNG_TRANSFORM_SWAP_ENDIAN   Byte-swap 16-bit samples
   //   PNG_TRANSFORM_STRIP_FILLER  Strip out filler bytes (deprecated).
   //   PNG_TRANSFORM_STRIP_FILLER_BEFORE Strip out leading filler bytes
   //   PNG_TRANSFORM_STRIP_FILLER_AFTER Strip out trailing filler bytes
   //
   // Bitwise-OR appropriate flags together
   int png_transforms = PNG_TRANSFORM_IDENTITY;


   png_set_rows(png_ptr, info_ptr, row_pointers);


   png_write_png(png_ptr, info_ptr, png_transforms, NULL);


   for(int row=0; row<NROWS; row++)
      free(row_pointers[row]);
   free(row_pointers);
}

template<>
void writePngFile<unsigned int>(unsigned int const * data, 
                  int NROWS, 
                  int NCOLS, 
                  string filename,
                  colormap<float, 256>* cmap)
{
   cout << NROWS << " " << NCOLS << endl;

   // Create the png_byte** that the png_info struct will need
   png_bytepp rptrs = (png_bytep *)malloc( sizeof(png_bytep) * NROWS);
   for(int row=0; row<NROWS; row++)
      rptrs[row] =  (png_byte *)malloc(sizeof(png_byte) * NCOLS * 3);

   unsigned int tempGrey;
   for(int row=0; row<NROWS; row++)
   {
      for(int col=0; col<NCOLS; col++)
      {
         tempGrey = data[row*NCOLS + col];
     
         if(cmap == NULL)
         {
            // For now, just write out red images, to verify byte-order/endianness
            rptrs[row][3*col+0] = (unsigned int)tempGrey;
            rptrs[row][3*col+1] = (unsigned int)tempGrey;
            rptrs[row][3*col+2] = (unsigned int)tempGrey;
         }
         else
         {
            rptrs[row][3*col+0] = (unsigned int)cmap->cmapData_[IDX_RED  ][tempGrey];
            rptrs[row][3*col+1] = (unsigned int)cmap->cmapData_[IDX_GREEN][tempGrey];
            rptrs[row][3*col+2] = (unsigned int)cmap->cmapData_[IDX_BLUE ][tempGrey];
         }
      }
   }

   FILE *fp = fopen(filename.c_str(), "wb");
   assert(fp);
   
   // Create PNG struct
   png_structp png_ptr = png_create_write_struct( PNG_LIBPNG_VER_STRING, 
                                                  (png_voidp)NULL, 
                                                  NULL, 
                                                  NULL);
   assert(png_ptr);

   // Create PNG info struct
   png_infop info_ptr = png_create_info_struct(png_ptr);
   if(!info_ptr)
   {
      png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
      assert(0);
   }

   // Initialize I/O for PNG on the file
   png_init_io(png_ptr, fp);

   png_set_IHDR( png_ptr, 
                 info_ptr, 
                 NCOLS,
                 NROWS,
                 8, 
                 PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, 
                 PNG_FILTER_TYPE_DEFAULT);

   int png_transforms = PNG_TRANSFORM_IDENTITY;


   png_set_rows(png_ptr, info_ptr, rptrs);


   png_write_png(png_ptr, info_ptr, png_transforms, NULL);


   for(int row=0; row<NROWS; row++)
      free(rptrs[row]);
   free(rptrs);
}

#endif
