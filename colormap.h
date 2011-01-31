#ifndef _COLORMAP_H_
#define _COLORMAP_H_


#define IDX_RED   0
#define IDX_GREEN 1
#define IDX_BLUE  2
#define IDX_ALPHA 3
#define NCHANNEL  4


// class colormap:
//    Stores a 256x3 matrix of values which defines how to map gray levels
//    to color information
template<typename CTYPE, int NCOLOR=256>
class colormap
{
public:

   CTYPE** cmapData_;


   CTYPE* operator[](int index) const
   {
      if(index < NCHANNEL)
         return cmapData_[index]; 
      else
         return NULL;
   }

   /////////////////////////////////////////////////////////////////////////////
   void Allocate(void)
   { 
      if(cmapData_ == NULL) 
      {
         cmapData_ = new CTYPE*[NCHANNEL];
         for(int ch=0; ch<NCHANNEL; ch++)
            cmapData_[ch] = new CTYPE[NCOLOR];
      }
      Reset();
   }

   /////////////////////////////////////////////////////////////////////////////
   void Destroy(void)
   { 
      if(cmapData_ != NULL) 
      {
         for(int ch=0; ch<NCHANNEL; ch++)
            delete[] cmapData_[ch];

         delete[] cmapData_;
      }
      cmapData_ = NULL;
   }

   /////////////////////////////////////////////////////////////////////////////
   void Reset(void)
   { 
      if(cmapData_ != NULL) 
      {
         for(int ch=0; ch<NCHANNEL; ch++)
            for(int color=0; color<NCOLOR; color++)
               cmapData_[ch][color] = -1;
      }
   }


   template<typename T>
   void copyToLinearArrayRowMajor(T* uintptr)
   {
      for(int c=0; c<256; c++)
         for(int ch=0; ch<3; ch++)
            uintptr[c*3+ch] = (T)(cmapData_[ch][c]);
      
   }

   template<typename T>
   void copyToLinearArrayColMajor(T* uintptr)
   {
      for(int c=0; c<256; c++)
         for(int ch=0; ch<3; ch++)
            uintptr[ch*256+c] = (T)(cmapData_[ch][c]);
      
   }

   /////////////////////////////////////////////////////////////////////////////
   colormap(void) : cmapData_(NULL) {}

   /////////////////////////////////////////////////////////////////////////////
   ~colormap(void) { Destroy(); }

   /////////////////////////////////////////////////////////////////////////////
   // Constructor:  load a colormap from file
   //    If the file contains map information in the range float[0,1] we 
   //    need to convert it to (CTYPE)[0,255].  The float[0,1] form
   //    is the way MATLAB stores colormap data
   colormap(string filename, int nChannel, bool fileFloats0to1=true) 
   {
      cmapData_ = NULL;
      Allocate();
      ifstream is(filename.c_str(), ios::in);
      double temp;
      for(int color=0; color<NCOLOR; color++)
      {
         for(int ch=0; ch<nChannel; ch++)
         {
            is >> temp;
            if(fileFloats0to1)
               temp = NCOLOR*(temp >= 0.999 ? 0.999 : temp);
            cmapData_[ch][color] = (CTYPE)temp;
         }
      }
   }


   colormap & operator=(colormap const & cmap2)
   {
      if(cmap2.cmapData_ != NULL)
      {
         Destroy();
         Allocate();
         
         for(int ch=0; ch<NCHANNEL; ch++)
            for(int color=0; color<NCOLOR; color++)
               cmapData_[ch][color] = cmap2.cmapData_[ch][color];

      } 
      else
      {
         Destroy();
      }
      return (*this);
   }

   colormap(colormap const & cmap2) : cmapData_(NULL)
   {
      if(cmap2.cmapData_ != NULL)
      {
         Allocate();
         for(int i=0; i<NCOLOR*NCHANNEL; i++)
            cmapData_[i] = cmap2.cmapData_[i];
      } 
      else
      {
         Destroy();
      }
   }


};

#endif
