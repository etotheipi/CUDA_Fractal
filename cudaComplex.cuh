#ifndef _COMPLEX_NUMBER_H_
#define _COMPLEX_NUMBER_H_


#include <iostream>

// ************************************************************************** //
// Basic cudaComplex numbers implemented in the simplest, most efficient
// ************************************************************************** //

template<typename T>
class cudaComplex
{
private:

   T RE_;
   T IM_;

public:

   __device__ cudaComplex() : RE_(0), IM_(0) {}
   __device__ cudaComplex(T re) : RE_(re), IM_(0) {}
   __device__ cudaComplex(T re, T im) : RE_(re), IM_(im) {}

   __device__ T const & real(void) const { return RE_; }
   __device__ T const & imag(void) const { return IM_; }
   __device__ T & real(void) { return RE_; }
   __device__ T & imag(void) { return IM_; }

   __device__ cudaComplex sq(void) { return cudaComplex( RE_*RE_ - IM_*IM_, 2*RE_*IM_ ); }
   __device__ T  conj_sq(void) { return RE_*RE_ + IM_*IM_; }

   __device__ cudaComplex conj() const { return cudaComplex( RE_, -IM_ ); }
   __device__ cudaComplex operator-() const { return cudaComplex(-RE_, -IM_ ); }

   __device__ cudaComplex operator+(cudaComplex const & z2) const;
   __device__ cudaComplex operator-(cudaComplex const & z2) const;
   __device__ cudaComplex operator*(cudaComplex const & z2) const;
   __device__ cudaComplex operator/(cudaComplex const & z2) const;

   // Define zfunctions to be complex versions of regular real-valued functions
   __device__ cudaComplex zexp(void) const;
   __device__ cudaComplex zlog(void) const;
   __device__ cudaComplex zsin(void) const;
   __device__ cudaComplex zcos(void) const;
   __device__ cudaComplex ztan(void) const;
   __device__ cudaComplex zpow(cudaComplex const & c) const;
   __device__ cudaComplex zpow(T const & c) const;

   template<typename T2> __device__ cudaComplex operator+(T2 const & n2) const { return cudaComplex( RE_ + n2, IM_      ); }
   template<typename T2> __device__ cudaComplex operator-(T2 const & n2) const { return cudaComplex( RE_ - n2, IM_      ); }
   template<typename T2> __device__ cudaComplex operator*(T2 const & n2) const { return cudaComplex( RE_ * n2, IM_ * n2 ); }
   template<typename T2> __device__ cudaComplex operator/(T2 const & n2) const { return cudaComplex( RE_ / n2, IM_ / n2 ); }


   // TODO:  Add versions of these functions that take references/pointers for
   //        the output variable, so that variables can be created once in the
   //        calling code, and no extra copying is going on (we lose 5-10% of
   //        our computation speed due to these extra copies

   __device__ friend cudaComplex operator+(float n2, cudaComplex const & z) { return cudaComplex( n2 + z.RE_,  z.IM_ ); }
   __device__ friend cudaComplex operator-(float n2, cudaComplex const & z) { return cudaComplex( n2 - z.RE_, -z.IM_ ); }
   __device__ friend cudaComplex operator*(float n2, cudaComplex const & z) { return cudaComplex( n2 * z.RE_, n2 * z.IM_ ); }
   __device__ friend cudaComplex operator/(float n2, cudaComplex const & z) 
   { 
      T denom = z.RE_*z.RE_ + z.IM_*z.IM_;
      return cudaComplex( n2*z.RE_ / denom, -n2*z.IM_ / denom);
   }
};

template<typename T> __device__ inline cudaComplex<T> cudaComplex<T>::operator+(cudaComplex<T> const & z2) const { return cudaComplex<T>(RE_+z2.RE_, IM_+z2.IM_); }
template<typename T> __device__ inline cudaComplex<T> cudaComplex<T>::operator-(cudaComplex<T> const & z2) const { return cudaComplex<T>(RE_-z2.RE_, IM_-z2.IM_); }
template<typename T> __device__ inline cudaComplex<T> cudaComplex<T>::operator*(cudaComplex<T> const & z2) const { return cudaComplex<T>(RE_*z2.RE_ - IM_*z2.IM_, IM_*z2.RE_ + RE_*z2.IM_);}
template<typename T> __device__ inline cudaComplex<T> cudaComplex<T>::operator/(cudaComplex<T> const & z2) const 
{ 
   cudaComplex<T> out;
   T denom = z2.RE_*z2.RE_ + z2.IM_*z2.IM_;

   out.RE_ = (RE_*z2.RE_ + IM_*z2.IM_) / denom;
   out.IM_ = (IM_*z2.RE_ - RE_*z2.IM_) / denom;
   return out;
}



template<typename T> __device__ inline cudaComplex<T> cudaComplex<T>::zexp(void) const
{
   return cudaComplex<T>( exp(RE_)*cos(IM_),
                          exp(RE_)*sin(IM_) );
}

template<typename T> __device__ inline cudaComplex<T> cudaComplex<T>::zlog(void) const
{
   return cudaComplex<T>( log(abs(RE_)),
                          atan2(IM_, RE_) );
}

template<typename T> __device__ inline cudaComplex<T> cudaComplex<T>::zsin(void) const
{
   return cudaComplex<T>( sin(RE_)*cosh(IM_),  
                          cos(RE_)*sinh(IM_) );
}

template<typename T> __device__ inline cudaComplex<T> cudaComplex<T>::zcos(void) const
{
   return cudaComplex<T>( cos(RE_)*cosh(IM_),     
                          sin(RE_)*sinh(IM_) );
}

template<typename T> __device__ inline cudaComplex<T> cudaComplex<T>::ztan(void) const
{
   return zsin()/zcos();
}

template<typename T> __device__ inline cudaComplex<T> cudaComplex<T>::zpow(cudaComplex const & c) const
{
   cudaComplex out = c*zlog();
   return out.zexp();
}

// There is a more straightforward way to implement zpow(c) when c is real,
// but I don't expect to use this function often/ever, so "slow" is fine
template<typename T> __device__ inline cudaComplex<T> cudaComplex<T>::zpow(T const & c) const
{
   cudaComplex out = c*zlog();
   return out.zexp();
}

/*

template<typename T>
class Quaternion
{
private:
   QR R;
   QI I;
   QJ J;
   QK K;

public:
   Quaternion(void)                 : R(0), I(0), J(0), K(0) {}
   Quaternion(T r)                  : R(r), I(0), J(0), K(0) {}
   Quaternion(T r, T i, T j, T k)   : R(r), I(i), J(j), K(k) {}





private:
   // *********************************************************************** //
   // Base class for all four dimensions of a quaternion
   // *********************************************************************** //
   class QuaternionDim
   {
   public:
      T value;

      QuaternionDim(void)  : value(0)   {}
      QuaternionDim(T val) : value(val) {}
      
      QuaternionDim  operator+(QuaternionDim const & q2) { return QuaternionDim( value+q2.value ); }
      QuaternionDim  operator-(QuaternionDim const & q2) { return QuaternionDim( value-q2.value ); }

      T const & operator()  {return  value;}     
      T       & operator()  {return  value;}     
      T         operator-() {return -value;}     
   };



      // *********************************************************************** //
      // The REAL part of a quaternion
      // *********************************************************************** //
      class QR : public QuaternionDim
      {
      public:
         // Operators with self
         QR          operator*(QR const & qr) { return QR( value*qr.value ); }
         QR          operator/(QR const & qr) { return QR( value/qr.value ); }
   
         // Operators with I
         Quaternion  operator+(QI const & qi) { return Quaternion(value,  qi.value, 0, 0); }
         Quaternion  operator-(QI const & qi) { return Quaternion(value, -qi.value, 0, 0); }
         QI          operator*(QI const & qi) { return QI(  value * qi.value ); }
         QI          operator/(QI const & qi) { return QI( -value * qi.value ); }

         // Operators with J
         Quaternion  operator+(QJ const & qj) { return Quaternion(value, 0,  qj.value, 0); }
         Quaternion  operator-(QJ const & qj) { return Quaternion(value, 0, -qj.value, 0); }
         QJ          operator*(QJ const & qj) { return QJ(  value * qj.value ); }
         QJ          operator/(QJ const & qj) { return QJ( -value * qj.value ); }

         // Operators with K
         Quaternion  operator+(QK const & qk) { return Quaternion(value, 0, 0,  qk.value); }
         Quaternion  operator-(QK const & qi) { return Quaternion(value, 0, 0, -qk.value); }
         QK          operator*(QK const & qk) { return QK(  value * qk.value ); }
         QK          operator/(QK const & qk) { return QK( -value * qk.value ); }
      };
   
      // *********************************************************************** //
      // Imaginary i  :  i*i = -1;  i*j=k; j*i=-k; i*k=-j; k*i=j;
      // *********************************************************************** //
      class QI : public QuaternionDim
      {
         // Operators with Reals
         Quaternion  operator+(QR const & qr) { return Quaternion( qr.value, value, 0, 0); }
         Quaternion  operator-(QR const & qr) { return Quaternion(-qr.value, value, 0, 0); }
         QI          operator*(QR const & qr) { return QI( value*qr.value ); }
         QI          operator/(QR const & qr) { return QI( value/qr.value ); }
   
         // Operators with self
         QR          operator*(QI const & qi) { return QR( -value*qi.value ); }
         QI          operator/(QI const & qi) { return QR(  value*qi.value ); }

         // Operators with J
         Quaternion  operator+(QJ const & qj) { return Quaternion(0, value,  qj.value, 0); }
         Quaternion  operator-(QJ const & qj) { return Quaternion(0, value, -qj.value, 0); }
         QK          operator*(QJ const & qj) { return QK(  value * qj.value ); }
         QK          operator/(QJ const & qj) { return QK( -value * qj.value ); }

         // Operators with K
         Quaternion  operator+(QK const & qk) { return Quaternion(0, value, 0,  qk.value); }
         Quaternion  operator-(QK const & qi) { return Quaternion(0, value, 0, -qk.value); }
         QJ          operator*(QK const & qk) { return QJ( -value * qk.value ); }
         QJ          operator/(QK const & qk) { return QJ(  value * qk.value ); }
      
      };
   
      // *********************************************************************** //
      // Imaginary j  :  j*j = -1;  j*i=-k; i*j=k; j*k=i; k*j=-i;
      // *********************************************************************** //
      class QJ : public QuaternionDim
      {
         // Operators with Reals
         Quaternion  operator+(QR const & qr) { return Quaternion( qr.value, 0, value, 0); }
         Quaternion  operator-(QR const & qr) { return Quaternion(-qr.value, 0, value, 0); }
         QJ          operator*(QR const & qr) { return QJ( value*qr.value ); }
         QJ          operator/(QR const & qr) { return QJ( value/qr.value ); }
   
         // Operators with I
         Quaternion  operator+(QI const & qi) { return Quaternion(0, qi.value, value, 0); }
         Quaternion  operator-(QI const & qi) { return Quaternion(0,-qi.value, value, 0); }
         QK          operator*(QI const & qi) { return QK( -value*qi.value ); }
         QK          operator/(QI const & qi) { return QK(  value*qi.value ); }

         // Operators with self
         QR          operator*(QJ const & qj) { return QR( -value * qj.value ); }
         QR          operator/(QJ const & qj) { return QR(  value * qj.value ); }

         // Operators with K
         Quaternion  operator+(QK const & qk) { return Quaternion(0, 0, value,  qk.value); }
         Quaternion  operator-(QK const & qk) { return Quaternion(0, 0, value, -qk.value); }
         QI          operator*(QK const & qk) { return QI(  value * qk.value ); }
         QI          operator/(QK const & qk) { return QI( -value * qk.value ); }
      
      };
   
      // *********************************************************************** //
      // Imaginary k  :  k*k = -1;  k*i=j; i*k=-j; k*j=-i; j*k=i;
      // *********************************************************************** //
      class QK : public QuaternionDim
      {
         // Operators with Reals
         Quaternion  operator+(QR const & qr) { return Quaternion( qr.value, 0, 0, value); }
         Quaternion  operator-(QR const & qr) { return Quaternion(-qr.value, 0, 0, value); }
         QK          operator*(QR const & qr) { return QK( value*qr.value ); }
         QK          operator/(QR const & qr) { return QK( value/qr.value ); }
   
         // Operators with I
         Quaternion  operator+(QI const & qi) { return Quaternion(0, qi.value, 0, value); }
         Quaternion  operator-(QI const & qi) { return Quaternion(0,-qi.value, 0, value); }
         QJ          operator*(QI const & qi) { return QJ(  value*qi.value ); }
         QJ          operator/(QI const & qi) { return QJ( -value*qi.value ); }

         // Operators with J
         Quaternion  operator+(QJ const & qj) { return Quaternion(0, 0, qj.value,  value); }
         Quaternion  operator-(QJ const & qj) { return Quaternion(0, 0, qj.value, -value); }
         QI          operator*(QJ const & qj) { return QI( -value * qj.value ); }
         QI          operator/(QJ const & qj) { return QI(  value * qj.value ); }

         // Operators with Self
         QR          operator*(QK const & qk) { return QR(  value * qk.value ); }
         QR          operator/(QK const & qk) { return QR( -value * qk.value ); }
      
      };

};
*/

template<typename T>
class cudaQuaternion
{
private:

   T R_;
   T I_;
   T J_;
   T K_;

public:

   __device__ cudaQuaternion()                   : R_(0), I_(0), J_(0), K_(0) {}
   __device__ cudaQuaternion(T r)                : R_(r), I_(0), J_(0), K_(0) {}
   __device__ cudaQuaternion(T r, T i)           : R_(r), I_(i), J_(0), K_(0) {}
   __device__ cudaQuaternion(T r, T i, T j, T k) : R_(r), I_(i), J_(j), K_(k) {}

   __device__ T const & R(void) const { return R_; }
   __device__ T const & I(void) const { return I_; }
   __device__ T const & J(void) const { return J_; }
   __device__ T const & K(void) const { return K_; }

   //__device__ cudaQuaternion sq(void) { return cudaQuaternion( RE*RE - IM*IM, 2*RE*IM ); }
   __device__ T  conj_sq(void) const { return R_*R_ + I_*I_ + J_*J_ + K_*K_; }

   __device__ cudaQuaternion conj() const      { return cudaQuaternion(  R_, -I_, -J_, -K_); }
   __device__ cudaQuaternion operator-() const { return cudaQuaternion( -R_, -I_, -J_, -K_); }

   __device__ cudaQuaternion operator+(cudaQuaternion const & q2) const;
   __device__ cudaQuaternion operator-(cudaQuaternion const & q2) const;
   __device__ cudaQuaternion operator*(cudaQuaternion const & q2) const;
   __device__ cudaQuaternion operator/(cudaQuaternion const & q2) const;

   // Not only do I have no idea how to do these ops with quaternions, I don't need to
   // Define zfunctions to be complex versions of regular real-valued functions
   //__device__ cudaQuaternion zexp(void) const;
   //__device__ cudaQuaternion zlog(void) const;
   //__device__ cudaQuaternion zsin(void) const;
   //__device__ cudaQuaternion zcos(void) const;
   //__device__ cudaQuaternion ztan(void) const;
   //__device__ cudaQuaternion zpow(cudaQuaternion const & c) const;
   //__device__ cudaQuaternion zpow(T const & c) const;

   //template<typename T2> __device__ cudaQuaternion operator+(T2 const & n2) const { return cudaQuaternion( RE + n2, IM      ); }
   //template<typename T2> __device__ cudaQuaternion operator-(T2 const & n2) const { return cudaQuaternion( RE - n2, IM      ); }
   //template<typename T2> __device__ cudaQuaternion operator*(T2 const & n2) const { return cudaQuaternion( RE * n2, IM * n2 ); }
   //template<typename T2> __device__ cudaQuaternion operator/(T2 const & n2) const { return cudaQuaternion( RE / n2, IM / n2 ); }


   // TODO:  Add versions of these functions that take references/pointers for
   //        the output variable, so that variables can be created once in the
   //        calling code, and no extra copying is going on (we lose 5-10% of
   //        our computation speed due to these extra copies

   //template<typename T2> friend cudaQuaternion operator+(T const & n2, cudaQuaternion const & z);
   //template<typename T2> friend cudaQuaternion operator-(T const & n2, cudaQuaternion const & z);
   //template<typename T2> friend cudaQuaternion operator*(T const & n2, cudaQuaternion const & z);
   //template<typename T2> friend cudaQuaternion operator/(T const & n2, cudaQuaternion const & z);
};

template<typename T> __device__ inline cudaQuaternion<T> cudaQuaternion<T>::operator+(cudaQuaternion<T> const & q2) const 
{ 
   return cudaQuaternion<T>( R_ + q2.R_,
                             I_ + q2.I_,
                             J_ + q2.J_,
                             K_ + q2.K_);
}
template<typename T> __device__ inline cudaQuaternion<T> cudaQuaternion<T>::operator-(cudaQuaternion<T> const & q2) const 
{ 
   return cudaQuaternion<T>( R_ - q2.R_,
                             I_ - q2.I_,
                             J_ - q2.J_,
                             K_ - q2.K_);
}
template<typename T> __device__ inline cudaQuaternion<T> cudaQuaternion<T>::operator*(cudaQuaternion<T> const & q2) const 
{ 
   return cudaQuaternion<T>(
               (R_ * q2.R_) - (I_ * q2.I_) - (J_ * q2.J_) - (K_ * q2.K_),
               (R_ * q2.I_) + (I_ * q2.R_) + (J_ * q2.K_) - (K_ * q2.J_),
               (R_ * q2.J_) - (I_ * q2.K_) + (J_ * q2.R_) + (K_ * q2.I_),
               (R_ * q2.K_) + (I_ * q2.J_) - (J_ * q2.I_) + (K_ * q2.R_));

}
template<typename T> __device__ inline cudaQuaternion<T> cudaQuaternion<T>::operator/(cudaQuaternion<T> const & q2) const 
{ 
   T denom = q2.conj_sq();

   cudaQuaternion<T> numer = (*this) * q2.conj();
   
   return cudaQuaternion<T>( numer.R_ / denom,
                             numer.I_ / denom,
                             numer.J_ / denom,
                             numer.K_ / denom);
}

/*
//template<typename T, typename T2> inline cudaQuaternion<T> cudaQuaternion<T>::operator+(T2 const & n2, cudaQuaternion<T> const & z) { return cudaQuaternion<T>(n2+z.RE,  z.IM); }
//template<typename T, typename T2> inline cudaQuaternion<T> cudaQuaternion<T>::operator-(T2 const & n2, cudaQuaternion<T> const & z) { return cudaQuaternion<T>(n2-z.RE, -z.IM); }
//template<typename T, typename T2> inline cudaQuaternion<T> cudaQuaternion<T>::operator*(T2 const & n2, cudaQuaternion<T> const & z) { return cudaQuaternion<T>(n2*z.RE, n2*z.IM);}
//template<typename T, typename T2> inline cudaQuaternion<T> cudaQuaternion<T>::operator/(T2 const & n2, cudaQuaternion<T> const & z) { return n2*(z.conj()) / z.conj_sq() ; }


template<typename T> __device__ inline cudaQuaternion<T> cudaQuaternion<T>::zexp(void) const
{
   return cudaQuaternion<T>( exp(RE)*cos(IM),
                          exp(RE)*sin(IM) );
}

template<typename T> __device__ inline cudaQuaternion<T> cudaQuaternion<T>::zlog(void) const
{
   return cudaQuaternion<T>( log(abs(RE)),
                          atan2(IM, RE) );
}

template<typename T> __device__ inline cudaQuaternion<T> cudaQuaternion<T>::zsin(void) const
{
   return cudaQuaternion<T>( sin(RE)*cosh(IM),  
                          cos(RE)*sinh(IM) );
}

template<typename T> __device__ inline cudaQuaternion<T> cudaQuaternion<T>::zcos(void) const
{
   return cudaQuaternion<T>( cos(RE)*cosh(IM),     
                          sin(RE)*sinh(IM) );
}

template<typename T> __device__ inline cudaQuaternion<T> cudaQuaternion<T>::ztan(void) const
{
   return zsin()/zcos();
}

template<typename T> __device__ inline cudaQuaternion<T> cudaQuaternion<T>::zpow(cudaQuaternion const & c) const
{
   cudaQuaternion out = c*zlog();
   return out.zexp();
}

// There is a more straightforward way to implement zpow(c) when c is real,
// but I don't expect to use this function often/ever, so "slow" is fine
template<typename T> __device__ inline cudaQuaternion<T> cudaQuaternion<T>::zpow(T const & c) const
{
   cudaQuaternion out = c*zlog();
   return out.zexp();
}
*/


#endif
