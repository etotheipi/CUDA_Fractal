#ifndef _COMPLEX_NUMBER_H_
#define _COMPLEX_NUMBER_H_


#include <iostream>

// ************************************************************************** //
// Basic complex numbers implemented in the simplest, most efficient
// ************************************************************************** //

template<typename T>
class Complex
{
private:

   T RE;
   T IM;

public:

   Complex() : RE(0), IM(0) {}
   Complex(T re) : RE(re), IM(0) {}
   Complex(T re, T im) : RE(re), IM(im) {}

   T const & real(void) const { return RE; }
   T const & imag(void) const { return IM; }
   T & real(void) { return RE; }
   T & imag(void) { return IM; }

   Complex sq(void) { return Complex( RE*RE - IM*IM, 2*RE*IM ); }
   T  conj_sq(void) { return RE*RE + IM*IM; }

   //Complex operator!() const { return Complex( RE, -IM ); }
   Complex conj() const { return Complex( RE, -IM ); }
   Complex operator-() const { return Complex(-RE, -IM ); }

   Complex operator+(Complex const & z2) const;
   Complex operator-(Complex const & z2) const;
   Complex operator*(Complex const & z2) const;
   Complex operator/(Complex const & z2) const;

   template<typename T2> Complex operator+(T2 const & n2) const { return Complex( RE + n2, IM      ); }
   template<typename T2> Complex operator-(T2 const & n2) const { return Complex( RE - n2, IM      ); }
   template<typename T2> Complex operator*(T2 const & n2) const { return Complex( RE * n2, IM * n2 ); }
   template<typename T2> Complex operator/(T2 const & n2) const { return Complex( RE / n2, IM / n2 ); }

   //template<typename T2> friend Complex operator+(T const & n2, Complex const & z);
   //template<typename T2> friend Complex operator-(T const & n2, Complex const & z);
   //template<typename T2> friend Complex operator*(T const & n2, Complex const & z);
   //template<typename T2> friend Complex operator/(T const & n2, Complex const & z);

   template<typename DTYPE> friend std::ostream& operator<<(std::ostream & os, Complex<DTYPE> const & z);

};

template<typename T> inline Complex<T> Complex<T>::operator+(Complex<T> const & z2) const { return Complex<T>(RE+z2.RE, IM+z2.IM); }
template<typename T> inline Complex<T> Complex<T>::operator-(Complex<T> const & z2) const { return Complex<T>(RE-z2.RE, IM-z2.IM); }
template<typename T> inline Complex<T> Complex<T>::operator*(Complex<T> const & z2) const { return Complex<T>(RE*z2.RE - IM*z2.IM, IM*z2.RE + RE*z2.IM);}
template<typename T> inline Complex<T> Complex<T>::operator/(Complex<T> const & z2) const 
{ 
   Complex<T> out;
   T denom = z2.RE*z2.RE + z2.IM*z2.IM;

   out.RE = (RE*z2.RE + IM*z2.IM) / denom;
   out.IM = (IM*z2.RE - RE*z2.IM) / denom;
   return out;
}

//template<typename T, typename T2> inline Complex<T> Complex<T>::operator+(T2 const & n2, Complex<T> const & z) { return Complex<T>(n2+z.RE,  z.IM); }
//template<typename T, typename T2> inline Complex<T> Complex<T>::operator-(T2 const & n2, Complex<T> const & z) { return Complex<T>(n2-z.RE, -z.IM); }
//template<typename T, typename T2> inline Complex<T> Complex<T>::operator*(T2 const & n2, Complex<T> const & z) { return Complex<T>(n2*z.RE, n2*z.IM);}
//template<typename T, typename T2> inline Complex<T> Complex<T>::operator/(T2 const & n2, Complex<T> const & z) { return n2*(z.conj()) / z.conj_sq() ; }

template<typename T> inline ostream& operator<<(std::ostream & os, Complex<T> const & z)
{
   T re = z.real();
   T im = z.imag();
   os << re;
   if(im >= 0)
      os << " + " <<  im << "(i)";
   else
      os << " - " << -im << "(i)";

   return os;
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

#endif
