//********************************************************************************
//
// Library of functions for calculations involving dipoles and optical forces.
//
// clang -std=c++14 -Wall -Wextra -pedantic -c -fPIC Dipoles.cpp -o Dipoles.o
// clang -shared Dipoles.o -o Dipoles.dylib
//
//********************************************************************************
#include "Dipoles.hpp"

//********************************************************************************
//  Start with calculation of the gradient of radiative dipole field as given
//  in Jackson, 3rd edition.
//********************************************************************************

void grad_E_cc(double *rvec, double *pvec, double kvec, double *gradEE)
{
    // Computes gradient of complex conjugate of electric field from the
    // scattered dipoles:
    // rvec: 3-vector describing position of dipole relative to point of interest;
    // pvec: complex 3-vector describing dipole moment;
    // kvec: scalar, acting value of wave vector, should be corrected for medium;
    // gradEE: for passing back complex gradients, in (real,imag) pairs.
    // Based on my hand-written notes, taking formula from Jackson, 3rd Edition,
    // page 411.

    using namespace std::complex_literals;
    
    //
    // First set up working matrices, A, B, C and gradients
    // and a few temporary variables
    //

    std::complex<double> dAdx[3][3], dAdy[3][3], dAdz[3][3];
    double B[3][3], dBdx[3][3], dBdy[3][3], dBdz[3][3];
    double C[3][3], dCdx[3][3], dCdy[3][3], dCdz[3][3];
    double r, r2, r3, r5, r7, k2;
    double x, y, z, kappa, rii, rij;
    std::complex<double> s, ikr;
    std::complex<double> F, dFd, dFdx, dFdy, dFdz;
    std::complex<double> G, dGd, dGdx, dGdy, dGdz;
    std::complex<double> pvecbar[3], Nabla_E_cc[3][3];

    x = rvec[0];
    y = rvec[1];
    z = rvec[2];

    r2 = x*x + y*y + z*z;
    r = sqrt(r2);
    r3 = r*r2;
    r5 = r3*r2;
    r7 = r5*r2;
    k2 = kvec*kvec;
    
    ikr = 1i * kvec * r;
    s = exp(-ikr);
    kappa = 1.0 / (4.0 * M_PI * EPS0);

    // Finding gradient of F coefficient
    F = k2 * s / r3;
    dFd = -F * (ikr + 3.0) / r2;
    dFdx = dFd * x;
    dFdy = dFd * y;
    dFdz = dFd * z;
    
    // Finding gradient of G coefficient
    G = -s * (ikr + 1.0) / r5;
    dGd = s * (5.0 * (ikr + 1.0) - k2 * r2) / r7;
    dGdx = dGd * x;
    dGdy = dGd * y;
    dGdz = dGd * z;

    // Generate B and C Matrices
    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            if (i == j){
                rii = rvec[i] * rvec[i];
                B[i][i] = rii - r2;
                C[i][i] = 3 * rii - r2;
            }
            else{
                rij = rvec[i] * rvec[j];
                B[i][j] = rij;
                C[i][j] = 3 * rij;
            }

    // Derivatives of B and C matrices
    dBdx[0][0] = 0.0;
    dBdx[0][1] = rvec[1];
    dBdx[0][2] = rvec[2];
    dBdx[1][0] = rvec[1];
    dBdx[1][1] = -2 * rvec[0];
    dBdx[1][2] = 0.0;
    dBdx[2][0] = rvec[2];
    dBdx[2][1] = 0.0;
    dBdx[2][2] = -2 * rvec[0];

    dBdy[0][0] = -2 * rvec[1];
    dBdy[0][1] = rvec[0];
    dBdy[0][2] = 0.0;
    dBdy[1][0] = rvec[0];
    dBdy[1][1] = 0.0;
    dBdy[1][2] = rvec[2];
    dBdy[2][0] = 0.0;
    dBdy[2][1] = rvec[2];
    dBdy[2][2] = -2 * rvec[1];

    dBdz[0][0] = -2 * rvec[2];
    dBdz[0][1] = 0.0;
    dBdz[0][2] = rvec[0];
    dBdz[1][0] = 0.0;
    dBdz[1][1] = -2 * rvec[2];
    dBdz[1][2] = rvec[1];
    dBdz[2][0] = rvec[0];
    dBdz[2][1] = rvec[1];
    dBdz[2][2] = 0.0;

    dCdx[0][0] = 4 * rvec[0];
    dCdx[0][1] = 3 * rvec[1];
    dCdx[0][2] = 3 * rvec[2];
    dCdx[1][0] = 3 * rvec[1];
    dCdx[1][1] = -2 * rvec[0];
    dCdx[1][2] = 0.0;
    dCdx[2][0] = 3 * rvec[2];
    dCdx[2][1] = 0.0;
    dCdx[2][2] = -2 * rvec[0];

    dCdy[0][0] = -2 * rvec[1];
    dCdy[0][1] = 3 * rvec[0];
    dCdy[0][2] = 0.0;
    dCdy[1][0] = 3 * rvec[0];
    dCdy[1][1] = 4 * rvec[1];
    dCdy[1][2] = 3 * rvec[2];
    dCdy[2][0] = 0.0;
    dCdy[2][1] = 3 * rvec[2];
    dCdy[2][2] = -2 * rvec[1];

    dCdz[0][0] = -2 * rvec[2];
    dCdz[0][1] = 0.0;
    dCdz[0][2] = 3 * rvec[0];
    dCdz[1][0] = 0.0;
    dCdz[1][1] = -2 * rvec[2];
    dCdz[1][2] = 3 * rvec[1];
    dCdz[2][0] = 3 * rvec[0];
    dCdz[2][1] = 3 * rvec[1];
    dCdz[2][2] = 4 * rvec[2];
    
    // Gradient of A - combining lots of matrices
    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++){
            dAdx[i][j] = (dFdx * B[i][j] + F * dBdx[i][j] + dGdx * C[i][j] + G * dCdx[i][j]) * kappa;
            dAdy[i][j] = (dFdy * B[i][j] + F * dBdy[i][j] + dGdy * C[i][j] + G * dCdy[i][j]) * kappa;
            dAdz[i][j] = (dFdz * B[i][j] + F * dBdz[i][j] + dGdz * C[i][j] + G * dCdz[i][j]) * kappa;
        }

    // Dot product of grad(A) and P* gives grad(E)
    for (int i=0; i<3; i++){
        pvecbar[i] = pvec[2*i]-pvec[2*i+1]*1i; // Conjugate
        for (int j=0; j<3; j++)
            Nabla_E_cc[i][j] = 0.0;
    }
    
    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++){
            Nabla_E_cc[0][i] += dAdx[i][j]*pvecbar[j];
            Nabla_E_cc[1][i] += dAdy[i][j]*pvecbar[j];
            Nabla_E_cc[2][i] += dAdz[i][j]*pvecbar[j];
        }
    
    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++){
            gradEE[2*(i*3+j)] = real(Nabla_E_cc[i][j]);
            gradEE[2*(i*3+j)+1] = imag(Nabla_E_cc[i][j]);
    }

    return;
}




//********************************************************************************
// Utility functions
//********************************************************************************
/*
double jnp(int order, double x){
    //
    // Function to provide first derivative of nth order Bessel function
    // using J'_n = (J_n-1 - J_n+1) / 2
    //
    double temp;
    temp = 0.5*(jn(order-1,x)-jn(order+1,x));
    return temp;
}
*/
