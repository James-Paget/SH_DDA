//********************************************************************************
//
// Library of beam types, defined parallel to z-axis.  Currently supports:
// 0. plane wave
// 1. gaussian beam - Barton 5
// 2. gaussian beam - complex point source
// 3. General order Bessel beam after Volke-Sepulveda
// 4. General Laguerre-Gaussian after Barnett and Allen
//
//********************************************************************************
// Version 1 August 2021 in Python, written by Tom Morling, plane waves and
//   Bessel beams, zero order and general order, and Barton 5 Gaussian.
// Version 2 August 2022 in C++, adapted by SH, plane waves and general order
//   Bessel beams, with Barton 5 Gaussian.
// Version 3 December 2022/January 2023 including working Gauss CSP beam.
// Version 4 February 2024, introducing Barnett's general Laguerre-Gaussian beam.
//********************************************************************************
// Note: This version for Apple Clang++ uses abs() calls.
//       Change to fabs() for gcc and linux.
//********************************************************************************
#include <complex>
#include <cmath>
#include <iostream>
#include <omp.h>
#include "Beams.hpp"

//********************************************************************************
//  Start with general interface that takes coordinates, performs transformations
//  and calls the appropriate beam functions.
//********************************************************************************

void compute_fields(double xx, double yy, double zz, BEAM *thisbeam, double *EE)
{
    // xx, yy, zz coordinates of point (double)
    // *thisbeam is a structure with ALL beam information
    // EE is array to hold and pass back calculated fields (complex)
    using namespace std::complex_literals;
    
    std::complex<double> Exyz[3],Epxyz[3];
    double x, y, z;
    //
    // Transform coordinates to beam frame
    // Rotation first
    //
    x = thisbeam->rotation[0]*xx + thisbeam->rotation[1]*yy + thisbeam->rotation[2]*zz;
    y = thisbeam->rotation[3]*xx + thisbeam->rotation[4]*yy + thisbeam->rotation[5]*zz;
    z = thisbeam->rotation[6]*xx + thisbeam->rotation[7]*yy + thisbeam->rotation[8]*zz;
    //
    // Translation part
    //
    x -= thisbeam->translation[0];
    y -= thisbeam->translation[1];
    z -= thisbeam->translation[2];

    //
    // Check beam types and
    // compute field components
    //
    switch (thisbeam->beamtype) {
        case BEAMTYPE_PLANE:
            //std::cout << "plane wave" << std::endl;
            plane_wave_fields(x, y, z, thisbeam, Epxyz);
            break;
            
        case BEAMTYPE_GAUSS_BARTON5:
            //std::cout << "gauss barton 5" << std::endl;
            gaussian_barton5_fields(x, y, z, thisbeam, Epxyz);
            break;
            
        case BEAMTYPE_GAUSS_CSP:
            //std::cout << "gauss complex source point" << std::endl;
            gaussian_csp_fields(x, y, z, thisbeam, Epxyz);
            break;
            
        case BEAMTYPE_BESSEL:
            //std::cout << "general bessel beam" << std::endl;
            general_bessel_fields(x, y, z, thisbeam, Epxyz);
            break;
            
        case BEAMTYPE_LAGUERRE_GAUSSIAN:
            //std::cout << "general LG beam" << std::endl;
            laguerre_gaussian_fields(x, y, z, thisbeam, Epxyz);
            break;
            
        default:
            break;
    }
    //
    // Back transform fields to lab frame
    //
    Exyz[0] = thisbeam->rotation[0]*Epxyz[0] + thisbeam->rotation[3]*Epxyz[1] + thisbeam->rotation[6]*Epxyz[2];
    Exyz[1] = thisbeam->rotation[1]*Epxyz[0] + thisbeam->rotation[4]*Epxyz[1] + thisbeam->rotation[7]*Epxyz[2];
    Exyz[2] = thisbeam->rotation[2]*Epxyz[0] + thisbeam->rotation[5]*Epxyz[1] + thisbeam->rotation[8]*Epxyz[2];
    
    EE[0] = real(Exyz[0]);
    EE[1] = imag(Exyz[0]);
    EE[2] = real(Exyz[1]);
    EE[3] = imag(Exyz[1]);
    EE[4] = real(Exyz[2]);
    EE[5] = imag(Exyz[2]);
    return;
}

void compute_fields_array(double *xx, double *yy, double *zz, size_t numpoints, BEAM *thisbeam, double *EE, size_t rows, size_t columns )
{
    //std::cout << numpoints <<" "<< rows <<" "<< (int)columns<<std::endl;
    // xx, yy, zz coordinates of point (double)
    // *thisbeam is a structure with ALL beam information
    // EE is array to hold and pass back calculated fields (complex)
    using namespace std::complex_literals;
    
    //double x[numpoints], y[numpoints], z[numpoints];
    size_t i;
    omp_set_num_threads(1);
#pragma omp parallel for private(i)
    for (i=0; i<numpoints; i++){
        double x, y, z;
        std::complex<double> Exyz[3],Epxyz[3];
        //
        // Transform coordinates to beam frame
        // Rotation first
        //
        x = thisbeam->rotation[0]*xx[i] + thisbeam->rotation[1]*yy[i] + thisbeam->rotation[2]*zz[i];
        y = thisbeam->rotation[3]*xx[i] + thisbeam->rotation[4]*yy[i] + thisbeam->rotation[5]*zz[i];
        z = thisbeam->rotation[6]*xx[i] + thisbeam->rotation[7]*yy[i] + thisbeam->rotation[8]*zz[i];
        //
        // Translation part
        //
        x -= thisbeam->translation[0];
        y -= thisbeam->translation[1];
        z -= thisbeam->translation[2];
    //
    // Check beam types and
    // compute field components
    //
    switch (thisbeam->beamtype) {
        case BEAMTYPE_PLANE:
            //std::cout << "plane wave" << std::endl;
            plane_wave_fields(x, y, z, thisbeam, Epxyz);
            break;
            
        case BEAMTYPE_GAUSS_BARTON5:
            //std::cout << "gauss barton 5" << std::endl;
                gaussian_barton5_fields(x, y, z, thisbeam, Epxyz);
            break;

        case BEAMTYPE_GAUSS_CSP:
            //std::cout << "gauss complex source point" << std::endl;
                gaussian_csp_fields(x, y, z, thisbeam, Epxyz);
            break;

        case BEAMTYPE_BESSEL:
            //std::cout << "general bessel beam" << std::endl;
                general_bessel_fields(x, y, z, thisbeam, Epxyz);
            break;

        case BEAMTYPE_LAGUERRE_GAUSSIAN:
            //std::cout << "general LG beam" << std::endl;
                laguerre_gaussian_fields(x, y, z, thisbeam, Epxyz);
            break;

        default:
            break;
    }
    //
    // Back transform fields to lab frame
    //
        Exyz[0] = thisbeam->rotation[0]*Epxyz[0] + thisbeam->rotation[3]*Epxyz[1] + thisbeam->rotation[6]*Epxyz[2];
        Exyz[1] = thisbeam->rotation[1]*Epxyz[0] + thisbeam->rotation[4]*Epxyz[1] + thisbeam->rotation[7]*Epxyz[2];
        Exyz[2] = thisbeam->rotation[2]*Epxyz[0] + thisbeam->rotation[5]*Epxyz[1] + thisbeam->rotation[8]*Epxyz[2];
        
        EE[i*6+0] = real(Exyz[0]);
        EE[i*6+1] = imag(Exyz[0]);
        EE[i*6+2] = real(Exyz[1]);
        EE[i*6+3] = imag(Exyz[1]);
        EE[i*6+4] = real(Exyz[2]);
        EE[i*6+5] = imag(Exyz[2]);
    }
    return;
}



void compute_field_gradients(double xx, double yy, double zz, BEAM *thisbeam, double *gradEE)
{
    // xx, yy, zz coordinates of point (double)
    // *thisbeam is a structure with ALL beam information
    // gradEE is array to hold and pass back calculated fields (complex)
    using namespace std::complex_literals;
    
    std::complex<double> gradExyz[9],gradEpxyz[3][3]; // This will be a problem if we go parallel
    double x, y, z, R[3][3];
    //
    // Transform coordinates to beam frame
    // Rotation first
    //
    x = thisbeam->rotation[0]*xx + thisbeam->rotation[1]*yy + thisbeam->rotation[2]*zz;
    y = thisbeam->rotation[3]*xx + thisbeam->rotation[4]*yy + thisbeam->rotation[5]*zz;
    z = thisbeam->rotation[6]*xx + thisbeam->rotation[7]*yy + thisbeam->rotation[8]*zz;
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            R[i][j] = thisbeam->rotation[i*3+j];
    //
    // Translation part
    //
    x -= thisbeam->translation[0];
    y -= thisbeam->translation[1];
    z -= thisbeam->translation[2];
    //
    // Check beam types and
    // compute field derivatives
    //
    switch (thisbeam->beamtype) {
        case BEAMTYPE_PLANE:
            //std::cout << "plane wave" << std::endl;
            plane_wave_field_gradients(x, y, z, thisbeam, gradEpxyz);
            break;
            
        case BEAMTYPE_GAUSS_BARTON5:
            //std::cout << "gauss barton 5" << std::endl;
            gaussian_barton5_field_gradients(x, y, z, thisbeam, gradEpxyz);
            break;
            
        case BEAMTYPE_GAUSS_CSP:
            //std::cout << "gauss complex source point" << std::endl;
            gaussian_csp_field_gradients(x, y, z, thisbeam, gradEpxyz);
            break;
            
        case BEAMTYPE_BESSEL:
            //std::cout << "general bessel beam" << std::endl;
            general_bessel_field_gradients(x, y, z, thisbeam, gradEpxyz);
            break;
            
        case BEAMTYPE_LAGUERRE_GAUSSIAN:
            //std::cout << "general LG beam" << std::endl;
            laguerre_gaussian_field_gradients_fd(x, y, z, thisbeam, gradEpxyz);
            break;
            
        default:
            break;
    }

    //
    // Back transform gradients to lab frame - transforms as second rank tensor
    //
    
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++){
            gradExyz[i*3+j] = 0.0 + 0.0i;
            for(int m=0; m<3; m++)
                for(int n=0; n<3; n++)
                    gradExyz[i*3+j] += R[m][i]*R[n][j]*gradEpxyz[m][n];
            }
    for(int i=0; i<9; i++){
        gradEE[2*i] = real(gradExyz[i]);
        gradEE[2*i+1] = imag(gradExyz[i]);
    }
    return;
}



//********************************************************************************
//  Laguerre-Gaussian Beam from Barnett and Allen, Optics Communications
//  110 (1994) 670-678.  Beam propagates along z axis.
//********************************************************************************
void laguerre_gaussian_fields(double x, double y, double z, BEAM *thisbeam, std::complex<double> *Epxyz)
{
    // x, y, z coordinates of point
    // E0 field strength
    // k wavevector parallel to beam axis
    // w0 beam waste parameter
    // l order of vortex beam, integer
    // p is the Gouy phase, integer
    // z0 is the Rayleigh range or equivalent
    // the jn functions are Bessel functions of first kind as available in standard C math lib.
    
    using namespace std::complex_literals;
    
    std::complex<double> a, b;
    std::complex<double> prefa, prefb, prefiamb, prefiapb;
    std::complex<double> eilphi, eisqrnomz;
    std::complex<double> Exyacc, Ezacc;
    double rho, phi, kap, kap2, denom, sqrtdenom, invsqrtdenom;
    double kaprho, kk, k2, zR, dkappa, factor, Ekap;
    int l, p, numpoints, i;
    //
    // Compute beam parameters
    //
    a = thisbeam->jones[0]+1i*thisbeam->jones[1];
    b = thisbeam->jones[2]+1i*thisbeam->jones[3];
    l = thisbeam->order;
    p = thisbeam->gouy;
    //index = (2*p+l+1)/2.0;
    //
    // Compute intermediate results
    //
    rho = sqrt(x*x+y*y);
    if ((fabs(x)<TINY) && (fabs(y)<TINY))
        phi=0.0;
    else
        phi = atan2(y,x);
    //
    kk = thisbeam->k;
    k2=kk*kk;
    zR = 0.5*kk*(thisbeam->w0)*(thisbeam->w0);
    eilphi = exp(phi*l*1i);
    prefa = a * eilphi;
    prefb = b * eilphi;
    prefiamb = 0.5 * eilphi * (1i*a-b) * exp(-phi*1i);
    prefiapb = 0.5 * eilphi * (1i*a+b) * exp(phi*1i);
    //
    // Need to integrate, so accumulate all sums in one loop
    // using Simpson's rule for odd number of points where
    // the final point with k = kappa will always be zero and
    // therefore not evaluated.  Therefore numpoints should be EVEN.
    // kappa=0 also gives zero, and can be ignored.
    //
    numpoints = thisbeam->numkpoints;
    //
    // kappa == 0
    //
    Exyacc = 0.0+0.0i;
    Ezacc = 0.0+0.0i;
    dkappa = kk/numpoints;
    for (i=1; i<numpoints; i++) {
        if (i%2==0) {
            factor=2.0*dkappa/3.0;
        } else {
            factor=4.0*dkappa/3.0;
        }
        kap = i*dkappa;
        kap2 = kap*kap;
        denom = k2-kap2;
        //invdenom = 1.0/denom;
        sqrtdenom = sqrt(denom);
        invsqrtdenom = 1.0/sqrtdenom;
        kaprho = kap*rho;
        eisqrnomz = exp(1i*sqrtdenom*z);
        //
        //Ekap = exp(-kzR*kap2*invdenom) * pow(kap2*invdenom,index)*invsqrtdenom;
        Ekap = eek(kap, kk, thisbeam->E0, zR, l, p);

        Exyacc += factor * Ekap * eisqrnomz * jn(l,kaprho);
        Ezacc += factor * Ekap * eisqrnomz * kap * invsqrtdenom * (prefiamb*jn(l-1,kaprho)-prefiapb*jn(l+1,kaprho));
    }
    //
    // Compute field components
    //
    Epxyz[0] = prefa * Exyacc;
    Epxyz[1] = prefb * Exyacc;
    Epxyz[2] = Ezacc;
    return;
}


void laguerre_gaussian_field_gradients_fd(double x, double y, double z, BEAM *thisbeam, std::complex<double> gradEpxyz[][3]){
    // x, y, z coordinates of point
    // E0 field strength
    // k wavevector parallel to beam axis
    // w0 beam waste parameter
    // l order of vortex beam, integer
    // p is the Gouy phase, integer
    // z0 is the Rayleigh range or equivalent
    // the jn functions are Bessel functions of first kind as available in standard C math lib.
    //
    // This function is going to use finite differences to compute the field gradients.
    // Try with 2-point central differences.
    //
    // S.H. 7-Jun-2024
    //
    using namespace std::complex_literals;
    
    std::complex<double> Emxyz[3],Epxyz[3];
    
    double delta = 1e-9; // Step for finite differences
    int i;
    //
    // Compute x derivatives of Ex, Ey, Ez:
    //
    laguerre_gaussian_fields(x+delta, y, z, thisbeam, Epxyz);
    laguerre_gaussian_fields(x-delta, y, z, thisbeam, Emxyz);
    for (i=0; i<3; i++)
        gradEpxyz[i][0] = (Epxyz[i]-Emxyz[i])/(2*delta);
    //
    // Compute y derivatives of Ex, Ey, Ez:
    //
    laguerre_gaussian_fields(x, y+delta, z, thisbeam, Epxyz);
    laguerre_gaussian_fields(x, y-delta, z, thisbeam, Emxyz);
    for (i=0; i<3; i++)
        gradEpxyz[i][1] = (Epxyz[i]-Emxyz[i])/(2*delta);
    //
    // Compute z derivatives of Ex, Ey, Ez:
    //
    laguerre_gaussian_fields(x, y, z+delta, thisbeam, Epxyz);
    laguerre_gaussian_fields(x, y, z-delta, thisbeam, Emxyz);
    for (i=0; i<3; i++)
        gradEpxyz[i][2] = (Epxyz[i]-Emxyz[i])/(2*delta);

    //std::cout<<"x derivs: "<<gradEpxyz[0][0]<<" "<<gradEpxyz[1][0]<<" "<<gradEpxyz[2][0]<<std::endl;
    //std::cout<<"y derivs: "<<gradEpxyz[0][1]<<" "<<gradEpxyz[1][1]<<" "<<gradEpxyz[2][1]<<std::endl;
    //std::cout<<"z derivs: "<<gradEpxyz[0][2]<<" "<<gradEpxyz[1][2]<<" "<<gradEpxyz[2][2]<<std::endl;
    return;
}

/*
void laguerre_gaussian_field_gradients_fd(double x, double y, double z, BEAM *thisbeam, std::complex<double> gradEpxyz[][3]){
    // x, y, z coordinates of point
    // E0 field strength
    // k wavevector parallel to beam axis
    // w0 beam waste parameter
    // l order of vortex beam, integer
    // p is the Gouy phase, integer
    // z0 is the Rayleigh range or equivalent
    // the jn functions are Bessel functions of first kind as available in standard C math lib.
    //
    // This function is going to use finite differences to compute the field gradients.
    // Try with 2-point central differences.
    //
    // S.H. 7-Jun-2024
    //
    using namespace std::complex_literals;
    
    std::complex<double> a, b;
    std::complex<double> prefa, prefb, prefiamb, prefiapb;
    std::complex<double> eilphi, eisqrnomz;
    std::complex<double> Exyacc, Ezacc;
    double rho, phi, kap, kap2, denom, sqrtdenom, invsqrtdenom;
    double kaprho, kk, k2, zR, dkappa, factor, Ekap;
    double Jlm1kaprho, Jlkaprho, Jlp1kaprho, Jlp2kaprho;
    int l, p, numpoints, i, j;
    //
    // Compute beam parameters
    //
    a = thisbeam->jones[0]+1i*thisbeam->jones[1];
    b = thisbeam->jones[2]+1i*thisbeam->jones[3];
    l = thisbeam->order;
    p = thisbeam->gouy;
    //index = (2*p+l+1)/2.0;
    //
    // Compute intermediate results
    //
    rho = sqrt(x*x+y*y);
    if ((fabs(x)<TINY) && (fabs(y)<TINY))
        phi=0.0;
    else
        phi = atan2(y,x);
    //
    kk = thisbeam->k;
    k2=kk*kk;
    zR = 0.5*kk*(thisbeam->w0)*(thisbeam->w0);
    eilphi = exp(phi*l*1i);
    prefa = a * eilphi;
    prefb = b * eilphi;
    prefiamb = 0.5 * eilphi * (1i*a-b) * exp(-phi*1i);
    prefiapb = 0.5 * eilphi * (1i*a+b) * exp(phi*1i);
    //
    // Need to integrate, so accumulate all sums in one loop
    // using Simpson's rule for odd number of points where
    // the final point with k = kappa will always be zero and
    // therefore not evaluated.  Therefore numpoints should be EVEN.
    // kappa=0 also gives zero, and can be ignored.
    //
    numpoints = thisbeam->numpoints;
    //
    // kappa == 0
    //
    Exyacc = 0.0+0.0i;
    Ezacc = 0.0+0.0i;
    dkappa = kk/numpoints;
    
    for (i=0; i<3; i++)
        for (j=0; j<3; j++)
            gradEpxyz[0][0] = 0.0 + 0.0i;
    
    for (i=1; i<numpoints; i++) {
        if (i%2==0) {
            factor=2.0*dkappa/3.0;
        } else {
            factor=4.0*dkappa/3.0;
        }
        kap = i*dkappa;
        kap2 = kap*kap;
        denom = k2-kap2;
        //invdenom = 1.0/denom;
        sqrtdenom = sqrt(denom);
        invsqrtdenom = 1.0/sqrtdenom;
        kaprho = kap*rho;
        eisqrnomz = exp(1i*sqrtdenom*z);
        //
        //Ekap = exp(-kzR*kap2*invdenom) * pow(kap2*invdenom,index)*invsqrtdenom;
        Ekap = eek(kap, kk, thisbeam->E0, zR, l, p);
        //
        // Bessel functions needed
        //
        Jlm1kaprho = jn(l-1,kaprho)
        Jlkaprho = jn(l,kaprho)
        Jlp1kaprho = jn(l+1,kaprho)
        Jlp2kaprho = jn(l+2,kaprho)
        //
        // Compute derivatives:
        // Ex derivatives
        //gradEpxyz[0][0] = pref1*kx*(2.0*Bn[2] + pref3);
        //gradEpxyz[0][1] = pref1*ky*pref3;
        gradEpxyz[0][2] += factor * Ekap * eilphi * 1i * sqrtdenom * eisqrnomz * a * Jlkaprho;
        
        // Ey derivatives
        //gradEpxyz[1][0] = pref1*ky*(Bn[2]-kx2*Bn[3]);
        //gradEpxyz[1][1] = pref1*kx*(Bn[2]-ky2*Bn[3]);
        //gradEpxyz[1][2] = -pref1*kx*ky*kz*Bn[3];
        gradEpxyz[1][2] += factor * Ekap * eilphi * 1i * sqrtdenom * eisqrnomz * b * Jlkaprho;

        // Ez derivatives
        //gradEpxyz[2][0] = pref1*((kz+1i*kx2)*Bn[2] - kx2*kz*Bn[3]) - pref2*(Bn[0]+kR2*Bn[2]);
        //gradEpxyz[2][1] = pref1*kx*ky*(1i*Bn[2]-kz*Bn[3]);
        //gradEpxyz[2][2] = pref1*kx*((1.0+1i*kz)*Bn[2] - kz2*Bn[3]);
        //
    }
    //std::cout<<"x derivs: "<<gradEpxyz[0][0]<<" "<<gradEpxyz[1][0]<<" "<<gradEpxyz[2][0]<<std::endl;
    //std::cout<<"y derivs: "<<gradEpxyz[0][1]<<" "<<gradEpxyz[1][1]<<" "<<gradEpxyz[2][1]<<std::endl;
    //std::cout<<"z derivs: "<<gradEpxyz[0][2]<<" "<<gradEpxyz[1][2]<<" "<<gradEpxyz[2][2]<<std::endl;
    return;
}

*/


//********************************************************************************
//  General Bessel Beam from Volke-Sepulveda et al. J. Opt. B: Quantum Semiclass.
//  Opt. 4 (2002) S82–S89.  Beam propagates along z axis.
//********************************************************************************

/*std::complex<double> *general_bessel(double xx, double yy, double zz, BEAM *thisbeam)
{
    // x, y, z coordinates of point
    // E0 field strength
    // kz, kt components of wavevector parallel and perpendicular to beam
    // a, b components of Jones vector
    // l order of optical vortex
    using namespace std::complex_literals;
    
    std::complex<double> a, b;
    static std::complex<double> Exyz[3],Epxyz[3]; // This will be a problem if we go parallel
    std::complex<double> eikz, eilphi, pref1, pref2;
    double r, phi, ktr;
    double x, y, z;
    int l;
    //
    // Transform coordinates to beam frame
    // Rotation first
    //
    x = thisbeam->rotation[0]*xx + thisbeam->rotation[1]*yy + thisbeam->rotation[2]*zz;
    y = thisbeam->rotation[3]*xx + thisbeam->rotation[4]*yy + thisbeam->rotation[5]*zz;
    z = thisbeam->rotation[6]*xx + thisbeam->rotation[7]*yy + thisbeam->rotation[8]*zz;
    //
    // Translation part
    //
    x -= thisbeam->translation[0];
    y -= thisbeam->translation[1];
    z -= thisbeam->translation[2];

    //
    // Compute beam parameters
    //
    a = thisbeam->jones[0]+1i*thisbeam->jones[1];
    b = thisbeam->jones[2]+1i*thisbeam->jones[3];
    l = thisbeam->order;
    //
    // Compute intermediate results
    //
//    printf("Jones matrix: %f %f %f %f\n",thisbeam->jones[0],thisbeam->jones[1],thisbeam->jones[2],thisbeam->jones[3]);
    r = sqrt(x*x+y*y);
    phi = atan2(y,x);
    eikz = exp(1i*thisbeam->kz*z);
    eilphi = exp(phi*l*1i);
    ktr = thisbeam->kt*r;
    pref1 = thisbeam->E0 * eikz * eilphi;
    pref2 = pref1 * jn(l, ktr);
    //
    // Compute field components
    //
    Epxyz[0] = pref2 * a;
    Epxyz[1] = pref2 * b;
    Epxyz[2] = pref1 * thisbeam->kt_by_kz * 0.5i * ((a + 1i * b) * exp(-1i * phi) * jn(l - 1, ktr)
            - (a - 1i * b) * exp(1i * phi) * jn(l + 1, ktr));
    //
    // Back transform fields to lab frame
    //
    Exyz[0] = thisbeam->rotation[0]*Epxyz[0] + thisbeam->rotation[3]*Epxyz[1] + thisbeam->rotation[6]*Epxyz[2];
    Exyz[1] = thisbeam->rotation[1]*Epxyz[0] + thisbeam->rotation[4]*Epxyz[1] + thisbeam->rotation[7]*Epxyz[2];
    Exyz[2] = thisbeam->rotation[2]*Epxyz[0] + thisbeam->rotation[5]*Epxyz[1] + thisbeam->rotation[8]*Epxyz[2];
    return Exyz;
}*/

void general_bessel_fields(double x, double y, double z, BEAM *thisbeam, std::complex<double> *Epxyz)
{
    // x, y, z coordinates of point
    // E0 field strength
    // kz, kt components of wavevector parallel and perpendicular to beam
    // a, b components of Jones vector
    // l order of optical vortex
    // the jn functions are Bessel functions of first kind as available in standard C math lib.
    
    using namespace std::complex_literals;
    
    std::complex<double> a, b;
    std::complex<double> eikz, eilphi, pref1, pref2;
    double r, phi, ktr;
    int l;
    //
    // Compute beam parameters
    //
    a = thisbeam->jones[0]+1i*thisbeam->jones[1];
    b = thisbeam->jones[2]+1i*thisbeam->jones[3];
    l = thisbeam->order;
    //
    // Compute intermediate results
    //
//    printf("Jones matrix: %f %f %f %f\n",thisbeam->jones[0],thisbeam->jones[1],thisbeam->jones[2],thisbeam->jones[3]);
    r = sqrt(x*x+y*y);
    if ((fabs(x)<TINY) && (fabs(y)<TINY))
        phi=0.0;
    else
        phi = atan2(y,x);
    eikz = exp(1i*thisbeam->kz*z);
    eilphi = exp(phi*l*1i);
    ktr = thisbeam->kt*r;
    pref1 = thisbeam->E0 * eikz * eilphi;
    pref2 = pref1 * jn(l, ktr);
    //
    // Compute field components
    //
    Epxyz[0] = pref2 * a;
    Epxyz[1] = pref2 * b;
    Epxyz[2] = pref1 * thisbeam->kt_by_kz * 0.5i * ((a + 1i * b) * exp(-1i * phi) * jn(l - 1, ktr)
            - (a - 1i * b) * exp(1i * phi) * jn(l + 1, ktr));
    return;
}


/*
std::complex<double> *general_bessel_gradient(double xx, double yy, double zz, BEAM *thisbeam)
{
    // x, y, z coordinates of point
    // E0 field strength
    // kz, kt components of wavevector parallel and perpendicular to beam
    // a, b components of Jones vector
    // l order of optical vortex
    using namespace std::complex_literals;
    
    std::complex<double> a, b, apib, amib, epip, emip;
    static std::complex<double> gradExyz[9],gradEpxyz[3][3]; // This will be a problem if we go parallel
    std::complex<double> eikz, eilphi, pref1, pref3, pref4, pref5, pref6, pref7, pref8;
    //std::complex<double> dExdx, dExdy, dExdz, dEydx, dEydy, dEydz, dEzdx, dEzdy, dEzdz;
    double r, phi, ktr, rsq;
    double jnl, jnpl;
    double x, y, z, R[3][3];
    int l;
    //
    // Transform coordinates to beam frame
    // Rotation first
    //
    x = thisbeam->rotation[0]*xx + thisbeam->rotation[1]*yy + thisbeam->rotation[2]*zz;
    y = thisbeam->rotation[3]*xx + thisbeam->rotation[4]*yy + thisbeam->rotation[5]*zz;
    z = thisbeam->rotation[6]*xx + thisbeam->rotation[7]*yy + thisbeam->rotation[8]*zz;
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            R[i][j] = thisbeam->rotation[i*3+j];
    //
    // Translation part
    //
    x -= thisbeam->translation[0];
    y -= thisbeam->translation[1];
    z -= thisbeam->translation[2];

    //
    // Compute beam parameters
    //
    a = thisbeam->jones[0]+1i*thisbeam->jones[1];
    b = thisbeam->jones[2]+1i*thisbeam->jones[3];
    l = thisbeam->order;

    //
    // Compute intermediate results
    //
    apib = a + 1i*b;
    amib = a - 1i*b;
    rsq = x*x+y*y;
    r = sqrt(rsq);
    phi = atan2(y,x);
    eikz = exp(1i*thisbeam->kz*z);
    eilphi = exp(phi*l*1i);
    ktr = thisbeam->kt*r;
    pref1 = thisbeam->E0 * eikz * eilphi;
    jnl = jn(l,ktr);
    pref3 = jnl * l / rsq * 1i;
    jnpl = jnp(l, ktr);
    pref4 = jnpl * thisbeam->kt / r;
    emip = exp(-phi*1i);
    epip = exp(phi*1i);
    pref5 = emip * jn(l-1,ktr);
    pref6 = epip * jn(l+1,ktr);
    pref7 = emip * jnp(l-1,ktr);
    pref8 = epip * jnp(l+1,ktr);
    //
    // Compute derivatives:
    // Ex derivatives
	gradEpxyz[0][0] = a * pref1 * (-y * pref3 + x * pref4);
    gradEpxyz[0][1] = a * pref1 * (x * pref3 + y * pref4);
    gradEpxyz[0][2] = a * pref1 * jnl * 1i * thisbeam->kz;

    // Ey derivatives
    gradEpxyz[1][0] = b * pref1 * (-y * pref3 + x * pref4);
    gradEpxyz[1][1] = b * pref1 * (x * pref3 + y * pref4);
    gradEpxyz[1][2] = b * pref1 * jnl * 1i * thisbeam->kz;

    // Ez derivatives
    gradEpxyz[2][0] = thisbeam->E0 * eikz * 0.5i * (thisbeam->kt_by_kz) * (eilphi * (apib * (1i * y * pref5 / rsq + pref7 * thisbeam->kt * x / r) - amib * (-1i * y * pref6 / rsq + pref8 * thisbeam->kt * x / r)) + eilphi * (-y * l) / rsq * (apib * pref5 - amib * pref6) * 1i);
	
    gradEpxyz[2][1] = thisbeam->E0 * eikz * 0.5i * (thisbeam->kt_by_kz) * (eilphi * (apib * (-1i * x * pref5 / rsq + pref7 * thisbeam->kt * y / r) - amib * (1i * x * pref6 / rsq + pref8 * thisbeam->kt * y / r)) + eilphi * (x * l) / rsq * (apib * pref5 - amib * pref6) * 1i);

    gradEpxyz[2][2] = pref1 * thisbeam->kt_by_kz * 0.5i * (apib * pref5 - amib * pref6);
    //
    // Back transform gradients to lab frame - transforms as second rank tensor
    //
    
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++){
            gradExyz[i*3+j] = 0.0 + 0.0i;
            for(int m=0; m<3; m++)
                for(int n=0; n<3; n++)
                    gradExyz[i*3+j] += R[m][i]*R[n][j]*gradEpxyz[m][n];
            }

    return gradExyz;
}
*/


void general_bessel_field_gradients(double x, double y, double z, BEAM *thisbeam, std::complex<double> gradEpxyz[][3]){
    // x, y, z coordinates of point
    // E0 field strength
    // kz, kt components of wavevector parallel and perpendicular to beam
    // a, b components of Jones vector
    // l order of optical vortex
    // the jn functions are Bessel functions of first kind as available in standard C math lib.

    using namespace std::complex_literals;
    
    std::complex<double> a, b, apib, amib, epip, emip;
    std::complex<double> eikz, eilphi, pref1, pref3, pref4, pref5, pref6, pref7, pref8;
    double r, phi, ktr, rsq;
    double jnl, jnpl;
    int l;
    //
    // Compute beam parameters
    //
    a = thisbeam->jones[0]+1i*thisbeam->jones[1];
    b = thisbeam->jones[2]+1i*thisbeam->jones[3];
    l = thisbeam->order;

    //
    // Compute intermediate results
    //
    apib = a + 1i*b;
    amib = a - 1i*b;
    rsq = x*x+y*y;
    r = sqrt(rsq);
    if ((fabs(x)<TINY) && (fabs(y)<TINY)){
//        std::cout<<"test"<<std::endl;
        phi=0.0;
    }
    else
        phi = atan2(y,x);
    eikz = exp(1i*thisbeam->kz*z);
    eilphi = exp(phi*l*1i);
    ktr = thisbeam->kt*r;
    pref1 = thisbeam->E0 * eikz * eilphi;
    jnl = jn(l,ktr);
    pref3 = jnl * l / rsq * 1i;
    jnpl = jnp(l, ktr);
    pref4 = jnpl * thisbeam->kt / r;
    emip = exp(-phi*1i);
    epip = exp(phi*1i);
    pref5 = emip * jn(l-1,ktr);
    pref6 = epip * jn(l+1,ktr);
    pref7 = emip * jnp(l-1,ktr);
    pref8 = epip * jnp(l+1,ktr);
    //
    // Compute derivatives:
    // Ex derivatives
    gradEpxyz[0][0] = a * pref1 * (-y * pref3 + x * pref4);
    gradEpxyz[0][1] = a * pref1 * (x * pref3 + y * pref4);
    gradEpxyz[0][2] = a * pref1 * jnl * 1i * thisbeam->kz;

    // Ey derivatives
    gradEpxyz[1][0] = b * pref1 * (-y * pref3 + x * pref4);
    gradEpxyz[1][1] = b * pref1 * (x * pref3 + y * pref4);
    gradEpxyz[1][2] = b * pref1 * jnl * 1i * thisbeam->kz;

    // Ez derivatives
    gradEpxyz[2][0] = thisbeam->E0 * eikz * 0.5i * (thisbeam->kt_by_kz) * (eilphi * (apib * (1i * y * pref5 / rsq + pref7 * thisbeam->kt * x / r) - amib * (-1i * y * pref6 / rsq + pref8 * thisbeam->kt * x / r)) + eilphi * (-y * l) / rsq * (apib * pref5 - amib * pref6) * 1i);
    
    gradEpxyz[2][1] = thisbeam->E0 * eikz * 0.5i * (thisbeam->kt_by_kz) * (eilphi * (apib * (-1i * x * pref5 / rsq + pref7 * thisbeam->kt * y / r) - amib * (1i * x * pref6 / rsq + pref8 * thisbeam->kt * y / r)) + eilphi * (x * l) / rsq * (apib * pref5 - amib * pref6) * 1i);

    gradEpxyz[2][2] = pref1 * thisbeam->kt_by_kz * 0.5i * (apib * pref5 - amib * pref6);
    //

    return;
}



//********************************************************************************
// Gaussian Beam:  Barton and Alexander, J. Appl. Phys. 66 (7), 2800-2802 (1989).
// Beam propagates along z axis with x polarisation.
//********************************************************************************

/*
std::complex<double> *gaussian_xpol(double xx, double yy, double zz, BEAM *thisbeam)
{
    // xx, yy, zz coordinates of point in lab frame
    // E0 field strength
    // k wavevector parallel to beam axis
    // w0 beam waste parameter
    using namespace std::complex_literals;
    using namespace std;
    
    static std::complex<double> Exyz[3],Epxyz[3]; // This will be a problem if we go parallel
    std::complex<double> eikz, phi0, pref1;
    double s, s2, s3, s4, s5;
    double x, y, z;
    double xi, eta, zeta;
    double rho2, rho4, rho6, rho8;
    std::complex<double> Q, Q2, Q3, Q4, Q5, Q6, Q7;
    //
    // Transform coordinates to beam frame
    // Rotation first
    //
    x = thisbeam->rotation[0]*xx + thisbeam->rotation[1]*yy + thisbeam->rotation[2]*zz;
    y = thisbeam->rotation[3]*xx + thisbeam->rotation[4]*yy + thisbeam->rotation[5]*zz;
    z = thisbeam->rotation[6]*xx + thisbeam->rotation[7]*yy + thisbeam->rotation[8]*zz;
    //
    // Translation part
    //
    x -= thisbeam->translation[0];
    y -= thisbeam->translation[1];
    z -= thisbeam->translation[2];
    //
    // Compute beam parameter and reduced coordinates
    //
    xi = x/thisbeam->w0;
    eta = y/thisbeam->w0;
    s = 1.0/(thisbeam->k * thisbeam->w0);
    zeta = z*s/thisbeam->w0;
    //
    // Compute intermediate results
    //
    s2 = s*s;
    s3 = s2*s;
    s4 = s2*s2;
    s5 = s2*s3;
    rho2 = xi*xi + eta*eta;
    rho4 = rho2*rho2;
    rho6 = rho4*rho2;
    rho8 = rho6*rho2;
    Q = 1.0/(1i + 2*zeta);
    Q2 = Q*Q;
    Q3 = Q2*Q;
    Q4 = Q3*Q;
    Q5 = Q4*Q;
    Q6 = Q5*Q;
    Q7 = Q6*Q;
    eikz = exp(-1i*thisbeam->k*z);
    phi0 = 1i*Q*exp(-1i*rho2*Q);
    pref1 = phi0 * thisbeam->E0 * eikz;
    //
    // Compute field components
    //
    Epxyz[0] = pref1 * (1.0 + s2*(-rho2*Q2 + 1i*rho4*Q3 - 2.0*Q2*xi*xi)
               + s4*(2*rho4*Q4 - 3i*rho6*Q5 - 0.5*rho8*Q6
                     + (8*rho2*Q4 - 2i*rho4*Q5)*xi*xi));
    Epxyz[1] = pref1 * (-s2*2*Q2*xi*eta + s4*(8*rho2*Q4 - 2i*rho4*Q5)*xi*eta);
    Epxyz[2] = pref1 * (-2*s*Q*xi + s3*(6*rho2*Q3 - 2i*rho4*Q4)*xi
                        + s5*(-20*rho4*Q5 + 10i*rho6*Q6 + rho8*Q7)*xi );
    std::cout << Epxyz[0] << Epxyz[1] << Epxyz[2] << std::endl;
    //
    // Back transform fields to lab frame
    //
    Exyz[0] = thisbeam->rotation[0]*Epxyz[0] + thisbeam->rotation[3]*Epxyz[1] + thisbeam->rotation[6]*Epxyz[2];
    Exyz[1] = thisbeam->rotation[1]*Epxyz[0] + thisbeam->rotation[4]*Epxyz[1] + thisbeam->rotation[7]*Epxyz[2];
    Exyz[2] = thisbeam->rotation[2]*Epxyz[0] + thisbeam->rotation[5]*Epxyz[1] + thisbeam->rotation[8]*Epxyz[2];
    return Exyz;
}
*/

void gaussian_barton5_fields(double x, double y, double z, BEAM *thisbeam, std::complex<double> *Epxyz)
{
    // xx, yy, zz coordinates of point in lab frame
    // E0 field strength
    // k wavevector parallel to beam axis
    // w0 beam waste parameter
    using namespace std::complex_literals;
    using namespace std;
    
    std::complex<double> eikz, phi0, pref1;
    double s, s2, s3, s4, s5;
    double xi, eta, zeta;
    double rho2, rho4, rho6, rho8;
    std::complex<double> Q, Q2, Q3, Q4, Q5, Q6, Q7;
    //
    // Compute beam parameter and reduced coordinates
    //
    xi = x/thisbeam->w0;
    eta = y/thisbeam->w0;
    s = 1.0/(thisbeam->k * thisbeam->w0);
    zeta = z*s/thisbeam->w0;
    //
    // Compute intermediate results
    //
    s2 = s*s;
    s3 = s2*s;
    s4 = s2*s2;
    s5 = s2*s3;
    rho2 = xi*xi + eta*eta;
    rho4 = rho2*rho2;
    rho6 = rho4*rho2;
    rho8 = rho6*rho2;
    Q = 1.0/(1i + 2*zeta);
    Q2 = Q*Q;
    Q3 = Q2*Q;
    Q4 = Q3*Q;
    Q5 = Q4*Q;
    Q6 = Q5*Q;
    Q7 = Q6*Q;
    eikz = exp(-1i*thisbeam->k*z);
    phi0 = 1i*Q*exp(-1i*rho2*Q);
    pref1 = phi0 * thisbeam->E0 * eikz;
    //
    // Compute field components
    //
    Epxyz[0] = pref1 * (1.0 + s2*(-rho2*Q2 + 1i*rho4*Q3 - 2.0*Q2*xi*xi)
               + s4*(2*rho4*Q4 - 3i*rho6*Q5 - 0.5*rho8*Q6
                     + (8*rho2*Q4 - 2i*rho4*Q5)*xi*xi));
    Epxyz[1] = pref1 * (-s2*2*Q2*xi*eta + s4*(8*rho2*Q4 - 2i*rho4*Q5)*xi*eta);
    Epxyz[2] = pref1 * (-2*s*Q*xi + s3*(6*rho2*Q3 - 2i*rho4*Q4)*xi
                        + s5*(-20*rho4*Q5 + 10i*rho6*Q6 + rho8*Q7)*xi );
    //std::cout << Epxyz[0] << Epxyz[1] << Epxyz[2] << std::endl;
    return;
}
    

/*
std::complex<double> *gaussian_xpol_gradient(double xx, double yy, double zz, BEAM *thisbeam)
{
    // xx, yy, zz coordinates of point in lab frame
    // E0 field strength
    // k wavevector parallel to beam axis
    // w0 beam waste parameter
    using namespace std::complex_literals;
    
    static std::complex<double> gradExyz[9],gradEpxyz[3][3]; // This will be a problem if we go parallel
    std::complex<double> eikz, phi0, pref1;
    double s, s2, s3, s4, s5;
    double xi, eta, zeta;
    double rho2, rho4, rho6, rho8;
    std::complex<double> Q, Q2, Q3, Q4, Q5, Q6, Q7;
    double x, y, z, R[3][3];
    double drho2dxi, drho2deta, drho4dxi, drho4deta, drho6dxi, drho6deta, drho8dxi, drho8deta;
    std::complex<double> dQdzeta, dQ2dzeta, dQ3dzeta, dQ4dzeta, dQ5dzeta, dQ6dzeta, dQ7dzeta;
    std::complex<double> dphi0dxipart, dphi0detapart, dphi0dzetapart;
    //
    // Transform coordinates to beam frame
    // Rotation first
    //
    x = thisbeam->rotation[0]*xx + thisbeam->rotation[1]*yy + thisbeam->rotation[2]*zz;
    y = thisbeam->rotation[3]*xx + thisbeam->rotation[4]*yy + thisbeam->rotation[5]*zz;
    z = thisbeam->rotation[6]*xx + thisbeam->rotation[7]*yy + thisbeam->rotation[8]*zz;
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            R[i][j] = thisbeam->rotation[i*3+j];
    //
    // Translation part
    //
    x -= thisbeam->translation[0];
    y -= thisbeam->translation[1];
    z -= thisbeam->translation[2];

    //
    // Compute beam parameter and reduced coordinates
    //
    xi = x/thisbeam->w0;
    eta = y/thisbeam->w0;
    s = 1.0/(thisbeam->k * thisbeam->w0);
    zeta = z*s/thisbeam->w0;

    //
    // Compute intermediate results
    //
    s2 = s*s;
    s3 = s2*s;
    s4 = s2*s2;
    s5 = s2*s3;
    //
    rho2 = xi*xi + eta*eta;
    rho4 = rho2*rho2;
    rho6 = rho4*rho2;
    rho8 = rho6*rho2;
    drho2dxi = 2*xi;
    drho2deta = 2*eta;
    drho4dxi = 4*xi*rho2;
    drho4deta = 4*eta*rho2;
    drho6dxi = 6*xi*rho4;
    drho6deta = 6*eta*rho4;
    drho8dxi = 8*xi*rho6;
    drho8deta = 8*eta*rho6;
    //
    Q = 1.0/(1i + 2*zeta);
    Q2 = Q*Q;
    Q3 = Q2*Q;
    Q4 = Q3*Q;
    Q5 = Q4*Q;
    Q6 = Q5*Q;
    Q7 = Q6*Q;
    dQdzeta = -2.0*Q2;
    dQ2dzeta = -4.0*Q3;
    dQ3dzeta = -6.0*Q4;
    dQ4dzeta = -8.0*Q5;
    dQ5dzeta = -10.0*Q6;
    dQ6dzeta = -12.0*Q7;
    dQ7dzeta = -14.0*Q7*Q;
    //
    eikz = exp(-1i*thisbeam->k*z);
    phi0 = 1i*Q*exp(-1i*rho2*Q);
    dphi0dxipart = -2i*Q*xi;
    dphi0detapart = -2i*Q*eta;
    dphi0dzetapart = -2.0*Q+2i*rho2*Q2-1i/s2;
    pref1 = phi0 * thisbeam->E0 * eikz;
    //
    // Compute derivatives:
    // Ex derivatives
    gradEpxyz[0][0] = pref1 * (s2*(-drho2dxi*Q2 + 1i*drho4dxi*Q3 - 4*xi*Q2)
                               + s4*(2*drho4dxi*Q4 - 3i*drho6dxi*Q5 - 0.5*drho8dxi*Q6
                               + (8*drho2dxi*Q4 - 2i*drho4dxi*Q5)*xi*xi + 2*xi*(8*rho2*Q4 - 2i*rho4*Q5))
                     + dphi0dxipart*(1.0 + s2*(-rho2*Q2 + 1i*rho4*Q3 - 2.0*Q2*xi*xi)
                                     + s4*(2*rho4*Q4 - 3i*rho6*Q5 - 0.5*rho8*Q6
                                           + (8*rho2*Q4 - 2i*rho4*Q5)*xi*xi))) / thisbeam->w0;
    
    gradEpxyz[0][1] = pref1 * (s2*(-drho2deta*Q2 + 1i*drho4deta*Q3)
                      + s4*(2*drho4deta*Q4 - 3i*drho6deta*Q5 - 0.5*drho8deta*Q6
                      + (8*drho2deta*Q4 - 2i*drho4deta*Q5)*xi*xi)
            + dphi0detapart*(1.0 + s2*(-rho2*Q2 + 1i*rho4*Q3 - 2.0*Q2*xi*xi)
                            + s4*(2*rho4*Q4 - 3i*rho6*Q5 - 0.5*rho8*Q6
                                  + (8*rho2*Q4 - 2i*rho4*Q5)*xi*xi))) / thisbeam->w0;
    
    gradEpxyz[0][2] = pref1 * (s2*(-rho2*dQ2dzeta + 1i*rho4*dQ3dzeta - 2.0*dQ2dzeta*xi*xi)
            + s4*(2*rho4*dQ4dzeta - 3i*rho6*dQ5dzeta - 0.5*rho8*dQ6dzeta
                + (8*rho2*dQ4dzeta - 2i*rho4*dQ5dzeta)*xi*xi)
                               + dphi0dzetapart*(1.0 + s2*(-rho2*Q2 + 1i*rho4*Q3 - 2.0*Q2*xi*xi)
                                                 + s4*(2*rho4*Q4 - 3i*rho6*Q5 - 0.5*rho8*Q6
                                                       + (8*rho2*Q4 - 2i*rho4*Q5)*xi*xi))) * thisbeam->k*s2;
    
    // Ey derivatives
    gradEpxyz[1][0] = pref1 * (-2*s2*Q2*eta
                + s4*((8*drho2dxi*Q4 - 2i*drho4dxi*Q5)*xi*eta
                * (8*rho2*Q4 - 2i*rho4*Q5)*eta)
                              + dphi0dxipart*(-s2*2*Q2*xi*eta + s4*(8*rho2*Q4 - 2i*rho4*Q5)*xi*eta)) / thisbeam->w0;
    
    gradEpxyz[1][1] = pref1 * (-2*s2*Q2*xi
                + s4*((8*drho2deta*Q4 - 2i*drho4deta*Q5)*xi*eta
                * (8*rho2*Q4 - 2i*rho4*Q5)*xi)
                              + dphi0detapart*(-s2*2*Q2*xi*eta + s4*(8*rho2*Q4 - 2i*rho4*Q5)*xi*eta)) / thisbeam->w0;

    gradEpxyz[1][2] = pref1 * (-s2*dQ2dzeta*xi*eta
                + s4*(8*rho2*dQ4dzeta - 2i*rho4*dQ5dzeta)*xi*eta
                               +dphi0dzetapart*(-s2*2*Q2*xi*eta + s4*(8*rho2*Q4 - 2i*rho4*Q5)*xi*eta)) * thisbeam->k*s2;
  
    // Ez derivatives
    gradEpxyz[2][0] = pref1 * (-2*s*Q
                + s3*((6*drho2dxi*Q3 - 2i*drho4dxi*Q4)*xi + 6*rho2*Q3 - 2i*rho4*Q4)
                + s5*((-20 * drho4dxi*Q5 + 10i*drho6dxi*Q6 + drho8dxi*Q7)*xi
                   -20*rho4*Q5 + 10i*rho6*Q6 + rho8*Q7) +
                dphi0dxipart*(-2*s*Q*xi + s3*(6*rho2*Q3 - 2i*rho4*Q4)*xi
                            + s5*(-20*rho4*Q5 + 10i*rho6*Q6 + rho8*Q7)*xi)) / thisbeam->w0;

    gradEpxyz[2][1] = pref1 * (s3*((6*drho2deta*Q3 - 2i*drho4deta*Q4)*xi)
                                   + s5*((-20 * drho4deta*Q5 + 10i*drho6deta*Q6 + drho8deta*Q7)*xi) +
                                   dphi0detapart*(-2*s*Q*xi + s3*(6*rho2*Q3 - 2i*rho4*Q4)*xi
                                               + s5*(-20*rho4*Q5 + 10i*rho6*Q6 + rho8*Q7)*xi)) / thisbeam->w0;

    gradEpxyz[2][2] = pref1 * (-2*s*dQdzeta*xi + s3*(6*rho2*dQ3dzeta - 2i*rho4*dQ4dzeta)*xi
            + s5*(-20*rho4*dQ5dzeta + 10i*rho6*dQ6dzeta + rho8*dQ7dzeta)*xi
                               + dphi0dzetapart*(-2*s*Q*xi + s3*(6*rho2*Q3 - 2i*rho4*Q4)*xi
                                                 + s5*(-20*rho4*Q5 + 10i*rho6*Q6 + rho8*Q7)*xi )) * thisbeam->k*s2;
    
    //
    // Back transform gradients to lab frame - transforms as second rank tensor
    //
    
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++){
            gradExyz[i*3+j] = 0.0 + 0.0i;
            for(int m=0; m<3; m++)
                for(int n=0; n<3; n++)
                    gradExyz[i*3+j] += R[m][i]*R[n][j]*gradEpxyz[m][n];
            }

    return gradExyz;
}
*/


void gaussian_barton5_field_gradients(double x, double y, double z, BEAM *thisbeam, std::complex<double> gradEpxyz[][3])
{
    // xx, yy, zz coordinates of point in lab frame
    // E0 field strength
    // k wavevector parallel to beam axis
    // w0 beam waste parameter
    using namespace std::complex_literals;
    
    std::complex<double> eikz, phi0, pref1;
    double s, s2, s3, s4, s5;
    double xi, eta, zeta;
    double rho2, rho4, rho6, rho8;
    std::complex<double> Q, Q2, Q3, Q4, Q5, Q6, Q7;
    double drho2dxi, drho2deta, drho4dxi, drho4deta, drho6dxi, drho6deta, drho8dxi, drho8deta;
    std::complex<double> dQdzeta, dQ2dzeta, dQ3dzeta, dQ4dzeta, dQ5dzeta, dQ6dzeta, dQ7dzeta;
    std::complex<double> dphi0dxipart, dphi0detapart, dphi0dzetapart;
    //
    // Compute beam parameter and reduced coordinates
    //
    xi = x/thisbeam->w0;
    eta = y/thisbeam->w0;
    s = 1.0/(thisbeam->k * thisbeam->w0);
    zeta = z*s/thisbeam->w0;

    //
    // Compute intermediate results
    //
    s2 = s*s;
    s3 = s2*s;
    s4 = s2*s2;
    s5 = s2*s3;
    //
    rho2 = xi*xi + eta*eta;
    rho4 = rho2*rho2;
    rho6 = rho4*rho2;
    rho8 = rho6*rho2;
    drho2dxi = 2*xi;
    drho2deta = 2*eta;
    drho4dxi = 4*xi*rho2;
    drho4deta = 4*eta*rho2;
    drho6dxi = 6*xi*rho4;
    drho6deta = 6*eta*rho4;
    drho8dxi = 8*xi*rho6;
    drho8deta = 8*eta*rho6;
    //
    Q = 1.0/(1i + 2*zeta);
    Q2 = Q*Q;
    Q3 = Q2*Q;
    Q4 = Q3*Q;
    Q5 = Q4*Q;
    Q6 = Q5*Q;
    Q7 = Q6*Q;
    dQdzeta = -2.0*Q2;
    dQ2dzeta = -4.0*Q3;
    dQ3dzeta = -6.0*Q4;
    dQ4dzeta = -8.0*Q5;
    dQ5dzeta = -10.0*Q6;
    dQ6dzeta = -12.0*Q7;
    dQ7dzeta = -14.0*Q7*Q;
    //
    eikz = exp(-1i*thisbeam->k*z);
    phi0 = 1i*Q*exp(-1i*rho2*Q);
    dphi0dxipart = -2i*Q*xi;
    dphi0detapart = -2i*Q*eta;
    dphi0dzetapart = -2.0*Q+2i*rho2*Q2-1i/s2;
    pref1 = phi0 * thisbeam->E0 * eikz;
    //
    // Compute derivatives:
    // Ex derivatives
    gradEpxyz[0][0] = pref1 * (s2*(-drho2dxi*Q2 + 1i*drho4dxi*Q3 - 4*xi*Q2)
                               + s4*(2*drho4dxi*Q4 - 3i*drho6dxi*Q5 - 0.5*drho8dxi*Q6
                               + (8*drho2dxi*Q4 - 2i*drho4dxi*Q5)*xi*xi + 2*xi*(8*rho2*Q4 - 2i*rho4*Q5))
                     + dphi0dxipart*(1.0 + s2*(-rho2*Q2 + 1i*rho4*Q3 - 2.0*Q2*xi*xi)
                                     + s4*(2*rho4*Q4 - 3i*rho6*Q5 - 0.5*rho8*Q6
                                           + (8*rho2*Q4 - 2i*rho4*Q5)*xi*xi))) / thisbeam->w0;
    
    gradEpxyz[0][1] = pref1 * (s2*(-drho2deta*Q2 + 1i*drho4deta*Q3)
                      + s4*(2*drho4deta*Q4 - 3i*drho6deta*Q5 - 0.5*drho8deta*Q6
                      + (8*drho2deta*Q4 - 2i*drho4deta*Q5)*xi*xi)
            + dphi0detapart*(1.0 + s2*(-rho2*Q2 + 1i*rho4*Q3 - 2.0*Q2*xi*xi)
                            + s4*(2*rho4*Q4 - 3i*rho6*Q5 - 0.5*rho8*Q6
                                  + (8*rho2*Q4 - 2i*rho4*Q5)*xi*xi))) / thisbeam->w0;
    
    gradEpxyz[0][2] = pref1 * (s2*(-rho2*dQ2dzeta + 1i*rho4*dQ3dzeta - 2.0*dQ2dzeta*xi*xi)
            + s4*(2*rho4*dQ4dzeta - 3i*rho6*dQ5dzeta - 0.5*rho8*dQ6dzeta
                + (8*rho2*dQ4dzeta - 2i*rho4*dQ5dzeta)*xi*xi)
                               + dphi0dzetapart*(1.0 + s2*(-rho2*Q2 + 1i*rho4*Q3 - 2.0*Q2*xi*xi)
                                                 + s4*(2*rho4*Q4 - 3i*rho6*Q5 - 0.5*rho8*Q6
                                                       + (8*rho2*Q4 - 2i*rho4*Q5)*xi*xi))) * thisbeam->k*s2;
    
    // Ey derivatives
    gradEpxyz[1][0] = pref1 * (-2*s2*Q2*eta
                + s4*((8*drho2dxi*Q4 - 2i*drho4dxi*Q5)*xi*eta
                * (8*rho2*Q4 - 2i*rho4*Q5)*eta)
                              + dphi0dxipart*(-s2*2*Q2*xi*eta + s4*(8*rho2*Q4 - 2i*rho4*Q5)*xi*eta)) / thisbeam->w0;
    
    gradEpxyz[1][1] = pref1 * (-2*s2*Q2*xi
                + s4*((8*drho2deta*Q4 - 2i*drho4deta*Q5)*xi*eta
                * (8*rho2*Q4 - 2i*rho4*Q5)*xi)
                              + dphi0detapart*(-s2*2*Q2*xi*eta + s4*(8*rho2*Q4 - 2i*rho4*Q5)*xi*eta)) / thisbeam->w0;

    gradEpxyz[1][2] = pref1 * (-s2*dQ2dzeta*xi*eta
                + s4*(8*rho2*dQ4dzeta - 2i*rho4*dQ5dzeta)*xi*eta
                               +dphi0dzetapart*(-s2*2*Q2*xi*eta + s4*(8*rho2*Q4 - 2i*rho4*Q5)*xi*eta)) * thisbeam->k*s2;
  
    // Ez derivatives
    gradEpxyz[2][0] = pref1 * (-2*s*Q
                + s3*((6*drho2dxi*Q3 - 2i*drho4dxi*Q4)*xi + 6*rho2*Q3 - 2i*rho4*Q4)
                + s5*((-20 * drho4dxi*Q5 + 10i*drho6dxi*Q6 + drho8dxi*Q7)*xi
                   -20*rho4*Q5 + 10i*rho6*Q6 + rho8*Q7) +
                dphi0dxipart*(-2*s*Q*xi + s3*(6*rho2*Q3 - 2i*rho4*Q4)*xi
                            + s5*(-20*rho4*Q5 + 10i*rho6*Q6 + rho8*Q7)*xi)) / thisbeam->w0;

    gradEpxyz[2][1] = pref1 * (s3*((6*drho2deta*Q3 - 2i*drho4deta*Q4)*xi)
                                   + s5*((-20 * drho4deta*Q5 + 10i*drho6deta*Q6 + drho8deta*Q7)*xi) +
                                   dphi0detapart*(-2*s*Q*xi + s3*(6*rho2*Q3 - 2i*rho4*Q4)*xi
                                               + s5*(-20*rho4*Q5 + 10i*rho6*Q6 + rho8*Q7)*xi)) / thisbeam->w0;

    gradEpxyz[2][2] = pref1 * (-2*s*dQdzeta*xi + s3*(6*rho2*dQ3dzeta - 2i*rho4*dQ4dzeta)*xi
            + s5*(-20*rho4*dQ5dzeta + 10i*rho6*dQ6dzeta + rho8*dQ7dzeta)*xi
                               + dphi0dzetapart*(-2*s*Q*xi + s3*(6*rho2*Q3 - 2i*rho4*Q4)*xi
                                                 + s5*(-20*rho4*Q5 + 10i*rho6*Q6 + rho8*Q7)*xi )) * thisbeam->k*s2;
    
    return;
}



//***********************************************************************************
// Gaussian Beam:  Sheppard and Saghafi, Vol. 16, No. 6/June 1999/J. Opt. Soc. Am. A.
// Beam propagates along z axis with x polarisation.
//***********************************************************************************

void gaussian_csp_fields(double x, double y, double z, BEAM *thisbeam, std::complex<double> *Epxyz)
{
    // x, y, z coordinates of point
    // E0 field strength
    // k wavevector parallel to beam axis
    // w0 beam waste parameter

    using namespace std::complex_literals;
    
    std::complex<double> fkr, gkr, fmgkr, pref1, ikz0, scale;
    double z0, kk, kx, ky;
    std::complex<double> kz, kR, kR2;
    //
    // Compute beam parameters
    //
    kk = thisbeam->k;
    z0 = kk*(thisbeam->w0)*(thisbeam->w0)/2.0;
    ikz0 = 1.0i*kk*z0;
    scale = gg(ikz0) - 0.5i*ff(ikz0)*ikz0;
    //
    // Compute coordinates
    //
    kx = kk*x;
    ky = kk*y;
    kz = kk*(z-1.0i*z0);
    kR2 = kx*kx + ky*ky + kz*kz;
    kR = sqrt(kR2);
    //
    // Compute intermediate results
    //
    fkr = ff(kR);
    gkr = gg(kR);
    fmgkr = fkr-gkr;
    pref1 = thisbeam->E0/scale;
    //
    // Compute field components
    //
    if (fabs(kR)<TINY){
        Epxyz[0] = pref1*(1.0 + 0.5i*kz);
        Epxyz[1] = pref1*(0.0 + 0.0i);
        Epxyz[2] = -pref1*0.5i*kx;

    }
    else{
        Epxyz[0] = pref1*(gkr + fmgkr*kx*kx/kR2 + 0.5i*fkr*kz);
        Epxyz[1] = pref1*(fmgkr*kx*ky/kR2);
        Epxyz[2] = pref1*(fmgkr*kx*kz/kR2 - 0.5i*fkr*kx);
    }
    return;
}


void gaussian_csp_field_gradients(double x, double y, double z, BEAM *thisbeam, std::complex<double> gradEpxyz[][3]){
    // x, y, z coordinates of point
    // E0 field strength
    // k wavevector parallel to beam axis
    // w0 beam waste parameter

    using namespace std::complex_literals;
    
    std::complex<double> pref1, pref2, pref3;
    std::complex<double> Bn[4];
//    double r, phi, ktr;
    double z0, kk, kx, ky, kx2, ky2;
    std::complex<double> kz, kz2, kR, kR2, ikz0, scale;
    //
    // Compute beam parameters
    //
    kk = thisbeam->k;
    z0 = kk*(thisbeam->w0)*(thisbeam->w0)/2.0;
    ikz0 = 1.0i*kk*z0;
    scale = gg(ikz0) - 0.5i*ff(ikz0)*ikz0;
    //
    // Compute coordinates
    //
    kx = kk*x;
    kx2 = kx*kx;
    ky = kk*y;
    ky2 = ky*ky;
    kz = kk*(z-1.0i*z0);
    kz2 = kz*kz;
    kR2 = kx2 + ky2 + kz2;
    kR = sqrt(kR2);
    //
    // These are spherical bessel functions order n, divided by argument^n i.e. j_n(x)/x^n.
    //
    if (fabs(kR)<TINY){
        Bn[0] = 1.0;
        Bn[1] = Bn[0]/3.0;
        Bn[2] = Bn[1]/5.0;
        Bn[3] = Bn[2]/7.0;
    } else {
        Bn[0] = sin(kR)/kR;
        Bn[1] = (Bn[0] - cos(kR))/kR2;
        Bn[2] = (3.0*Bn[1] - Bn[0])/kR2;
        Bn[3] = (5.0*Bn[2] - Bn[1])/kR2;
    }
//    std::cout << "kx: "<<kx<<" ky: "<<ky<<" kz: "<<kz<<" kR: "<<kR <<Bn[0]<<Bn[1]<<Bn[2]<<Bn[3]  <<std::endl;
    //
    // Compute intermediate results
    //
    pref1 = 1.5*kk*thisbeam->E0/scale;
    pref2 = 0.5i*kk*thisbeam->E0/scale;
    pref3 = (1.0-1i*kz)*Bn[2] - Bn[1] - kx2*Bn[3];
    //
    // Compute derivatives:
    // Ex derivatives
    gradEpxyz[0][0] = pref1*kx*(2.0*Bn[2] + pref3);
    gradEpxyz[0][1] = pref1*ky*pref3;
    gradEpxyz[0][2] = pref1*kz*pref3 + pref2*(Bn[0]+kR2*Bn[2]);

    // Ey derivatives
    gradEpxyz[1][0] = pref1*ky*(Bn[2]-kx2*Bn[3]);
    gradEpxyz[1][1] = pref1*kx*(Bn[2]-ky2*Bn[3]);
    gradEpxyz[1][2] = -pref1*kx*ky*kz*Bn[3];

    // Ez derivatives
    gradEpxyz[2][0] = pref1*((kz+1i*kx2)*Bn[2] - kx2*kz*Bn[3]) - pref2*(Bn[0]+kR2*Bn[2]);
    gradEpxyz[2][1] = pref1*kx*ky*(1i*Bn[2]-kz*Bn[3]);
    gradEpxyz[2][2] = pref1*kx*((1.0+1i*kz)*Bn[2] - kz2*Bn[3]);
    //
    
    //std::cout<<"x derivs: "<<gradEpxyz[0][0]<<" "<<gradEpxyz[1][0]<<" "<<gradEpxyz[2][0]<<std::endl;
    //std::cout<<"y derivs: "<<gradEpxyz[0][1]<<" "<<gradEpxyz[1][1]<<" "<<gradEpxyz[2][1]<<std::endl;
    //std::cout<<"z derivs: "<<gradEpxyz[0][2]<<" "<<gradEpxyz[1][2]<<" "<<gradEpxyz[2][2]<<std::endl;
    return;
}





//********************************************************************************
// Plane Wave, propagating along z axis, with x-y plane polarisation (specified
// using Jones vector.
//********************************************************************************
/*
std::complex<double> *plane_wave(double xx, double yy, double zz, BEAM *thisbeam){
    // x, y, z coordinates of point
    // E0 field strength
    // k wavevector parallel to beam direction
    // a, b components of Jones vector
    using namespace std::complex_literals;
    
    std::complex<double> Jx, Jy;
    static std::complex<double> Exyz[3],Epxyz[3]; // This will be a problem if we go parallel
    std::complex<double> eikz, pref1;
    double x, y, z;
    //
    // Transform coordinates to beam frame
    // Rotation first
    //
    x = thisbeam->rotation[0]*xx + thisbeam->rotation[1]*yy + thisbeam->rotation[2]*zz;
    y = thisbeam->rotation[3]*xx + thisbeam->rotation[4]*yy + thisbeam->rotation[5]*zz;
    z = thisbeam->rotation[6]*xx + thisbeam->rotation[7]*yy + thisbeam->rotation[8]*zz;
    //
    // Translation part
    //
    x -= thisbeam->translation[0];
    y -= thisbeam->translation[1];
    z -= thisbeam->translation[2];

    //
    // Compute beam parameters
    //
    Jx = thisbeam->jones[0]+1i*thisbeam->jones[1];
    Jy = thisbeam->jones[2]+1i*thisbeam->jones[3];
    //
    // Compute intermediate results
    //
//    printf("Jones matrix: %f %f %f %f\n",thisbeam->jones[0],thisbeam->jones[1],thisbeam->jones[2],thisbeam->jones[3]);
    eikz = exp(-1i*thisbeam->k*z);
    pref1 = thisbeam->E0 * eikz;
    //
    // Compute field components
    //
    Epxyz[0] = pref1 * Jx;
    Epxyz[1] = pref1 * Jy;
    Epxyz[2] = 0.0;
    //
    // Back transform fields to lab frame
    //
    Exyz[0] = thisbeam->rotation[0]*Epxyz[0] + thisbeam->rotation[3]*Epxyz[1] + thisbeam->rotation[6]*Epxyz[2];
    Exyz[1] = thisbeam->rotation[1]*Epxyz[0] + thisbeam->rotation[4]*Epxyz[1] + thisbeam->rotation[7]*Epxyz[2];
    Exyz[2] = thisbeam->rotation[2]*Epxyz[0] + thisbeam->rotation[5]*Epxyz[1] + thisbeam->rotation[8]*Epxyz[2];
    return Exyz;
}
*/

void plane_wave_fields(double x, double y, double z, BEAM *thisbeam, std::complex<double> *Epxyz){
    // x, y, z coordinates of point
    // E0 field strength
    // k wavevector parallel to beam direction
    // Jx, Jy components of Jones vector
    using namespace std::complex_literals;
    
    std::complex<double> Jx, Jy;
    std::complex<double> eikz, pref1;
    //
    // Compute beam parameters
    //
    Jx = thisbeam->jones[0]+1i*thisbeam->jones[1];
    Jy = thisbeam->jones[2]+1i*thisbeam->jones[3];
    //
    // Compute intermediate results
    //
    eikz = exp(-1i*thisbeam->k*z);
    pref1 = thisbeam->E0 * eikz;
    //
    // Compute field components
    //
    Epxyz[0] = pref1 * Jx;
    Epxyz[1] = pref1 * Jy;
    Epxyz[2] = 0.0;
    return;
}

/*
std::complex<double> *plane_wave_gradient(double xx, double yy, double zz, BEAM *thisbeam){
    // x, y, z coordinates of point
    // E0 field strength
    // k wavevector parallel to propagation direction
    // Jx, Jy components of Jones vector
    using namespace std::complex_literals;
    
    std::complex<double> Jx, Jy;
    static std::complex<double> gradExyz[9],gradEpxyz[3][3]; // This will be a problem if we go parallel
    std::complex<double> eikz, pref1;
    //std::complex<double> dExdx, dExdy, dExdz, dEydx, dEydy, dEydz, dEzdx, dEzdy, dEzdz;
    double x, y, z, R[3][3];
    //
    // Transform coordinates to beam frame
    // Rotation first
    //
    x = thisbeam->rotation[0]*xx + thisbeam->rotation[1]*yy + thisbeam->rotation[2]*zz;
    y = thisbeam->rotation[3]*xx + thisbeam->rotation[4]*yy + thisbeam->rotation[5]*zz;
    z = thisbeam->rotation[6]*xx + thisbeam->rotation[7]*yy + thisbeam->rotation[8]*zz;
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            R[i][j] = thisbeam->rotation[i*3+j];
    //
    // Translation part
    //
    x -= thisbeam->translation[0];
    y -= thisbeam->translation[1];
    z -= thisbeam->translation[2];

    //
    // Compute beam parameters
    //
    Jx = thisbeam->jones[0]+1i*thisbeam->jones[1];
    Jy = thisbeam->jones[2]+1i*thisbeam->jones[3];

    //
    // Compute intermediate results
    //
    eikz = exp(-1i*thisbeam->k*z);
    pref1 = thisbeam->E0 * eikz;
    //
    // Compute derivatives:
    // Ex derivatives
    gradEpxyz[0][0] = 0.0;
    gradEpxyz[0][1] = 0.0;
    gradEpxyz[0][2] = -1i * thisbeam->k * Jx * pref1;

    // Ey derivatives
    gradEpxyz[1][0] = 0.0;
    gradEpxyz[1][1] = 0.0;
    gradEpxyz[1][2] = -1i * thisbeam->k * Jy * pref1;

    // Ez derivatives
    gradEpxyz[2][0] = 0.0;
    gradEpxyz[2][1] = 0.0;
    gradEpxyz[2][2] = 0.0;
    //
    // Back transform gradients to lab frame - transforms as second rank tensor
    //
    
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++){
            gradExyz[i*3+j] = 0.0 + 0.0i;
            for(int m=0; m<3; m++)
                for(int n=0; n<3; n++)
                    gradExyz[i*3+j] += R[m][i]*R[n][j]*gradEpxyz[m][n];
            }

    return gradExyz;

}
*/

void plane_wave_field_gradients(double x, double y, double z, BEAM *thisbeam, std::complex<double> gradEpxyz[][3]){
    // x, y, z coordinates of point
    // E0 field strength
    // k wavevector parallel to propagation direction
    // Jx, Jy components of Jones vector
    using namespace std::complex_literals;
    
    std::complex<double> Jx, Jy;
    std::complex<double> eikz, pref1;
    //
    // Compute beam parameters
    //
    Jx = thisbeam->jones[0]+1i*thisbeam->jones[1];
    Jy = thisbeam->jones[2]+1i*thisbeam->jones[3];

    //
    // Compute intermediate results
    //
    eikz = exp(-1i*thisbeam->k*z);
    pref1 = thisbeam->E0 * eikz;
    //
    // Compute derivatives:
    // Ex derivatives
    gradEpxyz[0][0] = 0.0;
    gradEpxyz[0][1] = 0.0;
    gradEpxyz[0][2] = -1i * thisbeam->k * Jx * pref1;

    // Ey derivatives
    gradEpxyz[1][0] = 0.0;
    gradEpxyz[1][1] = 0.0;
    gradEpxyz[1][2] = -1i * thisbeam->k * Jy * pref1;

    // Ez derivatives
    gradEpxyz[2][0] = 0.0;
    gradEpxyz[2][1] = 0.0;
    gradEpxyz[2][2] = 0.0;

    return;

}


//********************************************************************************
// Utility functions
//********************************************************************************

double eek(double kappa, double kk, double E0, double zR, int order, int gouy){
    //
    // Function to provide E(kappa) for Barnett expression for focussed
    // Laguerre-Gaussian beams.
    // Using abs(order) to avoid symmetry breaking.
    //
    double temp;
    double k2, kap2, invdenom, index;
    k2 = kk*kk;
    kap2 = kappa*kappa;
    invdenom = 1.0/(k2-kap2);
    index = (2*gouy+abs(order)+1)/2.0;
    temp = E0*exp(-0.5*kk*kap2*zR*invdenom)*pow(kap2*invdenom,index)*sqrt(k2*invdenom);
    return temp;
}


double jnp(int order, double x){
    //
    // Function to provide first derivative of nth order Bessel function
    // using J'_n = (J_n-1 - J_n+1) / 2
    //
    double temp;
    temp = 0.5*(jn(order-1,x)-jn(order+1,x));
    return temp;
}

std::complex<double> ff(std::complex<double> x){
    //
    // Function to evaluate f(kR) from Sheppard and Saghafi paper.
    // f(kr) = j_0(kr) + j_2(kr) where j_n is spherical bessel function.
    // Evaluates to f(x) = -3.0*(np.cos(x)/x**2 - np.sin(x)/x**3)
    // with asymptotic value (x->0) of 1.0+0.0j.
    //
    // Not using built-in (C++17) complex math library - need gcc-12 for that!
    // std::complex<double> temp = std::sph_bessel(0,x) + std::sph_bessel(2,x);
    // Not using GSL - doesn't handle complex arguments.
    //
    // Therefore use sines and cosines and pay attention to asymptotic values.
    //
    std::complex<double> temp (1.0,0.0);
    std::complex<double> x2, x3;
    x2 = x*x;
    x3 = x2*x;
    if (fabs(x) > TINY){
        temp = 3.0*(sin(x)/x3 - cos(x)/x2);
    }
    return temp;
}

std::complex<double> gg(std::complex<double> x){
    //
    // Function to evaluate g(kR) from Sheppard and Saghafi paper.
    // g(kr) = j_0(kr) - (1/2)j_2(kr) where j_n is spherical bessel function.
    // Evaluates to f(x) = 1.5*(np.sin(x)/x + np.cos(x)/x**2 - np.sin(x)/x**3)
    // with asymptotic value (x->0) of 1.0+0.0j.
    //
    // Not using built-in (C++17) complex math library - need gcc-12 for that!
    // std::complex<double> temp = std::sph_bessel(0,x) - 0.5*std::sph_bessel(2,x);
    // Not using GSL - doesn't handle complex arguments.
    //
    // Therefore use sines and cosines and pay attention to asymptotic values.
    //
    std::complex<double> temp (1.0,0.0);
    std::complex<double> x2, x3;
    x2 = x*x;
    x3 = x2*x;
    if (fabs(x) > TINY){
        temp = 1.5*((x2-1.0)*sin(x)/x3 + cos(x)/x2);
    }
    return temp;
}

