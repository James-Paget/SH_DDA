// Header for the Dipoles library

#ifndef DIPOLES_HPP
#define DIPOLES_HPP

#define BEAMTYPE_PLANE 0
#define BEAMTYPE_GAUSS_BARTON5 1
#define BEAMTYPE_GAUSS_CSP 2
#define BEAMTYPE_BESSEL 3

#define TINY 1e-12
#define EPS0 8.85e-12//8.8541878128e-12

#define _USE_MATH_DEFINES
#include <complex>
#include <cmath>
#include <iostream>
//#include "/Users/phsh/Eigen/Dense"
#include "Eigen/Dense"
#include "Beams.hpp"
using namespace std;

#ifdef __cplusplus
    #include <cstdint>
#else
    #include <stdint.h>
    #include <stdbool.h>
#endif

#ifdef _WIN32
    #ifdef BUILD_CBMP
        #define EXPORT_SYMBOL __declspec(dllexport)
    #else
        #define EXPORT_SYMBOL __declspec(dllimport)
    #endif
#else
    #define EXPORT_SYMBOL
#endif

/*
typedef struct {
    int32_t beamtype;
    double E0;
    double k;
    double kz;
    double kt;
    double kt_by_kz;
    int order;
    double jones[4];
    double translation[3];
    double rotation[9];
    double w0;
//    std::complex<double> jones[2];
} BEAM;
*/


extern "C"
EXPORT_SYMBOL void grad_E_cc(double *position, double *polarisation, double kvec, double *gradEE);

extern "C"
EXPORT_SYMBOL void optical_force_array(double *array_of_particles, int number_of_particles, double dipole_radius, double* dipole_primitives, long* dipole_primitive_num, double* inv_polar, BEAM_COLLECTION* beam_collection, double* final_optical_forces);

extern "C"
EXPORT_SYMBOL void optical_force_torque_array(double *array_of_particles, int number_of_particles, double dipole_radius, double* dipole_primitives, long* dipole_primitive_num, double* inv_polar, BEAM_COLLECTION* beam_collection, double* final_optical_dipole_forces, double* final_optical_forces, double* final_optical_torques, double* final_optical_couples);


Eigen::MatrixXcd dipole_moment_array(Eigen::MatrixXd array_of_positions, int number_of_dipoles, double dipole_radius, int number_of_particles, long* dipole_primitive_num, Eigen::VectorXcd inverse_polars, BEAM_COLLECTION* beam_collection);

/*
extern "C"
EXPORT_SYMBOL void compute_field_gradients(double xx, double yy, double zz, BEAM *thisbeam, double *gradEE);

extern "C"
//EXPORT_SYMBOL std::complex<double> *general_bessel(double x, double y, double z, double E0, double kz, double kt, std::complex<double> *jones, int l, BEAM *thisbeam);
EXPORT_SYMBOL std::complex<double> *general_bessel(double x, double y, double z, BEAM *thisbeam);

extern "C"
EXPORT_SYMBOL std::complex<double> *general_bessel_gradient(double x, double y, double z, BEAM *thisbeam);

extern "C"
EXPORT_SYMBOL std::complex<double> *gaussian_xpol(double x, double y, double z, BEAM *thisbeam);

extern "C"
EXPORT_SYMBOL std::complex<double> *gaussian_xpol_gradient(double x, double y, double z, BEAM *thisbeam);

extern "C"
EXPORT_SYMBOL std::complex<double> *plane_wave(double x, double y, double z, BEAM *thisbeam);

extern "C"
EXPORT_SYMBOL std::complex<double> *plane_wave_gradient(double x, double y, double z, BEAM *thisbeam);

void plane_wave_fields(double x, double y, double z, BEAM *thisbeam, std::complex<double> *Epxyz);
void general_bessel_fields(double x, double y, double z, BEAM *thisbeam, std::complex<double> *Epxyz);
void gaussian_barton5_fields(double x, double y, double z, BEAM *thisbeam, std::complex<double> *Epxyz);

void plane_wave_field_gradients(double x, double y, double z, BEAM *thisbeam, std::complex<double> gradEpxyz[][3]);
void general_bessel_field_gradients(double x, double y, double z, BEAM *thisbeam, std::complex<double> gradEpxyz[][3]);
void gaussian_barton5_field_gradients(double x, double y, double z, BEAM *thisbeam, std::complex<double> gradEpxyz[][3]);


double jnp(int order, double x);
*/
#endif

