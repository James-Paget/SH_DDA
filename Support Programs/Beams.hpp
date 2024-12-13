// Header for the Beams library

#ifndef BEAMS_HPP
#define BEAMS_HPP

#define BEAMTYPE_PLANE 0
#define BEAMTYPE_GAUSS_BARTON5 1
#define BEAMTYPE_GAUSS_CSP 2
#define BEAMTYPE_BESSEL 3
#define BEAMTYPE_LAGUERRE_GAUSSIAN 4

#define TINY 1e-12

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
    int gouy;
    int numpoints;
//    std::complex<double> jones[2];
} BEAM;

extern "C"
EXPORT_SYMBOL void compute_fields(double xx, double yy, double zz, BEAM *thisbeam, double *EE);

extern "C"
EXPORT_SYMBOL void compute_field_gradients(double xx, double yy, double zz, BEAM *thisbeam, double *gradEE);

//extern "C"
//EXPORT_SYMBOL std::complex<double> *general_bessel(double x, double y, double z, double E0, double kz, double kt, std::complex<double> *jones, int l, BEAM *thisbeam);
//EXPORT_SYMBOL std::complex<double> *general_bessel(double x, double y, double z, BEAM *thisbeam);

//extern "C"
//EXPORT_SYMBOL std::complex<double> *general_bessel_gradient(double x, double y, double z, BEAM *thisbeam);

//extern "C"
//EXPORT_SYMBOL std::complex<double> *gaussian_xpol(double x, double y, double z, BEAM *thisbeam);

//extern "C"
//EXPORT_SYMBOL std::complex<double> *gaussian_xpol_gradient(double x, double y, double z, BEAM *thisbeam);

//extern "C"
//EXPORT_SYMBOL std::complex<double> *plane_wave(double x, double y, double z, BEAM *thisbeam);

//extern "C"
//EXPORT_SYMBOL std::complex<double> *plane_wave_gradient(double x, double y, double z, BEAM *thisbeam);

void plane_wave_fields(double x, double y, double z, BEAM *thisbeam, std::complex<double> *Epxyz);
void general_bessel_fields(double x, double y, double z, BEAM *thisbeam, std::complex<double> *Epxyz);
void gaussian_barton5_fields(double x, double y, double z, BEAM *thisbeam, std::complex<double> *Epxyz);
void gaussian_csp_fields(double x, double y, double z, BEAM *thisbeam, std::complex<double> *Epxyz);
void laguerre_gaussian_fields(double x, double y, double z, BEAM *thisbeam, std::complex<double> *Epxyz);

void plane_wave_field_gradients(double x, double y, double z, BEAM *thisbeam, std::complex<double> gradEpxyz[][3]);
void general_bessel_field_gradients(double x, double y, double z, BEAM *thisbeam, std::complex<double> gradEpxyz[][3]);
void gaussian_barton5_field_gradients(double x, double y, double z, BEAM *thisbeam, std::complex<double> gradEpxyz[][3]);
void gaussian_csp_field_gradients(double x, double y, double z, BEAM *thisbeam, std::complex<double> gradEpxyz[][3]);
void laguerre_gaussian_field_gradients_fd(double x, double y, double z, BEAM *thisbeam, std::complex<double> gradEpxyz[][3]);


double eek(double kappa, double kk, double E0, double zR, int order, int gouy);
double jnp(int order, double x);
std::complex<double> ff(std::complex<double> x);
std::complex<double> gg(std::complex<double> x);

#endif
