//********************************************************************************
//
// Library of functions for calculations involving dipoles and optical forces.
//
// General Instructions:
// clang / g++-14 -std=c++14 -Wall -Wextra -pedantic -c -fPIC Dipoles.cpp -o Dipoles.o
// clang / g++-14 -shared Dipoles.o Beams.o -o Dipoles.dylib
//
// Linux Instructions:
// g++-14 -std=c++14 -O3 -fopenmp -Wall -Wextra -pedantic -c -fPIC Dipoles.cpp -o Dipoles.o
// g++-14 -O3 -fopenmp -fPIC -c Beams.cpp -o Beams.o
// g++-14 -O3 -fopenmp -shared Dipoles.o Beams.o -o libDipoles.so
// g++-14 -O3 -shared -o libBeams.so -fPIC -fopenmp Beams.cpp
//
//********************************************************************************
#include "Dipoles.hpp"
#include "Beams.hpp"
#include <vector>

//********************************************************************************
//  Start with calculation of the gradient of radiative dipole field as given
//  in Jackson, 3rd edition.
//********************************************************************************

void grad_E_cc(double *rvec, double *pvec, double kvec, double *gradEE) {
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


//======================================================================================
void optical_force_array(double *array_of_particles, int number_of_particles, double dipole_radius, double* dipole_primitives, long* dipole_primitive_num, double* inv_polar, BEAM_COLLECTION* beam_collection, double* final_optical_forces) {
    //
    // array _of_particles is (Np,3) list of positions
    // dipole_primitive is (Nd,3) list of positions
    //
    // Need to:
    // (0) change array_of_positions to array of particle positions
    // (1) generate a full list of dipole positions
    // (2) do the optical force calculation on the full list
    // (3) construct the forces on the individual particles
    //
    // A quick line to remap the inverse polars to complex variables.
    //
    Eigen::Map<Eigen::VectorXcd> inverse_polars((std::complex<double>*)(inv_polar), number_of_particles);
    //Eigen::VectorXcd inverse_polars(number_of_particles);
    int i, j, ij, ti, tj;
    int num_beams;
    double rvec[3], kvec;
    double pvec[6];
    double xx, yy, zz;
    int cumulative_counter;
    //for (i=0; i<number_of_particles; i++){
    //    inverse_polars(i) = inv_polar[2*i] + inv_polar[2*i+1]*1i;
    //}
    //std::cout<<array_of_particles[0]<<" "<<array_of_particles[1]<<" "<<array_of_particles[2]<<endl;
    //std::cout<<dipole_primitive[0]<<" "<<dipole_primitive[1]<<" "<<dipole_primitive[2]<<endl;
    // Here is section (0):
    //
    //number_of_particles = len(array_of_particles)
    //number_of_dipoles_in_primitive = len(dipole_primitive)
    //number_of_dipoles = number_of_particles*number_of_dipoles_in_primitive
    //int number_of_dipoles = number_of_particles*number_of_dipoles_in_primitive;
    //int dpn_start_indices[number_of_particles];
    std::vector<int> dpn_start_indices(number_of_particles);
    int number_of_dipoles = 0;
    for(int i=0; i<number_of_particles; i++) {
        dpn_start_indices[i] = number_of_dipoles;
        number_of_dipoles += dipole_primitive_num[i];
    }
    //std::cout<<number_of_dipoles<<std::endl;
    //
    // Here is section (1):
    //
    // These are positions for ALL dipoles:
    //array_of_positions = np.zeros((number_of_dipoles, 3))
    Eigen::MatrixXd array_of_positions(number_of_dipoles, 3);

    for (i=0; i<number_of_particles; i++){
        for (j=0; j<dipole_primitive_num[i]; j++){
            for (ij=0; ij<3; ij++){
                //array_of_positions(i*number_of_dipoles_in_primitive+j,ij) = array_of_particles[i*3+ij] + dipole_primitive[j*3+ij];
                array_of_positions(dpn_start_indices[i] +j,ij) = array_of_particles[i*3+ij] + dipole_primitives[3*(dpn_start_indices[i] +j) +ij ];
            }
        }
    }
    //std::cout<<"Position array: "<<array_of_positions<<endl;
    //
    // Here is section (2):
    //
    Eigen::MatrixXcd p_array(number_of_dipoles, 3);
    //std::cout<<"Check before"<<std::endl;

    //
    // Has been adjusted while working on optical_force_torque_array for shape,args method
    //
    p_array = dipole_moment_array(array_of_positions, number_of_dipoles, dipole_radius, number_of_particles, dipole_primitive_num, inverse_polars, beam_collection);
    //std::cout<<"Check after"<<std::endl;
    //std::cout<<"P array: "<<p_array<<endl;
    //# print(p_array)
    //
    // Having found the polarisation array, need to calculate the gradients of
    // the dipole fields.
    //
    //displacements_matrix = displacement_matrix(array_of_positions)
    //grad_matrix = np.zeros([number_of_dipoles, number_of_dipoles], dtype=object)
    Eigen::MatrixXcd grad_matrix_T(3*number_of_dipoles,3*number_of_dipoles);
    //
    // Working directly with the transpose as the un-transposed matrix is not needed.
    //
    //for i in range(number_of_dipoles):
    //for j in range(number_of_dipoles):
    //if i == j:
    //grad_matrix[i][j] = 0
    //else:
    //grad_matrix[i][j] = Dipoles.py_grad_E_cc(displacements_matrix[i][j], p_array[i], k)
    Eigen::Matrix3cd Gradiiblock=Eigen::Matrix3cd::Zero();
    Eigen::Matrix3cd Gradijblock;
    double gradEE[18]; // to hold 9 complex values
    kvec = beam_collection->BEAM_ARRAY[0].k; // (scalar) assuming same for all beams!
    for (i=0; i<number_of_dipoles; i++){
        ti = 3*i;
        pvec[0] = p_array(i,0).real();
        pvec[1] = p_array(i,0).imag();
        pvec[2] = p_array(i,1).real();
        pvec[3] = p_array(i,1).imag();
        pvec[4] = p_array(i,2).real();
        pvec[5] = p_array(i,2).imag();
        for (j=0; j<number_of_dipoles; j++){
            tj = 3*j;
            if (i==j){ // All zeros here
                grad_matrix_T.block<3,3>(ti,ti) = Gradiiblock;
            }
            else{ // Compute the matrix
                rvec[0] = array_of_positions(i,0) - array_of_positions(j,0);
                rvec[1] = array_of_positions(i,1) - array_of_positions(j,1);
                rvec[2] = array_of_positions(i,2) - array_of_positions(j,2);
                grad_E_cc(rvec, pvec, kvec, gradEE);
                Eigen::Map<Eigen::Matrix3cd> Gradijblock((std::complex<double>*)(gradEE), 3, 3);
                // Note Gradijblock is column-major so already transposed, so will need to
                // modify the force calculation!
                // Apparently all Eigen matrices are stored column-major by default so may
                // need a rethink on some of the working.
                // I'm wondering if we even need to store this huge matrix?
                grad_matrix_T.block<3,3>(tj,ti) = Gradijblock.transpose();
            }
        }
    }
            //grad_matrix_T = np.transpose(grad_matrix)
            //#print("Array of particles:",array_of_positions)
            //#print("Displacements of particles:",displacements_matrix)
            //#print("dipole vectors:",p_array)
            //#print("Gradient matrix:",grad_matrix)
            //#print("Gradient matrix T:",grad_matrix_T)
            
    //optical_force_matrix = np.zeros([number_of_dipoles, number_of_dipoles], dtype=object)
    Eigen::MatrixXd optical_force_array=Eigen::MatrixXd::Zero(number_of_dipoles,3);
    Eigen::Vector3cd one_polar;
    //for i in range(number_of_dipoles):
    //for j in range(number_of_dipoles):
    //if i == j:
    //optical_force_matrix[i][j] = np.zeros(3)
    //else:
    //optical_force_matrix[i][j] = optical_force(
    //                                           grad_matrix_T[i][j], p_array[i]  # TRANSPOSE INPUT!!!!
    //                                           )
    for (i=0; i<number_of_dipoles; i++){
        ti = 3*i;
        one_polar = p_array.row(i);
        for (j=0; j<number_of_dipoles; j++){
            tj = 3*j;
            if (i!=j){ // Compute the matrix
                Gradijblock = grad_matrix_T.block<3,3>(ti,tj);
                //std::cout<<i<<" "<<j<<" "<<Gradijblock<<endl;
                optical_force_array.row(i) += 0.5*(Gradijblock*one_polar).real();
                // Need to check the ordering of above - has it computed correctly? - Yes
            }
        }
    }
    
    //std::cout<<"Optical force array scat:"<<endl;
    //std::cout<<optical_force_array<<endl;
    //optical_force_array_scat = np.sum(optical_force_matrix, axis=1)

    //grad_E_inc = np.zeros(number_of_dipoles, dtype=object)
    //optical_force_array_inc = np.zeros(number_of_dipoles, dtype=object)
    //for i in range(number_of_dipoles):
    //gradE = np.zeros((3,3),dtype=np.complex128)
    //Beams.all_incident_field_gradients(array_of_positions[i], beam_collection, gradE)
    num_beams = beam_collection->beams;

    //for i in range(nn):
    //    Beams.compute_field_gradients(x, y, z, the_beams[i], dgradEE)
    //    for j in range(3):
    //        for l in range(3):
    //            gradEE[j][l] += complex(dgradEE[(j*3+l)*2],-dgradEE[(j*3+l)*2+1]) # conjugate
    double dgradEE[18];
    Eigen::Matrix3cd dgradEEmat, grad_E_inc;
    for (i=0; i<number_of_dipoles; i++){
        xx = array_of_positions(i,0);
        yy = array_of_positions(i,1);
        zz = array_of_positions(i,2);
        grad_E_inc = Eigen::Matrix3cd::Zero();
        for (j=0; j<num_beams; j++){
            compute_field_gradients(xx, yy, zz, &beam_collection->BEAM_ARRAY[j], dgradEE);
            Eigen::Map<Eigen::Matrix3cd> dgradEEmat((std::complex<double>*)(dgradEE), 3, 3);
            grad_E_inc += dgradEEmat.conjugate(); // conjugated!
        }
        one_polar = p_array.row(i);
        optical_force_array.row(i) += 0.5*(grad_E_inc*one_polar).real();
    }
    
//    grad_E_inc[i] = gradE
//#        grad_E_inc[i] = incident_field_gradient(beam, array_of_positions[i])
//    optical_force_array_inc[i] = optical_force(
//                                               np.transpose(grad_E_inc[i]), p_array[i]
//                                               )
//    optical_force_array_tot = optical_force_array_scat + optical_force_array_inc
 
    //
    // This is section (3): Summing the dipole forces for each particle.
    //
    cumulative_counter = 0;
    for (i=0; i<number_of_particles; i++){
        final_optical_forces[i*3] = 0.0;
        final_optical_forces[i*3+1] = 0.0;
        final_optical_forces[i*3+2] = 0.0;
        for (j=0; j<dipole_primitive_num[i]; j++){
            final_optical_forces[i*3]   += optical_force_array(cumulative_counter +j,0);
            final_optical_forces[i*3+1] += optical_force_array(cumulative_counter +j,1);
            final_optical_forces[i*3+2] += optical_force_array(cumulative_counter +j,2);
        }
        cumulative_counter += dipole_primitive_num[i];
    }
    //final_optical_forces = np.zeros(number_of_particles, dtype=object)
    //for i in range(number_of_particles):
    //final_optical_forces[i] = np.sum(optical_force_array_tot[i*number_of_dipoles_in_primitive:(i+1)*number_of_dipoles_in_primitive],axis=0)
//#print(optical_force_array_tot)
//#print(final_optical_forces)
//    return final_optical_forces

    return;
}

//======================================================================================
void optical_force_torque_array(double *array_of_particles, int number_of_particles, double dipole_radius, double* dipole_primitives, long* dipole_primitive_num, double* inv_polar, BEAM_COLLECTION* beam_collection, double* final_optical_forces, double* final_optical_torques, double* final_optical_couples){
    //
    // This version returns the optical torques as well as forces, splitting them into the r X F contribution
    // and the p X E contributions.
    //
    // array _of_particles is (Np,3) list of positions
    // dipole_primitive is (Nd,3) list of positions
    //
    // Need to:
    // (0) change array_of_positions to array of particle positions
    // (1) generate a full list of dipole positions
    // (2) do the optical force calculation on the full list
    // (3) construct the forces on the individual particles
    //
    // A quick line to remap the inverse polars to complex variables.
    //

    //std::cout << "Checking NEW" << std::endl;
    //for(int p=0; p<number_of_particles; p++) {
    //    std::cout << "p value=" << p << "=== " << dipole_primitive_num[2*p] << std::endl;
    //}
    //std::cout << "OVER AND OUT" << std::endl;


    Eigen::Map<Eigen::VectorXcd> inverse_polars((std::complex<double>*)(inv_polar), number_of_particles);
    Eigen::VectorXcd inverse_polarsconj(number_of_particles);
    inverse_polarsconj = inverse_polars.conjugate();
    //Eigen::VectorXcd inverse_polars(number_of_particles);
    int i, j, ij, ti, tj;
    int num_beams;
    double rvec[3], kvec;
    double pvec[6];
    double xx, yy, zz;
    int cumulative_counter;
    //
    // Here is section (0):
    //
    //number_of_particles*number_of_dipoles_in_primitive;
    //int dpn_start_indices[number_of_particles];
    std::vector<int> dpn_start_indices(number_of_particles);
    int number_of_dipoles = 0;
    //std::cout << "Checking start indices " << std::endl;
    for(int i=0; i<number_of_particles; i++) {
        dpn_start_indices[i] = number_of_dipoles;
        number_of_dipoles += dipole_primitive_num[i];
        //std::cout << "  startInd: " << dpn_start_indices[i] << std::endl;
        //std::cout << "  dipole_primitive_num: " << dipole_primitive_num[i] << std::endl;
    }
    //
    // Here is section (1):
    //
    // These are positions for ALL dipoles:
    Eigen::MatrixXd array_of_positions(number_of_dipoles, 3);

    for (i=0; i<number_of_particles; i++){
        for (j=0; j<dipole_primitive_num[i]; j++){
            for (ij=0; ij<3; ij++){
                //array_of_positions(i*number_of_dipoles_in_primitive+j,ij) = array_of_particles[i*3+ij] + dipole_primitive[j*3+ij];
                array_of_positions(dpn_start_indices[i] +j,ij) = array_of_particles[i*3+ij] + dipole_primitives[3*(dpn_start_indices[i] +j) +ij ];
            }
        }
    }
    //
    // Here is section (2):
    //
    Eigen::MatrixXcd p_array(number_of_dipoles, 3);

    p_array = dipole_moment_array(array_of_positions, number_of_dipoles, dipole_radius, number_of_particles, dipole_primitive_num, inverse_polars, beam_collection);
    
    //
    // Having found the polarisation array, need to calculate the gradients of
    // the dipole fields.
    //
    Eigen::MatrixXcd grad_matrix_T(3*number_of_dipoles,3*number_of_dipoles);
    //
    // Working directly with the transpose as the un-transposed matrix is not needed.
    //
    Eigen::Matrix3cd Gradiiblock=Eigen::Matrix3cd::Zero();
    Eigen::Matrix3cd Gradijblock;
    double gradEE[18]; // to hold 9 complex values
    kvec = beam_collection->BEAM_ARRAY[0].k; // (scalar) assuming same for all beams!
    for (i=0; i<number_of_dipoles; i++){
        ti = 3*i;
        pvec[0] = p_array(i,0).real();
        pvec[1] = p_array(i,0).imag();
        pvec[2] = p_array(i,1).real();
        pvec[3] = p_array(i,1).imag();
        pvec[4] = p_array(i,2).real();
        pvec[5] = p_array(i,2).imag();
        for (j=0; j<number_of_dipoles; j++){
            tj = 3*j;
            if (i==j){ // All zeros here
                grad_matrix_T.block<3,3>(ti,ti) = Gradiiblock;
            }
            else{ // Compute the matrix
                rvec[0] = array_of_positions(i,0) - array_of_positions(j,0);
                rvec[1] = array_of_positions(i,1) - array_of_positions(j,1);
                rvec[2] = array_of_positions(i,2) - array_of_positions(j,2);
                grad_E_cc(rvec, pvec, kvec, gradEE);
                Eigen::Map<Eigen::Matrix3cd> Gradijblock((std::complex<double>*)(gradEE), 3, 3);
                // Note Gradijblock is column-major so already transposed, so will need to
                // modify the force calculation!
                // Apparently all Eigen matrices are stored column-major by default so may
                // need a rethink on some of the working.
                // I'm wondering if we even need to store this huge matrix?
                grad_matrix_T.block<3,3>(tj,ti) = Gradijblock.transpose();
            }
        }
    }
            
    Eigen::MatrixXd optical_force_array=Eigen::MatrixXd::Zero(number_of_dipoles,3);
    Eigen::Vector3cd one_polar;

    for (i=0; i<number_of_dipoles; i++){
        ti = 3*i;
        one_polar = p_array.row(i);
        for (j=0; j<number_of_dipoles; j++){
            tj = 3*j;
            if (i!=j){ // Compute the matrix
                Gradijblock = grad_matrix_T.block<3,3>(ti,tj);
                optical_force_array.row(i) += 0.5*(Gradijblock*one_polar).real();
                // Need to check the ordering of above - has it computed correctly? - Yes
            }
        }
    }
    
    num_beams = beam_collection->beams;

    double dgradEE[18];
    Eigen::Matrix3cd dgradEEmat, grad_E_inc;
    for (i=0; i<number_of_dipoles; i++){
        xx = array_of_positions(i,0);
        yy = array_of_positions(i,1);
        zz = array_of_positions(i,2);
        grad_E_inc = Eigen::Matrix3cd::Zero();
        for (j=0; j<num_beams; j++){
            compute_field_gradients(xx, yy, zz, &beam_collection->BEAM_ARRAY[j], dgradEE);
            Eigen::Map<Eigen::Matrix3cd> dgradEEmat((std::complex<double>*)(dgradEE), 3, 3);
            grad_E_inc += dgradEEmat.conjugate(); // conjugated!
        }
        one_polar = p_array.row(i);
        optical_force_array.row(i) += 0.5*(grad_E_inc*one_polar).real();
    }
    
 
    //
    // This is section (3): Summing the dipole forces for each particle.
    //
    cumulative_counter = 0;
    for (i=0; i<number_of_particles; i++){
        final_optical_forces[i*3] = 0.0;
        final_optical_forces[i*3+1] = 0.0;
        final_optical_forces[i*3+2] = 0.0;
        for (j=0; j<dipole_primitive_num[i]; j++){
            final_optical_forces[i*3]   += optical_force_array(cumulative_counter +j,0);
            final_optical_forces[i*3+1] += optical_force_array(cumulative_counter +j,1);
            final_optical_forces[i*3+2] += optical_force_array(cumulative_counter +j,2);
        }
        cumulative_counter += dipole_primitive_num[i];
    }
    //
    // Next the r X F contribution to torque:
    //
    //"""for i in range(number_of_particles):
    //    for j in range(number_of_dipoles_in_primitive):
    //        couples[i,0]+=dipole_primitive[j][1]*optical_force_array_tot[i*number_of_dipoles_in_primitive+j][2]-dipole_primitive[j][2]*optical_force_array_tot[i*number_of_dipoles_in_primitive+j][1]
    //        couples[i,1]+=dipole_primitive[j][2]*optical_force_array_tot[i*number_of_dipoles_in_primitive+j][0]-dipole_primitive[j][0]*optical_force_array_tot[i*number_of_dipoles_in_primitive+j][2]
    //        couples[i,2]+=dipole_primitive[j][0]*optical_force_array_tot[i*number_of_dipoles_in_primitive+j][1]-dipole_primitive[j][1]*optical_force_array_tot[i*number_of_dipoles_in_primitive+j][0]
    //"""
    cumulative_counter = 0;
    for (i=0; i<number_of_particles; i++){
        final_optical_torques[i*3] = 0.0;
        final_optical_torques[i*3+1] = 0.0;
        final_optical_torques[i*3+2] = 0.0;
        for (j=0; j<dipole_primitive_num[i]; j++){
            final_optical_torques[i*3]   += dipole_primitives[3*(cumulative_counter +j)+1] * optical_force_array(cumulative_counter +j,2) - dipole_primitives[3*(cumulative_counter +j)+2] * optical_force_array(cumulative_counter +j,1);
            final_optical_torques[i*3+1] += dipole_primitives[3*(cumulative_counter +j)+2] * optical_force_array(cumulative_counter +j,0) - dipole_primitives[3*(cumulative_counter +j)+0] * optical_force_array(cumulative_counter +j,2);
            final_optical_torques[i*3+2] += dipole_primitives[3*(cumulative_counter +j)+0] * optical_force_array(cumulative_counter +j,1) - dipole_primitives[3*(cumulative_counter +j)+1] * optical_force_array(cumulative_counter +j,0);
        }
        cumulative_counter += dipole_primitive_num[i];
    }

    //
    // Finally the p X E contribution to torque:
    // (NB, THIS PART WILL NEED CHANGING WHEN WE MOVE TO ANISOTROPIC CASE)
    //
    //couples = np.zeros((n_particles,3),dtype=np.double)
    //a0conj = a0.conjugate()
    //p_arrayconj = p_array.conjugate()
    //#print(p_array,p_arrayconj)
    //for i in range(number_of_particles):
    //    for j in range(number_of_dipoles_in_primitive):
    //        ij = i*number_of_dipoles_in_primitive+j
    //        #print(i,j,ij,p_array[ij])
    //        couples[i,0]+=0.5*np.real((p_array[ij][1]*p_arrayconj[ij][2]-p_array[ij][2]*p_arrayconj[ij][1])/a0conj[i])
    //        couples[i,1]+=0.5*np.real((p_array[ij][2]*p_arrayconj[ij][0]-p_array[ij][0]*p_arrayconj[ij][2])/a0conj[i])
    //        couples[i,2]+=0.5*np.real((p_array[ij][0]*p_arrayconj[ij][1]-p_array[ij][1]*p_arrayconj[ij][0])/a0conj[i])

    Eigen::MatrixXcd p_arrayconj(number_of_dipoles,3);

    p_arrayconj = p_array.conjugate();
    
    cumulative_counter = 0;
    for (i=0; i<number_of_particles; i++){
        final_optical_couples[i*3] = 0.0;
        final_optical_couples[i*3+1] = 0.0;
        final_optical_couples[i*3+2] = 0.0;
        for (j=0; j<dipole_primitive_num[i]; j++){
            ij = cumulative_counter +j;
            final_optical_couples[i*3+0] += 0.5*((p_array(ij,1) * p_arrayconj(ij,2) - p_array(ij,2) * p_arrayconj(ij,1)) * inverse_polarsconj(i)).real();
            final_optical_couples[i*3+1] += 0.5*((p_array(ij,2) * p_arrayconj(ij,0) - p_array(ij,0) * p_arrayconj(ij,2)) * inverse_polarsconj(i)).real();
            final_optical_couples[i*3+2] += 0.5*((p_array(ij,0) * p_arrayconj(ij,1) - p_array(ij,1) * p_arrayconj(ij,0)) * inverse_polarsconj(i)).real();
        }
        cumulative_counter += dipole_primitive_num[i];
    }
    
    return;
}


//======================================================================================
Eigen::MatrixXcd dipole_moment_array(Eigen::MatrixXd array_of_positions, int number_of_dipoles, double dipole_radius, int number_of_particles, long* dipole_primitive_num, Eigen::VectorXcd inverse_polars, BEAM_COLLECTION* beam_collection){
    //
    // array_of_positions contains all positions of dipoles in NdNp x 3 list.
    // number_of_dipoles is total across all particles.
    // number_of_dipoles_in_primitive is assuming same dipole number in every particle.
    //
    int i,j,ii,ti,tj;
    int num_beams;
    double x,y,z,r,x2,y2,z2,r2,xr2,yr2,zr2,k,k2;
    double xx, yy, zz;
    std::complex<double> ikr, pref, func1;
    //int dipole_primitive_num_summed[number_of_particles];
    std::vector<int> dipole_primitive_num_summed(number_of_particles);
    int dipole_summed_counter = 0;
/*        list_of_displacements = [u - v for u, v in it.combinations(array_of_positions, 2)]
        number_of_displacements = len(list_of_displacements)
        array_of_displacements = np.zeros(number_of_displacements, dtype=object)

        for i in range(number_of_displacements):
            array_of_displacements[i] = list_of_displacements[i]
        array_of_distances = np.array([np.linalg.norm(w) for w in array_of_displacements])

        number_of_dipoles = len(array_of_positions)
        #print("y",number_of_dipoles)
*/
    // Above section makes list of all vector displacements and distances.  These can be
    // computed as needed while the Aij matrix is being filled.
    //

    /*    A_matrix = np.zeros([number_of_dipoles, number_of_dipoles], dtype=object)
        Ajk_array = np.zeros(
            number_of_displacements, dtype=object
        )  # initialize array to store A_jk matrices
        Ajj_array = np.zeros(
            number_of_dipoles, dtype=object
        )  # initialize array to store A_jj matrices
        iu = np.triu_indices(number_of_dipoles, 1)
        di = np.diag_indices(number_of_dipoles)
     */
    //std::cout<<"In dipole moment function"<<std::endl;
    Eigen::MatrixXcd A_matrix(3*number_of_dipoles, 3*number_of_dipoles);
    //    k = 2*np.pi/wavelength
    k = beam_collection->BEAM_ARRAY[0].k; // assuming same for all beams!
    k2 = k*k;
    /*    for i in range(
            number_of_displacements
        ):  # for loop that goes over every x,y,z value of displacement
            A = np.zeros([3, 3], dtype=complex)
            Ajk_array[i] = Ajk(A,
                array_of_displacements[i][0],
                array_of_displacements[i][1],
                array_of_displacements[i][2],
                array_of_distances[i],k
            )
        for i in range(number_of_dipoles):  # creates D_jj matrices
            ii = i//number_of_dipoles_in_primitive
            A = np.zeros([3, 3], dtype=complex)
            Ajj_array[i] = Ajj(A,polarizability[ii])
    A_matrix[iu] = Ajk_array
    A_matrix.T[iu] = A_matrix[iu]
    A_matrix[di] = Ajj_array
    temporary_array = np.zeros(number_of_dipoles, dtype=object)
    for i in range(number_of_dipoles):
        temporary_array[i] = np.concatenate(A_matrix[i])
    A = np.hstack(temporary_array)
#    print("A matrix",A)
     */
    //
    // Need to fill the A matrix in 3x3 blocks
    //
    Eigen::Matrix3cd Aiiblock=Eigen::Matrix3cd::Zero();
    Eigen::Matrix3cd Aijblock;

    for(i=0; i<number_of_particles; i++) {
        dipole_summed_counter += dipole_primitive_num[i];
        dipole_primitive_num_summed[i] = dipole_summed_counter;
    }
    ii=0;
    for (i=0; i<number_of_dipoles; i++){

        // Find which particle the ith dipole belongs to
        if(ii >= dipole_primitive_num_summed[ii]) {
            ii++;
        }
        //ii = i/number_of_dipoles_in_primitive; // INTEGER DIVISION NEEDED

        ti = 3*i;
        for (j=0; j<number_of_dipoles; j++){
            tj = 3*j;
            if (i==j){
                Aiiblock(0,0) = inverse_polars(ii);
                Aiiblock(1,1) = inverse_polars(ii);
                Aiiblock(2,2) = inverse_polars(ii);
                A_matrix.block<3,3>(ti,ti) = Aiiblock;
            }
            else if (j>i){ // only calculate a triangle
                x = array_of_positions(j,0) - array_of_positions(i,0);
                y = array_of_positions(j,1) - array_of_positions(i,1);
                z = array_of_positions(j,2) - array_of_positions(i,2);
                x2 = x*x;
                y2 = y*y;
                z2 = z*z;
                r2 = x2+y2+z2;
                xr2 = x2/r2;
                yr2 = y2/r2;
                zr2 = z2/r2;
                r = sqrt(r2);
                ikr = 1i*k*r;
                pref = (exp(ikr)/r) / (4*M_PI*EPS0);
                func1 = (k2 + (3.0*(ikr-1.0)/r2))/r2;
                Aijblock(0,1) = x*y*func1;//(x, y, r,k)
                Aijblock(1,0) = Aijblock(0,1);
                Aijblock(0,2) = x*z*func1;//(x, z, r,k)
                Aijblock(2,0) = Aijblock(0,2);
                Aijblock(1,2) = y*z*func1;//(y, z, r,k)
                Aijblock(2,1) = Aijblock(1,2);
                //func2a = k2 * (a2/r2 - 1);
                //func2b = ((ikr - 1)/r2) * (3*a2/r2 - 1);
                Aijblock(0,0) = k2*(xr2 - 1) + ((ikr - 1.0)/r2)*(3*xr2 - 1);//(x, r,k);
                Aijblock(1,1) = k2*(yr2 - 1) + ((ikr - 1.0)/r2)*(3*yr2 - 1);//(y, r,k);
                Aijblock(2,2) = k2*(zr2 - 1) + ((ikr - 1.0)/r2)*(3*zr2 - 1);//(z, r,k);
                A_matrix.block<3,3>(ti,tj) = Aijblock*pref;
                A_matrix.block<3,3>(tj,ti) = Aijblock*pref;
            }
        }
    }
    /*
    for (i=0; i<number_of_dipoles; i++){
        for (j=0; j<number_of_dipoles; j++){
            std::cout<<"Block ["<<i<<","<<j<<"]:"<<endl;
            std::cout<<A_matrix.block<3,3>(3*i,3*j)<<std::endl;
        }
    }
     */
//    std::cout<<"Block[0,0]:"<<endl;
//    std::cout<<A_matrix.block<3,3>(0,0)<<std::endl;
//    std::cout<<"Block[0,1]:"<<endl;
//    std::cout<<A_matrix.block<3,3>(0,3)<<std::endl;
//    std::cout<<"Block[1,0]:"<<endl;
//    std::cout<<A_matrix.block<3,3>(3,0)<<std::endl;
//    std::cout<<"Block[2,3]:"<<endl;
//    std::cout<<A_matrix.block<3,3>(6,9)<<std::endl;
    

    //# initialize array of external electric field vectors
    //    E_array = np.zeros(number_of_dipoles, dtype=object)
    //    E = np.zeros((number_of_dipoles,3),dtype=np.complex128)
    Eigen::VectorXcd E(3*number_of_dipoles);
    double Eblock[6];
    num_beams = beam_collection->beams;
    //std::cout<<"number of beams "<<num_beams<<std::endl;
    //    Beams.all_incident_fields_array(array_of_positions, number_of_dipoles, beam_collection, E)
    for (i=0; i<number_of_dipoles; i++){
        xx = array_of_positions(i,0);
        yy = array_of_positions(i,1);
        zz = array_of_positions(i,2);
        E(i*3) = 0.0;
        E(i*3+1) = 0.0;
        E(i*3+2) = 0.0;
        for (j=0; j<num_beams; j++){
            compute_fields(xx, yy, zz, &beam_collection->BEAM_ARRAY[j], Eblock);
            E(i*3) += Eblock[0] +1i*Eblock[1];
            E(i*3+1) += Eblock[2] +1i*Eblock[3];
            E(i*3+2) += Eblock[4] +1i*Eblock[5];
        }
    }
    //std::cout<<"E values: "<<E<<endl;
    // E_array = E
    //    E = np.hstack(E_array)  # merges the array of E field vectors to form 3Nx1 vector

    //    P_list = np.hsplit(np.linalg.solve(A, E), number_of_dipoles)
    //std::cout<<"About to solve system"<<std::endl;
//    Eigen::VectorXcd P_list = A_matrix.fullPivLu().solve(E);
    Eigen::VectorXcd P_list = A_matrix.partialPivLu().solve(E);
    //std::cout<<"Solved system"<<std::endl;
    //std::cout<<"P values: "<<P_list<<endl;
/*
 P_array = np.zeros(number_of_dipoles, dtype=object)
        for i in range(number_of_dipoles):
            P_array[i] = P_list[i]

        return P_array
*/
    Eigen::MatrixXcd P_array(number_of_dipoles,3);
    for (i=0; i<number_of_dipoles; i++){
        P_array(i,0) = P_list(3*i);
        P_array(i,1) = P_list(3*i+1);
        P_array(i,2) = P_list(3*i+2);
    }
    return P_array;
}


//======================================================================================

/*
def Ajj(A,polarisability):
    """
    A_jj matrix; diagonal elements of big matrix of matrices
    """
    ajj = 1 / polarisability
    #print("Hey",ajj)
    #A = np.zeros([3, 3], dtype=complex)
    for i in range(3):
        A[i,i] = ajj
    #np.fill_diagonal(A, ajj)
    #print(A)
    return A
*/

/*
def Ajk(A,x, y, z, r,k):
    """
    A_jk matrix; off-diagonal elements of big matrix
    """
    #A = np.zeros([3, 3], dtype=complex)
    A[0][0] = func2(x, r,k)
    A[1][1] = func2(y, r,k)
    A[2][2] = func2(z, r,k)
    A[0][1] = func1(x, y, r,k)
    A[1][0] = A[0][1]
    A[0][2] = func1(x, z, r,k)
    A[2][0] = A[0][2]
    A[1][2] = func1(y, z, r,k)
    A[2][1] = A[1][2]
    # *(1/(4*np.pi*8.85e-12))

    return ((cmath.exp(1j * k * r)) / (r)) * A * (1 / (4 * np.pi * 8.85e-12))
*/
 
/*
@nb.njit()
def func1(a, b, r,k):
    """
    Non-diagonal elements of matrix A_jk
    """
    C = ((a * b) / (r ** 2)) * (k ** 2 + (3 * ((1j * k * r) - 1) / (r ** 2)))

    return C
*/

/*
@nb.njit()
def func2(a, r,k):
    """
    Diagonal elements of matrix A_jk
    """
    #print("k: ",k," a: ",a," r: ",r)
    return (k ** 2) * (((a ** 2) / (r ** 2)) - 1) + (((1j * k * r) - 1) / (r ** 2)) * (
        ((3 * (a ** 2)) / (r ** 2)) - 1
    )
*/

/*
@nb.njit()
def optical_force(grad, p):
    """Calulates the optical force from the TRANSPOSE of the gradient of the  field"""
    Force = np.zeros(3)
    Force[0] = (1 / 2) * np.real(
        p[0] * grad[0, 0] + p[1] * grad[0, 1] + p[2] * grad[0, 2]
    )
    Force[1] = (1 / 2) * np.real(
        p[0] * grad[1, 0] + p[1] * grad[1, 1] + p[2] * grad[1, 2]
    )
    Force[2] = (1 / 2) * np.real(
        p[0] * grad[2, 0] + p[1] * grad[2, 1] + p[2] * grad[2, 2]
    )
    return Force
*/
