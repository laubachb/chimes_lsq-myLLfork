#pragma once

// Shared between chimes_lsq_gpu.cu and chimes_lsq_gpu_host.cpp

#define LSQ_MAX_POLY_ORDER 24
#define LSQ_MAX_CLUSTER_PAIRS 6

struct LSQPairParams {
    int    snum;
    int    vstart;
    double s_minim;
    double s_maxim;
    double x_diff;
    double x_avg;
    double lambda;
    int    cheby_type;   // 0=MORSE  1=INVRSE_R  2=NONE
    int    fcut_type;    // 0=CUBIC  1=TERSOFF
    int    fcut_power;
    double fcut_offset;
};

struct LSQClusterGpu {
    int    vstart;          // first param index for this cluster type in A
    int    n_terms;         // N_ALLOWED_POWERS
    int    term_offset;     // offset into flattened LSQPowerTermGpu array
    int    npairs;          // 3 (trip) or 6 (quad)
    int    fcut_type;
    int    fcut_power;
    double fcut_offset;
    double s_minim[LSQ_MAX_CLUSTER_PAIRS];
    double s_maxim[LSQ_MAX_CLUSTER_PAIRS];
    double x_diff[LSQ_MAX_CLUSTER_PAIRS];
    double x_avg[LSQ_MAX_CLUSTER_PAIRS];
};

struct LSQPowerTermGpu {
    int param_offset;       // PARAM_INDICES[i] (added to cluster.vstart)
    int pow[LSQ_MAX_CLUSTER_PAIRS];  // ALLOWED_POWERS[i][*] in cluster pair order
};

struct LSQTripGpu {
    int    a1, a2, a3;
    int    cluster_idx;
    int    pair_index[3];
    int    pt[3];           // FF_2BODY pair-type indices for ij, ik, jk
    double lambda[3];
    int    cheby_type[3];
    int    snum[3];
    double rlen[3];
    double rab[9];
};

struct LSQQuadGpu {
    int    a1, a2, a3, a4;
    int    cluster_idx;
    int    pair_index[6];
    int    pt[6];
    double lambda[6];
    int    cheby_type[6];
    int    snum[6];
    double rlen[6];
    double rab[18];
};

#ifdef USE_CUDA

bool lsq_gpu_is_initialized();

void lsq_gpu_begin_frame_accum(int nparams, int natoms);

bool lsq_gpu_launch_deriv_2b(
    int npairs, int nparams, int natoms, int n_pair_types,
    const int *h_a1, const int *h_a2, const int *h_ptype,
    const double *h_rlen, const double *h_rab,
    const LSQPairParams *h_pair_params,
    double perm_scale, double deriv_const, int fit_stress, int fit_energy);

bool lsq_gpu_launch_deriv_3b(
    int ntrips, int nparams, int natoms,
    const LSQTripGpu *h_trips,
    const LSQClusterGpu *h_clusters, int n_clusters,
    const LSQPowerTermGpu *h_power_terms, int n_power_terms,
    double perm_scale, double deriv_const, int fit_stress, int fit_energy);

bool lsq_gpu_launch_deriv_4b(
    int nquads, int nparams, int natoms,
    const LSQQuadGpu *h_quads,
    const LSQClusterGpu *h_clusters, int n_clusters,
    const LSQPowerTermGpu *h_power_terms, int n_power_terms,
    double perm_scale, double deriv_const, int fit_stress, int fit_energy);

bool lsq_gpu_finish_frame_accum(
    int nparams, int natoms,
    double *h_fx, double *h_fy, double *h_fz,
    double *h_stress_xx, double *h_stress_xy, double *h_stress_xz,
    double *h_stress_yy, double *h_stress_yz, double *h_stress_zz,
    double *h_frame_energies);

#endif
