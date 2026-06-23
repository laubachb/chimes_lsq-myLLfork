#pragma once

// Shared between chimes_lsq_gpu.cu and chimes_lsq_gpu_host.cpp

#define LSQ_MAX_POLY_ORDER 24
#define LSQ_MAX_CLUSTER_PAIRS 6

// Must match MAX_ATOM_TYPES in functions.h
#define LSQ_MAX_ATOM_TYPES 10
#define LSQ_MAX_ATOM_TYPES2 (LSQ_MAX_ATOM_TYPES * LSQ_MAX_ATOM_TYPES)
#define LSQ_MAX_TRIP_MAP   (LSQ_MAX_ATOM_TYPES * LSQ_MAX_ATOM_TYPES * LSQ_MAX_ATOM_TYPES)
#define LSQ_MAX_QUAD_MAP   (LSQ_MAX_TRIP_MAP * LSQ_MAX_ATOM_TYPES)

struct LSQBoxGpu {
    double hmat[9];
    double invr_hmat[9];
};

struct LSQFrameGpu {
    int    natoms;
    int    nall;
    int    natmtyp;
    int    use_mic;          // natoms == nall
    int    cheby_fix_type;   // 0=ZERO_DERIV  1=CONSTANT_DERIV  2=SMOOTH
    double rcut_2b;
    double rcut_3b;
    double rcut_4b;
    double rcut_pad;
    double perm_2b;
    double perm_3b;
    double perm_4b;
    double cheby_smooth_distance;
};

struct LSQPairParams {
    int    snum;
    int    snum_3b;
    int    snum_4b;
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

bool lsq_gpu_begin_frame_accum(int nparams, int natoms);

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

// Tier-1 pipeline: static cache, GPU neighbor enumeration, batched sync
void lsq_gpu_set_batch_frames(int n);
int  lsq_gpu_batch_frames();
void lsq_gpu_flush_batch();

bool lsq_gpu_upload_static_tables(
    int natmtyp, int n_pair_types,
    const int *h_ipm,
    const LSQPairParams *h_pair_params,
    int use_3b, const int *h_trip_map, const int *h_trip_pair_idx,
    int n_trip_clusters, const LSQClusterGpu *h_trip_clusters,
    int n_trip_terms, const LSQPowerTermGpu *h_trip_terms,
    int use_4b, const int *h_quad_map, const int *h_quad_pair_idx,
    int n_quad_clusters, const LSQClusterGpu *h_quad_clusters,
    int n_quad_terms, const LSQPowerTermGpu *h_quad_terms);

bool lsq_gpu_upload_frame(
    const double *h_coords, int nall,
    const int *h_parent, const int *h_atom_type_idx,
    const LSQBoxGpu *box, const LSQFrameGpu *frame);

bool lsq_gpu_enumerate_2b(int *out_npairs);
bool lsq_gpu_enumerate_3b(int *out_ntrips);
bool lsq_gpu_enumerate_4b(int *out_nquads);

bool lsq_gpu_launch_deriv_2b_device(
    int npairs, int nparams, int natoms,
    double perm_scale, double deriv_const, int fit_stress, int fit_energy);

bool lsq_gpu_launch_deriv_3b_device(
    int ntrips, int nparams, int natoms,
    double perm_scale, double deriv_const, int fit_stress, int fit_energy);

bool lsq_gpu_launch_deriv_4b_device(
    int nquads, int nparams, int natoms,
    double perm_scale, double deriv_const, int fit_stress, int fit_energy);

#endif
