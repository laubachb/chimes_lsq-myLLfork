/*
    ChIMES LSQ — GPU acceleration for A-matrix derivative construction
*/
#pragma once

#ifdef USE_CUDA

class Cheby;
class A_MAT;

struct CLUSTER_LIST;

void lsq_gpu_init(int device_id);
int lsq_gpu_device_for_rank(int rank);
bool lsq_gpu_available();
void lsq_gpu_finalize();

void lsq_gpu_set_batch_frames(int n);
void lsq_gpu_flush_batch();

// Compute 2/3/4-body Chebyshev derivatives on GPU. Returns false → use CPU ZCalc_Deriv.
bool lsq_gpu_deriv_cheby(Cheby &cheby, A_MAT &a_matrix,
                         CLUSTER_LIST &trips, CLUSTER_LIST &quads);

#endif // USE_CUDA
