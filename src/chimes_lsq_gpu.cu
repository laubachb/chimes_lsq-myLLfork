/*
    ChIMES LSQ — CUDA kernels for ∂F/∂θ (2/3/4-body Chebyshev derivatives)
*/
#include "chimes_lsq_gpu.cuh"
#include "chimes_lsq_gpu_types.h"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define MAX_POLY_ORDER LSQ_MAX_POLY_ORDER
#define CUDA_CHECK(call) do {                                                   \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error at %s:%d — %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                    \
        return false;                                                            \
    }                                                                           \
} while (0)

struct LSQGpuState {
    bool initialized;
    int device_id;
    int max_pairs, max_trips, max_quads;
    int nparams, natoms, n_pair_types;
    int batch_frames;
    int batch_pending;
    bool static_uploaded;

    int *d_a1, *d_a2, *d_ptype;
    double *d_rlen, *d_rab;
    LSQTripGpu *d_trips;
    LSQQuadGpu *d_quads;
    LSQClusterGpu *d_trip_clusters, *d_quad_clusters;
    LSQPowerTermGpu *d_trip_power_terms, *d_quad_power_terms;
    int n_trip_clusters, n_quad_clusters;
    int n_trip_power_terms, n_quad_power_terms;
    LSQPairParams *d_pair_params;

    double *d_fx, *d_fy, *d_fz;
    double *d_stress_xx, *d_stress_xy, *d_stress_xz;
    double *d_stress_yy, *d_stress_yz, *d_stress_zz;
    double *d_frame_energies;

    // Frame geometry + lookup (Tier 1)
    double *d_coords;
    int    *d_parent;
    int    *d_atom_type_idx;
    int    *d_ipm;
    int    *d_trip_map;
    int    *d_trip_pair_idx;
    int    *d_quad_map;
    int    *d_quad_pair_idx;
    LSQBoxGpu  h_box;
    LSQFrameGpu h_frame;
    int max_coords;
    unsigned int *d_enum_counter;

    int fit_stress, fit_energy;
    int use_3b, use_4b;
};

static LSQGpuState g_gpu = {};

__device__ static void lsq_transform(double rlen, double x_diff, double x_avg,
                                     double lambda, int cheby_type,
                                     double &x, double &exprlen)
{
    switch (cheby_type) {
    case 0:
        exprlen = exp(-rlen / lambda);
        x = (exprlen - x_avg) / x_diff;
        break;
    case 1:
        exprlen = 0.0;
        x = (1.0 / rlen - x_avg) / x_diff;
        break;
    default:
        exprlen = 0.0;
        x = (rlen - x_avg) / x_diff;
        break;
    }
}

__device__ static double lsq_dx_dr(double xdiff, double rlen, double lambda,
                                   int cheby_type, double exprlen)
{
    switch (cheby_type) {
    case 0: return (-exprlen / lambda) / xdiff;
    case 1: return -1.0 / (rlen * rlen * xdiff);
    default: return 1.0 / xdiff;
    }
}

__device__ static void lsq_set_polys_edge(double rlen, double x_diff, double x_avg,
                                        double lambda, int cheby_type, int snum,
                                        double deriv_const,
                                        double *Tn, double *Tnd)
{
    double x = 0.0, exprlen = 0.0;
    lsq_transform(rlen, x_diff, x_avg, lambda, cheby_type, x, exprlen);

    Tn[0] = 1.0;  Tn[1] = x;
    Tnd[0] = 1.0; Tnd[1] = 2.0 * x;

    for (int i = 2; i <= snum; i++) {
        Tn[i]  = 2.0 * x * Tn[i-1]  - Tn[i-2];
        Tnd[i] = 2.0 * x * Tnd[i-1] - Tnd[i-2];
    }

    double dx_dr = deriv_const * lsq_dx_dr(x_diff, rlen, lambda, cheby_type, exprlen);
    for (int i = snum; i >= 1; i--)
        Tnd[i] = i * dx_dr * Tnd[i-1];
    Tnd[0] = 0.0;
}

__device__ static void lsq_fcut_edge(double rlen, double rmin, double rmax,
                                     int fcut_type, int fcut_power, double fcut_offset,
                                     double &fcut, double &fcutderiv)
{
    if (fcut_type == 0) {
        double f0 = 1.0 - rlen / rmax;
        fcut = 1.0;
        for (int p = 0; p < fcut_power; p++) fcut *= f0;
        fcutderiv = 1.0;
        for (int p = 0; p < fcut_power - 1; p++) fcutderiv *= f0;
        fcutderiv *= -1.0 * fcut_power / rmax;
    } else {
        const double PI = 3.141592653589793;
        double thresh = rmax * (1.0 - fcut_offset);
        if (rlen < thresh) {
            fcut = 1.0; fcutderiv = 0.0;
        } else if (rlen >= rmax) {
            fcut = 0.0; fcutderiv = 0.0;
        } else {
            double arg = (rlen - thresh) / (rmax - thresh) * PI + PI * 0.5;
            double fd  = PI / (rmax - thresh);
            fcut = 0.5 + 0.5 * sin(arg);
            fcutderiv = 0.5 * cos(arg) * fd;
        }
    }
    (void)rmin;
}

__device__ static bool lsq_proceed(double rlen, double rmin, double rmax, int fcut_type)
{
    if (fcut_type == 0 || fcut_type == 1) return rlen < rmax;
    return rlen > rmin && rlen < rmax;
}

__device__ static void accum_force_stress(
    int ia1, int ia2, int pidx, double coeff,
    double rx, double ry, double rz, double rlen,
    int nparams, int natoms, int fit_stress, int fit_energy, double ener_val,
    double *fx, double *fy, double *fz,
    double *sxx, double *sxy, double *sxz, double *syy, double *syz, double *szz,
    double *frame_energies)
{
    if (pidx < 0 || pidx >= nparams || rlen <= 0.0) return;
    double rinv = 1.0 / rlen;
    double dx = coeff * rx * rinv;
    double dy = coeff * ry * rinv;
    double dz = coeff * rz * rinv;

    atomicAdd(&fx[ia1 * nparams + pidx],  dx);
    atomicAdd(&fy[ia1 * nparams + pidx],  dy);
    atomicAdd(&fz[ia1 * nparams + pidx],  dz);
    atomicAdd(&fx[ia2 * nparams + pidx], -dx);
    atomicAdd(&fy[ia2 * nparams + pidx], -dy);
    atomicAdd(&fz[ia2 * nparams + pidx], -dz);

    if (fit_stress == 1) {
        atomicAdd(&sxx[pidx], -coeff * rx * rx * rinv);
        atomicAdd(&syy[pidx], -coeff * ry * ry * rinv);
        atomicAdd(&szz[pidx], -coeff * rz * rz * rinv);
    } else if (fit_stress == 2) {
        atomicAdd(&sxx[pidx], -coeff * rx * rx * rinv);
        atomicAdd(&sxy[pidx], -coeff * rx * ry * rinv);
        atomicAdd(&sxz[pidx], -coeff * rx * rz * rinv);
        atomicAdd(&syy[pidx], -coeff * ry * ry * rinv);
        atomicAdd(&syz[pidx], -coeff * ry * rz * rinv);
        atomicAdd(&szz[pidx], -coeff * rz * rz * rinv);
    }
    if (fit_energy)
        atomicAdd(&frame_energies[pidx], ener_val);
}

__global__ void kDeriv2B(int npairs,
                         const int *a1, const int *a2, const int *ptype,
                         const double *rlen_in, const double *rab,
                         const LSQPairParams *pair_params,
                         int nparams, int natoms,
                         double perm_scale, double deriv_const,
                         int fit_stress, int fit_energy,
                         double *fx, double *fy, double *fz,
                         double *sxx, double *sxy, double *sxz,
                         double *syy, double *syz, double *szz,
                         double *frame_energies)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= npairs) return;

    int pt = ptype[pid];
    const LSQPairParams &pp = pair_params[pt];
    double rlen = rlen_in[pid];
    if (rlen <= pp.s_minim || rlen >= pp.s_maxim) return;

    double Tn[MAX_POLY_ORDER + 1], Tnd[MAX_POLY_ORDER + 1];
    lsq_set_polys_edge(rlen, pp.x_diff, pp.x_avg, pp.lambda, pp.cheby_type,
                       pp.snum, deriv_const, Tn, Tnd);

    double fcut, fcutderiv;
    lsq_fcut_edge(rlen, pp.s_minim, pp.s_maxim, pp.fcut_type, pp.fcut_power,
                  pp.fcut_offset, fcut, fcutderiv);

    double rx = rab[pid * 3 + 0], ry = rab[pid * 3 + 1], rz = rab[pid * 3 + 2];
    int vs = pp.vstart, sn = pp.snum;

    for (int i = 0; i < sn; i++) {
        int pidx = vs + i;
        double tmp = perm_scale * (fcut * Tnd[i + 1] + fcutderiv * Tn[i + 1]);
        double ev = (fit_energy) ? perm_scale * fcut * Tn[i + 1] : 0.0;
        accum_force_stress(a1[pid], a2[pid], pidx, tmp, rx, ry, rz, rlen,
                           nparams, natoms, fit_stress, fit_energy, ev,
                           fx, fy, fz, sxx, sxy, sxz, syy, syz, szz, frame_energies);
    }
}

__global__ void kDeriv3B(int ntrips,
                         const LSQTripGpu *trips,
                         const LSQClusterGpu *clusters,
                         const LSQPowerTermGpu *power_terms,
                         int nparams, int natoms,
                         double perm_scale, double deriv_const,
                         int fit_stress, int fit_energy,
                         double *fx, double *fy, double *fz,
                         double *sxx, double *sxy, double *sxz,
                         double *syy, double *syz, double *szz,
                         double *frame_energies)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ntrips) return;

    const LSQTripGpu &tr = trips[tid];
    const LSQClusterGpu &cl = clusters[tr.cluster_idx];

    for (int e = 0; e < 3; e++) {
        int pi = tr.pair_index[e];
        if (!lsq_proceed(tr.rlen[e], cl.s_minim[pi], cl.s_maxim[pi], cl.fcut_type))
            return;
    }

    double Tn[3][MAX_POLY_ORDER + 1], Tnd[3][MAX_POLY_ORDER + 1];
    double fcut[3], fcutd[3];

    for (int e = 0; e < 3; e++) {
        int pi = tr.pair_index[e];
        lsq_set_polys_edge(tr.rlen[e], cl.x_diff[pi], cl.x_avg[pi],
                           tr.lambda[e], tr.cheby_type[e], tr.snum[e],
                           deriv_const, Tn[e], Tnd[e]);
        lsq_fcut_edge(tr.rlen[e], cl.s_minim[pi], cl.s_maxim[pi],
                      cl.fcut_type, cl.fcut_power, cl.fcut_offset,
                      fcut[e], fcutd[e]);
    }

    for (int ti = 0; ti < cl.n_terms; ti++) {
        const LSQPowerTermGpu &term = power_terms[cl.term_offset + ti];
        int pidx = cl.vstart + term.param_offset;

        int pow_ij = term.pow[tr.pair_index[0]];
        int pow_ik = term.pow[tr.pair_index[1]];
        int pow_jk = term.pow[tr.pair_index[2]];

        double deriv_ij = fcut[0] * Tnd[0][pow_ij] + fcutd[0] * Tn[0][pow_ij];
        double deriv_ik = fcut[1] * Tnd[1][pow_ik] + fcutd[1] * Tn[1][pow_ik];
        double deriv_jk = fcut[2] * Tnd[2][pow_jk] + fcutd[2] * Tn[2][pow_jk];

        double f_ij = perm_scale * (deriv_ij * fcut[1] * fcut[2] * Tn[1][pow_ik] * Tn[2][pow_jk]);
        double f_ik = perm_scale * (deriv_ik * fcut[0] * fcut[2] * Tn[0][pow_ij] * Tn[2][pow_jk]);
        double f_jk = perm_scale * (deriv_jk * fcut[0] * fcut[1] * Tn[0][pow_ij] * Tn[1][pow_ik]);

        double ev = 0.0;
        if (fit_energy)
            ev = perm_scale * fcut[0] * fcut[1] * fcut[2] * Tn[0][pow_ij] * Tn[1][pow_ik] * Tn[2][pow_jk];

        accum_force_stress(tr.a1, tr.a2, pidx, f_ij,
                           tr.rab[0], tr.rab[1], tr.rab[2], tr.rlen[0],
                           nparams, natoms, fit_stress, 0, 0.0,
                           fx, fy, fz, sxx, sxy, sxz, syy, syz, szz, frame_energies);
        accum_force_stress(tr.a1, tr.a3, pidx, f_ik,
                           tr.rab[3], tr.rab[4], tr.rab[5], tr.rlen[1],
                           nparams, natoms, fit_stress, 0, 0.0,
                           fx, fy, fz, sxx, sxy, sxz, syy, syz, szz, frame_energies);
        accum_force_stress(tr.a2, tr.a3, pidx, f_jk,
                           tr.rab[6], tr.rab[7], tr.rab[8], tr.rlen[2],
                           nparams, natoms, fit_stress, fit_energy, ev,
                           fx, fy, fz, sxx, sxy, sxz, syy, syz, szz, frame_energies);
    }
}

__global__ void kDeriv4B(int nquads,
                         const LSQQuadGpu *quads,
                         const LSQClusterGpu *clusters,
                         const LSQPowerTermGpu *power_terms,
                         int nparams, int natoms,
                         double perm_scale, double deriv_const,
                         int fit_stress, int fit_energy,
                         double *fx, double *fy, double *fz,
                         double *sxx, double *sxy, double *sxz,
                         double *syy, double *syz, double *szz,
                         double *frame_energies)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= nquads) return;

    const LSQQuadGpu &qd = quads[qid];
    const LSQClusterGpu &cl = clusters[qd.cluster_idx];

    for (int e = 0; e < 6; e++) {
        int pi = qd.pair_index[e];
        if (!lsq_proceed(qd.rlen[e], cl.s_minim[pi], cl.s_maxim[pi], cl.fcut_type))
            return;
    }

    double Tn[6][MAX_POLY_ORDER + 1], Tnd[6][MAX_POLY_ORDER + 1];
    double fcut[6], fcutd[6];

    for (int e = 0; e < 6; e++) {
        int pi = qd.pair_index[e];
        lsq_set_polys_edge(qd.rlen[e], cl.x_diff[pi], cl.x_avg[pi],
                           qd.lambda[e], qd.cheby_type[e], qd.snum[e],
                           deriv_const, Tn[e], Tnd[e]);
        lsq_fcut_edge(qd.rlen[e], cl.s_minim[pi], cl.s_maxim[pi],
                      cl.fcut_type, cl.fcut_power, cl.fcut_offset,
                      fcut[e], fcutd[e]);
    }

    const int atom_a[6][2] = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
    };
    const int atom_idx[4] = {qd.a1, qd.a2, qd.a3, qd.a4};

    for (int ti = 0; ti < cl.n_terms; ti++) {
        const LSQPowerTermGpu &term = power_terms[cl.term_offset + ti];
        int pidx = cl.vstart + term.param_offset;

        int pow[6];
        for (int e = 0; e < 6; e++)
            pow[e] = term.pow[qd.pair_index[e]];

        double deriv[6];
        for (int e = 0; e < 6; e++)
            deriv[e] = perm_scale * (fcut[e] * Tnd[e][pow[e]] + fcutd[e] * Tn[e][pow[e]]);

        double fwc[6];
        for (int e = 0; e < 6; e++) {
            fwc[e] = deriv[e];
            for (int o = 0; o < 6; o++)
                if (o != e) fwc[e] *= fcut[o] * Tn[o][pow[o]];
        }

        double ev = perm_scale;
        if (fit_energy) {
            for (int e = 0; e < 6; e++) ev *= fcut[e] * Tn[e][pow[e]];
        } else {
            ev = 0.0;
        }

        for (int e = 0; e < 6; e++) {
            int ia1 = atom_idx[atom_a[e][0]];
            int ia2 = atom_idx[atom_a[e][1]];
            double ener_part = (e == 0 && fit_energy) ? ev : 0.0;
            accum_force_stress(ia1, ia2, pidx, fwc[e],
                               qd.rab[e*3+0], qd.rab[e*3+1], qd.rab[e*3+2], qd.rlen[e],
                               nparams, natoms, fit_stress, (e == 0) ? fit_energy : 0, ener_part,
                               fx, fy, fz, sxx, sxy, sxz, syy, syz, szz, frame_energies);
        }
    }
}

// --- Tier 1: GPU geometry, neighbor enumeration, static cache ---

__device__ static void lsq_get_dist(const double *coords, const LSQBoxGpu *box,
                                    int a1, int a2, int use_mic,
                                    double &rx, double &ry, double &rz, double &rlen)
{
    double x1 = coords[a1 * 3 + 0], y1 = coords[a1 * 3 + 1], z1 = coords[a1 * 3 + 2];
    double x2 = coords[a2 * 3 + 0], y2 = coords[a2 * 3 + 1], z2 = coords[a2 * 3 + 2];

    double s1x = box->invr_hmat[0]*x1 + box->invr_hmat[1]*y1 + box->invr_hmat[2]*z1;
    double s1y = box->invr_hmat[3]*x1 + box->invr_hmat[4]*y1 + box->invr_hmat[5]*z1;
    double s1z = box->invr_hmat[6]*x1 + box->invr_hmat[7]*y1 + box->invr_hmat[8]*z1;
    double s2x = box->invr_hmat[0]*x2 + box->invr_hmat[1]*y2 + box->invr_hmat[2]*z2;
    double s2y = box->invr_hmat[3]*x2 + box->invr_hmat[4]*y2 + box->invr_hmat[5]*z2;
    double s2z = box->invr_hmat[6]*x2 + box->invr_hmat[7]*y2 + box->invr_hmat[8]*z2;

    double tx = s2x - s1x, ty = s2y - s1y, tz = s2z - s1z;
    if (use_mic) {
        tx -= round(tx);
        ty -= round(ty);
        tz -= round(tz);
    }
    rx = box->hmat[0]*tx + box->hmat[1]*ty + box->hmat[2]*tz;
    ry = box->hmat[3]*tx + box->hmat[4]*ty + box->hmat[5]*tz;
    rz = box->hmat[6]*tx + box->hmat[7]*ty + box->hmat[8]*tz;
    rlen = sqrt(rx*rx + ry*ry + rz*rz);
}

__device__ static int lsq_cluster_id3(int t0, int t1, int t2, int max_types)
{
    return (t0 + 1) + (t1 + 1) * max_types + (t2 + 1) * max_types * max_types;
}

__device__ static int lsq_cluster_id4(int t0, int t1, int t2, int t3, int max_types)
{
    return lsq_cluster_id3(t0, t1, t2, max_types) + (t3 + 1) * max_types * max_types * max_types;
}

__global__ void kEnum2B(const double *coords, const int *parent, const int *atype,
                        const int *ipm, const LSQPairParams *pp,
                        LSQBoxGpu box, LSQFrameGpu frame,
                        int *o_a1, int *o_a2, int *o_pt, double *o_rlen, double *o_rab,
                        unsigned int *counter, int cap)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nall = frame.nall;
    int natoms = frame.natoms;
    int jobs = natoms * nall;
    if (tid >= jobs) return;

    int a1 = tid / nall;
    int a2 = tid % nall;
    if (a2 == a1) return;

    double rx, ry, rz, rlen;
    lsq_get_dist(coords, &box, a1, a2, frame.use_mic, rx, ry, rz, rlen);
    if (rlen >= frame.rcut_2b + frame.rcut_pad) return;

    int t1 = atype[parent[a1]];
    int t2 = atype[parent[a2]];
    int pt = ipm[t1 * frame.natmtyp + t2];
    const LSQPairParams &p = pp[pt];
    if (rlen <= p.s_minim || rlen >= p.s_maxim) return;

    unsigned int slot = atomicAdd(counter, 1u);
    if (slot >= (unsigned)cap) return;
    o_a1[slot] = a1;
    o_a2[slot] = parent[a2];
    o_pt[slot] = pt;
    o_rlen[slot] = rlen;
    o_rab[slot * 3 + 0] = rx;
    o_rab[slot * 3 + 1] = ry;
    o_rab[slot * 3 + 2] = rz;
}

__global__ void kEnum3B(const double *coords, const int *parent, const int *atype,
                        const int *ipm, const LSQPairParams *pp,
                        const int *trip_map, const int *trip_pair_idx,
                        const LSQClusterGpu *clusters,
                        LSQBoxGpu box, LSQFrameGpu frame,
                        LSQTripGpu *out, unsigned int *counter, int cap)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nall = frame.nall;
    int natoms = frame.natoms;
    long long jobs = (long long)natoms * nall * nall;
    if ((long long)tid >= jobs) return;

    int a1 = tid / (nall * nall);
    int rem = tid % (nall * nall);
    int a2 = rem / nall;
    int a3 = rem % nall;
    if (a2 == a1 || a3 == a1 || a3 == a2) return;
    if (frame.perm_3b == 1.0 && parent[a2] > parent[a3]) return;

    double rlen[3], rab[9];
    lsq_get_dist(coords, &box, a1, a2, frame.use_mic, rab[0], rab[1], rab[2], rlen[0]);
    if (rlen[0] >= frame.rcut_3b + frame.rcut_pad) return;
    lsq_get_dist(coords, &box, a1, a3, frame.use_mic, rab[3], rab[4], rab[5], rlen[1]);
    if (rlen[1] >= frame.rcut_3b + frame.rcut_pad) return;
    lsq_get_dist(coords, &box, a2, a3, frame.use_mic, rab[6], rab[7], rab[8], rlen[2]);
    if (rlen[2] >= frame.rcut_3b + frame.rcut_pad) return;

    int t0 = atype[parent[a1]], t1 = atype[parent[a2]], t2 = atype[parent[a3]];
    int cmap = lsq_cluster_id3(t0, t1, t2, LSQ_MAX_ATOM_TYPES);
    if (cmap < 0 || cmap >= LSQ_MAX_TRIP_MAP) return;
    int cidx = trip_map[cmap];
    if (cidx < 0) return;

    const LSQClusterGpu &cl = clusters[cidx];
    int pi[3] = { trip_pair_idx[cmap * 3 + 0], trip_pair_idx[cmap * 3 + 1], trip_pair_idx[cmap * 3 + 2] };
    for (int e = 0; e < 3; e++) {
        if (!lsq_proceed(rlen[e], cl.s_minim[pi[e]], cl.s_maxim[pi[e]], cl.fcut_type))
            return;
    }

    int pt[3] = {
        ipm[t0 * frame.natmtyp + t1],
        ipm[t0 * frame.natmtyp + t2],
        ipm[atype[parent[a2]] * frame.natmtyp + atype[parent[a3]]]
    };

    unsigned int slot = atomicAdd(counter, 1u);
    if (slot >= (unsigned)cap) return;

    LSQTripGpu tr = {};
    tr.a1 = a1;
    tr.a2 = parent[a2];
    tr.a3 = parent[a3];
    tr.cluster_idx = cidx;
    for (int k = 0; k < 3; k++) {
        tr.pair_index[k] = pi[k];
        tr.pt[k] = pt[k];
        tr.rlen[k] = rlen[k];
        tr.lambda[k] = pp[pt[k]].lambda;
        tr.cheby_type[k] = pp[pt[k]].cheby_type;
        tr.snum[k] = pp[pt[k]].snum_3b;
        tr.rab[k * 3 + 0] = rab[k * 3 + 0];
        tr.rab[k * 3 + 1] = rab[k * 3 + 1];
        tr.rab[k * 3 + 2] = rab[k * 3 + 2];
    }
    out[slot] = tr;
}

__global__ void kEnum4B(const double *coords, const int *parent, const int *atype,
                        const int *ipm, const LSQPairParams *pp,
                        const int *quad_map, const int *quad_pair_idx,
                        const LSQClusterGpu *clusters,
                        LSQBoxGpu box, LSQFrameGpu frame,
                        LSQQuadGpu *out, unsigned int *counter, int cap)
{
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nall = frame.nall;
    int natoms = frame.natoms;
    long long cube = (long long)nall * nall * nall;
    long long jobs = (long long)natoms * cube;
    if (tid >= jobs) return;

    int a1 = (int)(tid / cube);
    long long rem = tid % cube;
    int a2 = (int)(rem / (nall * nall));
    rem = rem % (nall * nall);
    int a3 = (int)(rem / nall);
    int a4 = (int)(rem % nall);

    if (a2 == a1 || a3 == a1 || a4 == a1 || a3 == a2 || a4 == a2 || a4 == a3) return;
    if (frame.perm_4b == 1.0 && parent[a2] > parent[a3]) return;
    if (frame.perm_4b == 1.0 && parent[a3] > parent[a4]) return;

    int pairs[6][2] = {{a1,a2},{a1,a3},{a1,a4},{a2,a3},{a2,a4},{a3,a4}};
    double rlen[6], rab[18];
    for (int e = 0; e < 6; e++) {
        lsq_get_dist(coords, &box, pairs[e][0], pairs[e][1], frame.use_mic,
                     rab[e*3+0], rab[e*3+1], rab[e*3+2], rlen[e]);
        if (rlen[e] >= frame.rcut_4b + frame.rcut_pad) return;
    }

    int t0 = atype[parent[a1]], t1 = atype[parent[a2]];
    int t2 = atype[parent[a3]], t3 = atype[parent[a4]];
    int cmap = lsq_cluster_id4(t0, t1, t2, t3, LSQ_MAX_ATOM_TYPES);
    if (cmap < 0 || cmap >= LSQ_MAX_QUAD_MAP) return;
    int cidx = quad_map[cmap];
    if (cidx < 0) return;

    const LSQClusterGpu &cl = clusters[cidx];
    int pi[6];
    for (int f = 0; f < 6; f++) pi[f] = quad_pair_idx[cmap * 6 + f];
    for (int e = 0; e < 6; e++) {
        if (!lsq_proceed(rlen[e], cl.s_minim[pi[e]], cl.s_maxim[pi[e]], cl.fcut_type))
            return;
    }

    int pt[6] = {
        ipm[t0*frame.natmtyp+t1], ipm[t0*frame.natmtyp+t2], ipm[t0*frame.natmtyp+t3],
        ipm[atype[parent[a2]]*frame.natmtyp+atype[parent[a3]]],
        ipm[atype[parent[a2]]*frame.natmtyp+atype[parent[a4]]],
        ipm[atype[parent[a3]]*frame.natmtyp+atype[parent[a4]]]
    };

    unsigned int slot = atomicAdd(counter, 1u);
    if (slot >= (unsigned)cap) return;

    LSQQuadGpu qd = {};
    qd.a1 = a1;
    qd.a2 = parent[a2];
    qd.a3 = parent[a3];
    qd.a4 = parent[a4];
    qd.cluster_idx = cidx;
    for (int f = 0; f < 6; f++) {
        qd.pair_index[f] = pi[f];
        qd.pt[f] = pt[f];
        qd.rlen[f] = rlen[f];
        qd.lambda[f] = pp[pt[f]].lambda;
        qd.cheby_type[f] = pp[pt[f]].cheby_type;
        qd.snum[f] = pp[pt[f]].snum_4b;
        qd.rab[f*3+0] = rab[f*3+0];
        qd.rab[f*3+1] = rab[f*3+1];
        qd.rab[f*3+2] = rab[f*3+2];
    }
    out[slot] = qd;
}

static bool ensure_accum(int nparams, int natoms)
{
    if (g_gpu.nparams >= nparams && g_gpu.natoms >= natoms &&
        g_gpu.d_fx && g_gpu.d_fy && g_gpu.d_fz)
        return true;

    if (g_gpu.d_fx) cudaFree(g_gpu.d_fx);
    if (g_gpu.d_fy) cudaFree(g_gpu.d_fy);
    if (g_gpu.d_fz) cudaFree(g_gpu.d_fz);
    if (g_gpu.d_stress_xx) cudaFree(g_gpu.d_stress_xx);
    if (g_gpu.d_stress_xy) cudaFree(g_gpu.d_stress_xy);
    if (g_gpu.d_stress_xz) cudaFree(g_gpu.d_stress_xz);
    if (g_gpu.d_stress_yy) cudaFree(g_gpu.d_stress_yy);
    if (g_gpu.d_stress_yz) cudaFree(g_gpu.d_stress_yz);
    if (g_gpu.d_stress_zz) cudaFree(g_gpu.d_stress_zz);
    if (g_gpu.d_frame_energies) cudaFree(g_gpu.d_frame_energies);

    size_t fsize = (size_t)natoms * nparams;
    CUDA_CHECK(cudaMalloc(&g_gpu.d_fx, fsize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_fy, fsize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_fz, fsize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_stress_xx, nparams * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_stress_xy, nparams * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_stress_xz, nparams * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_stress_yy, nparams * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_stress_yz, nparams * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_stress_zz, nparams * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_frame_energies, nparams * sizeof(double)));
    g_gpu.nparams = nparams;
    g_gpu.natoms = natoms;
    return true;
}

static bool ensure_2b(int npairs, int n_pair_types)
{
    if (g_gpu.max_pairs >= npairs && g_gpu.n_pair_types >= n_pair_types) return true;
    if (g_gpu.d_a1) cudaFree(g_gpu.d_a1);
    if (g_gpu.d_a2) cudaFree(g_gpu.d_a2);
    if (g_gpu.d_ptype) cudaFree(g_gpu.d_ptype);
    if (g_gpu.d_rlen) cudaFree(g_gpu.d_rlen);
    if (g_gpu.d_rab) cudaFree(g_gpu.d_rab);
    if (g_gpu.d_pair_params) cudaFree(g_gpu.d_pair_params);
    int cap = (npairs < 4096) ? 4096 : npairs;
    CUDA_CHECK(cudaMalloc(&g_gpu.d_a1, cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_a2, cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_ptype, cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_rlen, cap * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_rab, cap * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_pair_params, n_pair_types * sizeof(LSQPairParams)));
    g_gpu.max_pairs = cap;
    g_gpu.n_pair_types = n_pair_types;
    return true;
}

static bool ensure_3b(int ntrips, int n_clusters, int n_power_terms)
{
    if (g_gpu.max_trips >= ntrips && g_gpu.n_trip_clusters >= n_clusters &&
        g_gpu.n_trip_power_terms >= n_power_terms) return true;
    if (g_gpu.d_trips) cudaFree(g_gpu.d_trips);
    if (g_gpu.d_trip_clusters) cudaFree(g_gpu.d_trip_clusters);
    if (g_gpu.d_trip_power_terms) cudaFree(g_gpu.d_trip_power_terms);
    int cap = (ntrips < 4096) ? 4096 : ntrips;
    CUDA_CHECK(cudaMalloc(&g_gpu.d_trips, cap * sizeof(LSQTripGpu)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_trip_clusters, n_clusters * sizeof(LSQClusterGpu)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_trip_power_terms, n_power_terms * sizeof(LSQPowerTermGpu)));
    g_gpu.max_trips = cap;
    g_gpu.n_trip_clusters = n_clusters;
    g_gpu.n_trip_power_terms = n_power_terms;
    return true;
}

static bool ensure_4b(int nquads, int n_clusters, int n_power_terms)
{
    if (g_gpu.max_quads >= nquads && g_gpu.n_quad_clusters >= n_clusters &&
        g_gpu.n_quad_power_terms >= n_power_terms) return true;
    if (g_gpu.d_quads) cudaFree(g_gpu.d_quads);
    if (g_gpu.d_quad_clusters) cudaFree(g_gpu.d_quad_clusters);
    if (g_gpu.d_quad_power_terms) cudaFree(g_gpu.d_quad_power_terms);
    int cap = (nquads < 4096) ? 4096 : nquads;
    CUDA_CHECK(cudaMalloc(&g_gpu.d_quads, cap * sizeof(LSQQuadGpu)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_quad_clusters, n_clusters * sizeof(LSQClusterGpu)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_quad_power_terms, n_power_terms * sizeof(LSQPowerTermGpu)));
    g_gpu.max_quads = cap;
    g_gpu.n_quad_clusters = n_clusters;
    g_gpu.n_quad_power_terms = n_power_terms;
    return true;
}

void lsq_gpu_init(int device_id)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return;
    if (device_id < 0 || device_id >= ndev) device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceSetLimit(cudaLimitStackSize, 16384);
    g_gpu.device_id = device_id;
    g_gpu.batch_frames = 1;
    g_gpu.batch_pending = 0;
    g_gpu.initialized = true;
}

int lsq_gpu_device_for_rank(int rank)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) return 0;
    return rank % ndev;
}

bool lsq_gpu_available()
{
    int ndev = 0;
    return cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0;
}

void lsq_gpu_finalize()
{
    if (!g_gpu.initialized) return;
    if (g_gpu.d_a1) cudaFree(g_gpu.d_a1);
    if (g_gpu.d_a2) cudaFree(g_gpu.d_a2);
    if (g_gpu.d_ptype) cudaFree(g_gpu.d_ptype);
    if (g_gpu.d_rlen) cudaFree(g_gpu.d_rlen);
    if (g_gpu.d_rab) cudaFree(g_gpu.d_rab);
    if (g_gpu.d_trips) cudaFree(g_gpu.d_trips);
    if (g_gpu.d_quads) cudaFree(g_gpu.d_quads);
    if (g_gpu.d_trip_clusters) cudaFree(g_gpu.d_trip_clusters);
    if (g_gpu.d_quad_clusters) cudaFree(g_gpu.d_quad_clusters);
    if (g_gpu.d_trip_power_terms) cudaFree(g_gpu.d_trip_power_terms);
    if (g_gpu.d_quad_power_terms) cudaFree(g_gpu.d_quad_power_terms);
    if (g_gpu.d_pair_params) cudaFree(g_gpu.d_pair_params);
    if (g_gpu.d_fx) cudaFree(g_gpu.d_fx);
    if (g_gpu.d_fy) cudaFree(g_gpu.d_fy);
    if (g_gpu.d_fz) cudaFree(g_gpu.d_fz);
    if (g_gpu.d_stress_xx) cudaFree(g_gpu.d_stress_xx);
    if (g_gpu.d_stress_xy) cudaFree(g_gpu.d_stress_xy);
    if (g_gpu.d_stress_xz) cudaFree(g_gpu.d_stress_xz);
    if (g_gpu.d_stress_yy) cudaFree(g_gpu.d_stress_yy);
    if (g_gpu.d_stress_yz) cudaFree(g_gpu.d_stress_yz);
    if (g_gpu.d_stress_zz) cudaFree(g_gpu.d_stress_zz);
    if (g_gpu.d_frame_energies) cudaFree(g_gpu.d_frame_energies);
    if (g_gpu.d_coords) cudaFree(g_gpu.d_coords);
    if (g_gpu.d_parent) cudaFree(g_gpu.d_parent);
    if (g_gpu.d_atom_type_idx) cudaFree(g_gpu.d_atom_type_idx);
    if (g_gpu.d_ipm) cudaFree(g_gpu.d_ipm);
    if (g_gpu.d_trip_map) cudaFree(g_gpu.d_trip_map);
    if (g_gpu.d_trip_pair_idx) cudaFree(g_gpu.d_trip_pair_idx);
    if (g_gpu.d_quad_map) cudaFree(g_gpu.d_quad_map);
    if (g_gpu.d_quad_pair_idx) cudaFree(g_gpu.d_quad_pair_idx);
    if (g_gpu.d_enum_counter) cudaFree(g_gpu.d_enum_counter);
    g_gpu = {};
}

bool lsq_gpu_is_initialized() { return g_gpu.initialized; }

void lsq_gpu_begin_frame_accum(int nparams, int natoms)
{
    if (!g_gpu.initialized) return;
    if (!ensure_accum(nparams, natoms)) return;
    size_t fsize = (size_t)natoms * nparams;
    cudaMemset(g_gpu.d_fx, 0, fsize * sizeof(double));
    cudaMemset(g_gpu.d_fy, 0, fsize * sizeof(double));
    cudaMemset(g_gpu.d_fz, 0, fsize * sizeof(double));
    cudaMemset(g_gpu.d_stress_xx, 0, nparams * sizeof(double));
    cudaMemset(g_gpu.d_stress_xy, 0, nparams * sizeof(double));
    cudaMemset(g_gpu.d_stress_xz, 0, nparams * sizeof(double));
    cudaMemset(g_gpu.d_stress_yy, 0, nparams * sizeof(double));
    cudaMemset(g_gpu.d_stress_yz, 0, nparams * sizeof(double));
    cudaMemset(g_gpu.d_stress_zz, 0, nparams * sizeof(double));
    cudaMemset(g_gpu.d_frame_energies, 0, nparams * sizeof(double));
}

bool lsq_gpu_launch_deriv_2b(
    int npairs, int nparams, int natoms, int n_pair_types,
    const int *h_a1, const int *h_a2, const int *h_ptype,
    const double *h_rlen, const double *h_rab,
    const LSQPairParams *h_pair_params,
    double perm_scale, double deriv_const, int fit_stress, int fit_energy)
{
    if (!g_gpu.initialized) return false;
    if (npairs == 0) return true;
    if (!ensure_2b(npairs, n_pair_types)) return false;

    CUDA_CHECK(cudaMemcpy(g_gpu.d_a1, h_a1, npairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_gpu.d_a2, h_a2, npairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_gpu.d_ptype, h_ptype, npairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_gpu.d_rlen, h_rlen, npairs * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_gpu.d_rab, h_rab, npairs * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_gpu.d_pair_params, h_pair_params,
                          n_pair_types * sizeof(LSQPairParams), cudaMemcpyHostToDevice));

    g_gpu.fit_stress = fit_stress;
    g_gpu.fit_energy = fit_energy;

    int block = 256, grid = (npairs + block - 1) / block;
    kDeriv2B<<<grid, block>>>(npairs, g_gpu.d_a1, g_gpu.d_a2, g_gpu.d_ptype,
        g_gpu.d_rlen, g_gpu.d_rab, g_gpu.d_pair_params, nparams, natoms,
        perm_scale, deriv_const, fit_stress, fit_energy,
        g_gpu.d_fx, g_gpu.d_fy, g_gpu.d_fz,
        g_gpu.d_stress_xx, g_gpu.d_stress_xy, g_gpu.d_stress_xz,
        g_gpu.d_stress_yy, g_gpu.d_stress_yz, g_gpu.d_stress_zz,
        g_gpu.d_frame_energies);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool lsq_gpu_launch_deriv_3b(
    int ntrips, int nparams, int natoms,
    const LSQTripGpu *h_trips,
    const LSQClusterGpu *h_clusters, int n_clusters,
    const LSQPowerTermGpu *h_power_terms, int n_power_terms,
    double perm_scale, double deriv_const, int fit_stress, int fit_energy)
{
    if (!g_gpu.initialized) return false;
    if (ntrips == 0) return true;
    if (!ensure_3b(ntrips, n_clusters, n_power_terms)) return false;

    CUDA_CHECK(cudaMemcpy(g_gpu.d_trips, h_trips, ntrips * sizeof(LSQTripGpu), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_gpu.d_trip_clusters, h_clusters,
                          n_clusters * sizeof(LSQClusterGpu), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_gpu.d_trip_power_terms, h_power_terms,
                          n_power_terms * sizeof(LSQPowerTermGpu), cudaMemcpyHostToDevice));

    int block = 256, grid = (ntrips + block - 1) / block;
    kDeriv3B<<<grid, block>>>(ntrips, g_gpu.d_trips, g_gpu.d_trip_clusters, g_gpu.d_trip_power_terms,
        nparams, natoms, perm_scale, deriv_const, fit_stress, fit_energy,
        g_gpu.d_fx, g_gpu.d_fy, g_gpu.d_fz,
        g_gpu.d_stress_xx, g_gpu.d_stress_xy, g_gpu.d_stress_xz,
        g_gpu.d_stress_yy, g_gpu.d_stress_yz, g_gpu.d_stress_zz,
        g_gpu.d_frame_energies);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool lsq_gpu_launch_deriv_4b(
    int nquads, int nparams, int natoms,
    const LSQQuadGpu *h_quads,
    const LSQClusterGpu *h_clusters, int n_clusters,
    const LSQPowerTermGpu *h_power_terms, int n_power_terms,
    double perm_scale, double deriv_const, int fit_stress, int fit_energy)
{
    if (!g_gpu.initialized) return false;
    if (nquads == 0) return true;
    if (!ensure_4b(nquads, n_clusters, n_power_terms)) return false;

    CUDA_CHECK(cudaMemcpy(g_gpu.d_quads, h_quads, nquads * sizeof(LSQQuadGpu), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_gpu.d_quad_clusters, h_clusters,
                          n_clusters * sizeof(LSQClusterGpu), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_gpu.d_quad_power_terms, h_power_terms,
                          n_power_terms * sizeof(LSQPowerTermGpu), cudaMemcpyHostToDevice));

    int block = 256, grid = (nquads + block - 1) / block;
    kDeriv4B<<<grid, block>>>(nquads, g_gpu.d_quads, g_gpu.d_quad_clusters, g_gpu.d_quad_power_terms,
        nparams, natoms, perm_scale, deriv_const, fit_stress, fit_energy,
        g_gpu.d_fx, g_gpu.d_fy, g_gpu.d_fz,
        g_gpu.d_stress_xx, g_gpu.d_stress_xy, g_gpu.d_stress_xz,
        g_gpu.d_stress_yy, g_gpu.d_stress_yz, g_gpu.d_stress_zz,
        g_gpu.d_frame_energies);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool lsq_gpu_finish_frame_accum(
    int nparams, int natoms,
    double *h_fx, double *h_fy, double *h_fz,
    double *h_stress_xx, double *h_stress_xy, double *h_stress_xz,
    double *h_stress_yy, double *h_stress_yz, double *h_stress_zz,
    double *h_frame_energies)
{
    if (!g_gpu.initialized) return false;
    CUDA_CHECK(cudaDeviceSynchronize());
    size_t fsize = (size_t)natoms * nparams;
    CUDA_CHECK(cudaMemcpy(h_fx, g_gpu.d_fx, fsize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fy, g_gpu.d_fy, fsize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fz, g_gpu.d_fz, fsize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_stress_xx, g_gpu.d_stress_xx, nparams * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_stress_xy, g_gpu.d_stress_xy, nparams * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_stress_xz, g_gpu.d_stress_xz, nparams * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_stress_yy, g_gpu.d_stress_yy, nparams * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_stress_yz, g_gpu.d_stress_yz, nparams * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_stress_zz, g_gpu.d_stress_zz, nparams * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_frame_energies, g_gpu.d_frame_energies, nparams * sizeof(double), cudaMemcpyDeviceToHost));
    g_gpu.batch_pending++;
    if (g_gpu.batch_frames > 1 && g_gpu.batch_pending < g_gpu.batch_frames)
        return true;
    g_gpu.batch_pending = 0;
    return true;
}

void lsq_gpu_set_batch_frames(int n)
{
    if (n < 1) n = 1;
    g_gpu.batch_frames = n;
}

int lsq_gpu_batch_frames() { return g_gpu.batch_frames; }

void lsq_gpu_flush_batch()
{
    if (!g_gpu.initialized) return;
    cudaDeviceSynchronize();
    g_gpu.batch_pending = 0;
}

static bool ensure_geo(int nall)
{
    if (g_gpu.max_coords >= nall && g_gpu.d_coords) return true;
    if (g_gpu.d_coords) cudaFree(g_gpu.d_coords);
    if (g_gpu.d_parent) cudaFree(g_gpu.d_parent);
    if (g_gpu.d_atom_type_idx) cudaFree(g_gpu.d_atom_type_idx);
    int cap = (nall < 256) ? 256 : nall;
    CUDA_CHECK(cudaMalloc(&g_gpu.d_coords, cap * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_parent, cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_atom_type_idx, cap * sizeof(int)));
    if (!g_gpu.d_enum_counter)
        CUDA_CHECK(cudaMalloc(&g_gpu.d_enum_counter, sizeof(unsigned int)));
    g_gpu.max_coords = cap;
    return true;
}

static bool ensure_maps()
{
    if (g_gpu.d_ipm) return true;
    CUDA_CHECK(cudaMalloc(&g_gpu.d_ipm, LSQ_MAX_ATOM_TYPES2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_trip_map, LSQ_MAX_TRIP_MAP * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_trip_pair_idx, LSQ_MAX_TRIP_MAP * 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_quad_map, LSQ_MAX_QUAD_MAP * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_quad_pair_idx, LSQ_MAX_QUAD_MAP * 6 * sizeof(int)));
    return true;
}

bool lsq_gpu_upload_static_tables(
    int natmtyp, int n_pair_types,
    const int *h_ipm,
    const LSQPairParams *h_pair_params,
    int use_3b, const int *h_trip_map, const int *h_trip_pair_idx,
    int n_trip_clusters, const LSQClusterGpu *h_trip_clusters,
    int n_trip_terms, const LSQPowerTermGpu *h_trip_terms,
    int use_4b, const int *h_quad_map, const int *h_quad_pair_idx,
    int n_quad_clusters, const LSQClusterGpu *h_quad_clusters,
    int n_quad_terms, const LSQPowerTermGpu *h_quad_terms)
{
    if (!g_gpu.initialized) return false;
    if (!ensure_maps()) return false;
    if (!ensure_2b(4096, n_pair_types)) return false;

    CUDA_CHECK(cudaMemcpy(g_gpu.d_ipm, h_ipm, LSQ_MAX_ATOM_TYPES2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_gpu.d_pair_params, h_pair_params,
                          n_pair_types * sizeof(LSQPairParams), cudaMemcpyHostToDevice));
    g_gpu.n_pair_types = n_pair_types;
    g_gpu.use_3b = use_3b;
    g_gpu.use_4b = use_4b;

    if (use_3b) {
        if (!ensure_3b(4096, n_trip_clusters, n_trip_terms)) return false;
        CUDA_CHECK(cudaMemcpy(g_gpu.d_trip_map, h_trip_map, LSQ_MAX_TRIP_MAP * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g_gpu.d_trip_pair_idx, h_trip_pair_idx,
                              LSQ_MAX_TRIP_MAP * 3 * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g_gpu.d_trip_clusters, h_trip_clusters,
                              n_trip_clusters * sizeof(LSQClusterGpu), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g_gpu.d_trip_power_terms, h_trip_terms,
                              n_trip_terms * sizeof(LSQPowerTermGpu), cudaMemcpyHostToDevice));
        g_gpu.n_trip_clusters = n_trip_clusters;
        g_gpu.n_trip_power_terms = n_trip_terms;
    }
    if (use_4b) {
        if (!ensure_4b(4096, n_quad_clusters, n_quad_terms)) return false;
        CUDA_CHECK(cudaMemcpy(g_gpu.d_quad_map, h_quad_map, LSQ_MAX_QUAD_MAP * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g_gpu.d_quad_pair_idx, h_quad_pair_idx,
                              LSQ_MAX_QUAD_MAP * 6 * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g_gpu.d_quad_clusters, h_quad_clusters,
                              n_quad_clusters * sizeof(LSQClusterGpu), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g_gpu.d_quad_power_terms, h_quad_terms,
                              n_quad_terms * sizeof(LSQPowerTermGpu), cudaMemcpyHostToDevice));
        g_gpu.n_quad_clusters = n_quad_clusters;
        g_gpu.n_quad_power_terms = n_quad_terms;
    }
    g_gpu.static_uploaded = true;
    return true;
}

bool lsq_gpu_upload_frame(
    const double *h_coords, int nall,
    const int *h_parent, const int *h_atom_type_idx,
    const LSQBoxGpu *box, const LSQFrameGpu *frame)
{
    if (!g_gpu.initialized || !g_gpu.static_uploaded) return false;
    if (!ensure_geo(nall)) return false;
    CUDA_CHECK(cudaMemcpy(g_gpu.d_coords, h_coords, nall * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_gpu.d_parent, h_parent, nall * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_gpu.d_atom_type_idx, h_atom_type_idx, nall * sizeof(int), cudaMemcpyHostToDevice));
    g_gpu.h_box = *box;
    g_gpu.h_frame = *frame;
    return true;
}

static bool read_enum_count(int cap, int *out_n)
{
    unsigned int hcount = 0;
    CUDA_CHECK(cudaMemcpy(&hcount, g_gpu.d_enum_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if ((int)hcount > cap) {
        fprintf(stderr, "GPU enum overflow: %u > %d (increase caps or reduce system)\n", hcount, cap);
        return false;
    }
    *out_n = (int)hcount;
    return true;
}

bool lsq_gpu_enumerate_2b(int *out_npairs)
{
    if (!g_gpu.initialized) return false;
    int cap = g_gpu.max_pairs;
    unsigned int z = 0;
    CUDA_CHECK(cudaMemcpy(g_gpu.d_enum_counter, &z, sizeof(unsigned int), cudaMemcpyHostToDevice));

    int nall = g_gpu.h_frame.nall;
    int natoms = g_gpu.h_frame.natoms;
    int jobs = natoms * nall;
    int block = 256, grid = (jobs + block - 1) / block;
    kEnum2B<<<grid, block>>>(g_gpu.d_coords, g_gpu.d_parent, g_gpu.d_atom_type_idx,
        g_gpu.d_ipm, g_gpu.d_pair_params, g_gpu.h_box, g_gpu.h_frame,
        g_gpu.d_a1, g_gpu.d_a2, g_gpu.d_ptype, g_gpu.d_rlen, g_gpu.d_rab,
        g_gpu.d_enum_counter, cap);
    CUDA_CHECK(cudaGetLastError());
    return read_enum_count(cap, out_npairs);
}

bool lsq_gpu_enumerate_3b(int *out_ntrips)
{
    if (!g_gpu.initialized || !g_gpu.use_3b) { *out_ntrips = 0; return true; }
    int cap = g_gpu.max_trips;
    unsigned int z = 0;
    CUDA_CHECK(cudaMemcpy(g_gpu.d_enum_counter, &z, sizeof(unsigned int), cudaMemcpyHostToDevice));

    int nall = g_gpu.h_frame.nall;
    int natoms = g_gpu.h_frame.natoms;
    long long jobs = (long long)natoms * nall * nall;
    int block = 256;
    int grid = (int)((jobs + block - 1) / block);
    kEnum3B<<<grid, block>>>(g_gpu.d_coords, g_gpu.d_parent, g_gpu.d_atom_type_idx,
        g_gpu.d_ipm, g_gpu.d_pair_params, g_gpu.d_trip_map, g_gpu.d_trip_pair_idx,
        g_gpu.d_trip_clusters, g_gpu.h_box, g_gpu.h_frame,
        g_gpu.d_trips, g_gpu.d_enum_counter, cap);
    CUDA_CHECK(cudaGetLastError());
    return read_enum_count(cap, out_ntrips);
}

bool lsq_gpu_enumerate_4b(int *out_nquads)
{
    if (!g_gpu.initialized || !g_gpu.use_4b) { *out_nquads = 0; return true; }
    int cap = g_gpu.max_quads;
    unsigned int z = 0;
    CUDA_CHECK(cudaMemcpy(g_gpu.d_enum_counter, &z, sizeof(unsigned int), cudaMemcpyHostToDevice));

    int nall = g_gpu.h_frame.nall;
    int natoms = g_gpu.h_frame.natoms;
    long long cube = (long long)nall * nall * nall;
    long long jobs = (long long)natoms * cube;
    int block = 256;
    int grid = (int)((jobs + block - 1) / block);
    kEnum4B<<<grid, block>>>(g_gpu.d_coords, g_gpu.d_parent, g_gpu.d_atom_type_idx,
        g_gpu.d_ipm, g_gpu.d_pair_params, g_gpu.d_quad_map, g_gpu.d_quad_pair_idx,
        g_gpu.d_quad_clusters, g_gpu.h_box, g_gpu.h_frame,
        g_gpu.d_quads, g_gpu.d_enum_counter, cap);
    CUDA_CHECK(cudaGetLastError());
    return read_enum_count(cap, out_nquads);
}

bool lsq_gpu_launch_deriv_2b_device(
    int npairs, int nparams, int natoms,
    double perm_scale, double deriv_const, int fit_stress, int fit_energy)
{
    if (!g_gpu.initialized || npairs == 0) return true;
    int block = 256, grid = (npairs + block - 1) / block;
    kDeriv2B<<<grid, block>>>(npairs, g_gpu.d_a1, g_gpu.d_a2, g_gpu.d_ptype,
        g_gpu.d_rlen, g_gpu.d_rab, g_gpu.d_pair_params, nparams, natoms,
        perm_scale, deriv_const, fit_stress, fit_energy,
        g_gpu.d_fx, g_gpu.d_fy, g_gpu.d_fz,
        g_gpu.d_stress_xx, g_gpu.d_stress_xy, g_gpu.d_stress_xz,
        g_gpu.d_stress_yy, g_gpu.d_stress_yz, g_gpu.d_stress_zz,
        g_gpu.d_frame_energies);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool lsq_gpu_launch_deriv_3b_device(
    int ntrips, int nparams, int natoms,
    double perm_scale, double deriv_const, int fit_stress, int fit_energy)
{
    if (!g_gpu.initialized || ntrips == 0) return true;
    int block = 256, grid = (ntrips + block - 1) / block;
    kDeriv3B<<<grid, block>>>(ntrips, g_gpu.d_trips, g_gpu.d_trip_clusters, g_gpu.d_trip_power_terms,
        nparams, natoms, perm_scale, deriv_const, fit_stress, fit_energy,
        g_gpu.d_fx, g_gpu.d_fy, g_gpu.d_fz,
        g_gpu.d_stress_xx, g_gpu.d_stress_xy, g_gpu.d_stress_xz,
        g_gpu.d_stress_yy, g_gpu.d_stress_yz, g_gpu.d_stress_zz,
        g_gpu.d_frame_energies);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool lsq_gpu_launch_deriv_4b_device(
    int nquads, int nparams, int natoms,
    double perm_scale, double deriv_const, int fit_stress, int fit_energy)
{
    if (!g_gpu.initialized || nquads == 0) return true;
    int block = 256, grid = (nquads + block - 1) / block;
    kDeriv4B<<<grid, block>>>(nquads, g_gpu.d_quads, g_gpu.d_quad_clusters, g_gpu.d_quad_power_terms,
        nparams, natoms, perm_scale, deriv_const, fit_stress, fit_energy,
        g_gpu.d_fx, g_gpu.d_fy, g_gpu.d_fz,
        g_gpu.d_stress_xx, g_gpu.d_stress_xy, g_gpu.d_stress_xz,
        g_gpu.d_stress_yy, g_gpu.d_stress_yz, g_gpu.d_stress_zz,
        g_gpu.d_frame_energies);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#endif // USE_CUDA
