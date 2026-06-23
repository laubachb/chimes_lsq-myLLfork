/*
    ChIMES LSQ — host-side GPU integration for 2/3/4-body derivatives
*/
#ifdef USE_CUDA

#include "chimes_lsq_gpu.cuh"
#include "chimes_lsq_gpu_types.h"
#include "Cheby.h"
#include "A_Matrix.h"
#include "Fcut.h"
#include "Cluster.h"

#include <vector>
#include <cstring>

static int cheby_type_to_int(Cheby_trans t)
{
    switch (t) {
    case Cheby_trans::MORSE:     return 0;
    case Cheby_trans::INVRSE_R:  return 1;
    default:                     return 2;
    }
}

static int cheby_fix_to_int(Cheby_fix t)
{
    switch (t) {
    case Cheby_fix::ZERO_DERIV:     return 0;
    case Cheby_fix::CONSTANT_DERIV: return 1;
    case Cheby_fix::SMOOTH:         return 2;
    default:                        return 2;
    }
}

static void fill_pair_params(vector<PAIRS> &ff, std::vector<LSQPairParams> &h_pp)
{
    h_pp.resize(ff.size());
    for (size_t p = 0; p < ff.size(); p++) {
        h_pp[p].snum        = ff[p].SNUM;
        h_pp[p].snum_3b     = ff[p].SNUM_3B_CHEBY;
        h_pp[p].snum_4b     = ff[p].SNUM_4B_CHEBY;
        h_pp[p].vstart      = (int)p * ff[p].SNUM;
        h_pp[p].s_minim     = ff[p].S_MINIM;
        h_pp[p].s_maxim     = ff[p].S_MAXIM;
        h_pp[p].x_diff      = ff[p].X_DIFF;
        h_pp[p].x_avg       = ff[p].X_AVG;
        h_pp[p].lambda      = ff[p].LAMBDA;
        h_pp[p].cheby_type  = cheby_type_to_int(ff[p].CHEBY_TYPE);
        h_pp[p].fcut_type   = (ff[p].FORCE_CUTOFF.TYPE == FCUT_TYPE::TERSOFF) ? 1 : 0;
        h_pp[p].fcut_power  = ff[p].FORCE_CUTOFF.POWER;
        h_pp[p].fcut_offset = ff[p].FORCE_CUTOFF.OFFSET;
    }
}

static void build_cluster_tables(CLUSTER_LIST &list, int npairs_per_cluster,
                                 int vstart0,
                                 std::vector<LSQClusterGpu> &clusters,
                                 std::vector<LSQPowerTermGpu> &terms)
{
    int term_off = 0;
    int vstart = vstart0;
    for (size_t ci = 0; ci < list.VEC.size(); ci++) {
        CLUSTER &cl = list.VEC[ci];
        LSQClusterGpu cg = {};
        cg.vstart = vstart;
        cg.n_terms = cl.N_ALLOWED_POWERS;
        cg.term_offset = term_off;
        cg.npairs = npairs_per_cluster;
        cg.fcut_type = (cl.FORCE_CUTOFF.TYPE == FCUT_TYPE::TERSOFF) ? 1 : 0;
        cg.fcut_power = cl.FORCE_CUTOFF.POWER;
        cg.fcut_offset = cl.FORCE_CUTOFF.OFFSET;
        for (int j = 0; j < npairs_per_cluster; j++) {
            cg.s_minim[j] = cl.S_MINIM[j];
            cg.s_maxim[j] = cl.S_MAXIM[j];
            cg.x_diff[j] = cl.X_DIFF[j];
            cg.x_avg[j] = cl.X_AVG[j];
        }
        clusters.push_back(cg);
        for (int i = 0; i < cl.N_ALLOWED_POWERS; i++) {
            LSQPowerTermGpu pt = {};
            pt.param_offset = cl.PARAM_INDICES[i];
            for (int j = 0; j < npairs_per_cluster; j++)
                pt.pow[j] = cl.ALLOWED_POWERS[i][j];
            terms.push_back(pt);
        }
        term_off += cl.N_ALLOWED_POWERS;
        vstart += cl.N_TRUE_ALLOWED_POWERS;
    }
}

static void fill_cluster_map(CLUSTER_LIST &list, int map_size, int npairs,
                             std::vector<int> &cluster_map,
                             std::vector<int> &pair_idx_flat)
{
    cluster_map.assign(map_size, -1);
    pair_idx_flat.assign(map_size * npairs, 0);
    for (size_t i = 0; i < list.INT_MAP.size() && (int)i < map_size; i++) {
        cluster_map[i] = list.INT_MAP[i];
        if ((int)i < (int)list.PAIR_INDICES.size()) {
            for (int k = 0; k < npairs; k++)
                pair_idx_flat[i * npairs + k] = list.PAIR_INDICES[i][k];
        }
    }
}

static void scatter_to_amat(A_MAT &a_matrix, JOB_CONTROL &controls,
                            int nparams, int natoms, double inv_vol,
                            int fit_stress,
                            const std::vector<double> &h_fx,
                            const std::vector<double> &h_fy,
                            const std::vector<double> &h_fz,
                            const std::vector<double> &h_sxx,
                            const std::vector<double> &h_sxy,
                            const std::vector<double> &h_sxz,
                            const std::vector<double> &h_syy,
                            const std::vector<double> &h_syz,
                            const std::vector<double> &h_szz,
                            const std::vector<double> &h_ener)
{
    for (int a = 0; a < natoms; a++) {
        for (int p = 0; p < nparams; p++) {
            int idx = a * nparams + p;
            a_matrix.FORCES[a][p].X += h_fx[idx];
            a_matrix.FORCES[a][p].Y += h_fy[idx];
            a_matrix.FORCES[a][p].Z += h_fz[idx];
        }
    }
    if (fit_stress == 1) {
        for (int p = 0; p < nparams; p++) {
            a_matrix.STRESSES[p].XX += h_sxx[p] * inv_vol;
            a_matrix.STRESSES[p].YY += h_syy[p] * inv_vol;
            a_matrix.STRESSES[p].ZZ += h_szz[p] * inv_vol;
        }
    } else if (fit_stress == 2) {
        for (int p = 0; p < nparams; p++) {
            a_matrix.STRESSES[p].XX += h_sxx[p] * inv_vol;
            a_matrix.STRESSES[p].XY += h_sxy[p] * inv_vol;
            a_matrix.STRESSES[p].XZ += h_sxz[p] * inv_vol;
            a_matrix.STRESSES[p].YY += h_syy[p] * inv_vol;
            a_matrix.STRESSES[p].YZ += h_syz[p] * inv_vol;
            a_matrix.STRESSES[p].ZZ += h_szz[p] * inv_vol;
        }
    }
    if (controls.FIT_ENER) {
        for (int p = 0; p < nparams; p++)
            a_matrix.FRAME_ENERGIES[p] += h_ener[p];
    }
}

static bool upload_frame_geometry(FRAME &system, NEIGHBORS &nlist, JOB_CONTROL &controls)
{
    const int nall = system.ALL_ATOMS;
    std::vector<double> coords(nall * 3);
    std::vector<int> parent(nall), atype_idx(nall);

    if (system.ATOMS == system.ALL_ATOMS) {
        for (int i = 0; i < nall; i++) {
            coords[i * 3 + 0] = system.COORDS[i].X;
            coords[i * 3 + 1] = system.COORDS[i].Y;
            coords[i * 3 + 2] = system.COORDS[i].Z;
        }
    } else {
        for (int i = 0; i < nall; i++) {
            coords[i * 3 + 0] = system.ALL_COORDS[i].X;
            coords[i * 3 + 1] = system.ALL_COORDS[i].Y;
            coords[i * 3 + 2] = system.ALL_COORDS[i].Z;
        }
    }
    for (int i = 0; i < nall; i++) {
        parent[i] = system.PARENT[i];
        atype_idx[i] = system.ATOMTYPE_IDX[system.PARENT[i]];
    }

    LSQBoxGpu box = {};
    for (int i = 0; i < 9; i++) {
        box.hmat[i] = system.BOXDIM.HMAT[i];
        box.invr_hmat[i] = system.BOXDIM.INVR_HMAT[i];
    }

    LSQFrameGpu frame = {};
    frame.natoms = system.ATOMS;
    frame.nall = nall;
    frame.natmtyp = controls.NATMTYP;
    frame.use_mic = (system.ATOMS == system.ALL_ATOMS) ? 1 : 0;
    frame.cheby_fix_type = cheby_fix_to_int(controls.cheby_fix_type);
    frame.rcut_2b = nlist.MAX_CUTOFF;
    frame.rcut_3b = nlist.MAX_CUTOFF_3B;
    frame.rcut_4b = nlist.MAX_CUTOFF_4B;
    frame.rcut_pad = nlist.RCUT_PADDING;
    frame.perm_2b = nlist.PERM_SCALE[2];
    frame.perm_3b = nlist.PERM_SCALE[3];
    frame.perm_4b = nlist.PERM_SCALE[4];
    frame.cheby_smooth_distance = controls.cheby_smooth_distance;

    return lsq_gpu_upload_frame(coords.data(), nall, parent.data(), atype_idx.data(), &box, &frame);
}

// Cached static tables (pair params + cluster metadata + lookup maps)
static std::vector<LSQPairParams> g_cached_pp;
static std::vector<LSQClusterGpu> g_cached_trip_clusters, g_cached_quad_clusters;
static std::vector<LSQPowerTermGpu> g_cached_trip_terms, g_cached_quad_terms;
static std::vector<int> g_cached_ipm, g_cached_trip_map, g_cached_trip_pi;
static std::vector<int> g_cached_quad_map, g_cached_quad_pi;
static bool g_static_ready = false;
static int g_cached_natmtyp = -1;

static bool ensure_static_tables(Cheby &cheby, CLUSTER_LIST &trips, CLUSTER_LIST &quads)
{
    JOB_CONTROL &controls = cheby.CONTROLS;
    vector<PAIRS> &ff = cheby.FF_2BODY;

    if (g_static_ready && g_cached_natmtyp == controls.NATMTYP)
        return true;

    fill_pair_params(ff, g_cached_pp);

    g_cached_ipm.assign(LSQ_MAX_ATOM_TYPES2, 0);
    for (size_t i = 0; i < cheby.INT_PAIR_MAP.size() && i < (size_t)LSQ_MAX_ATOM_TYPES2; i++)
        g_cached_ipm[i] = cheby.INT_PAIR_MAP[i];

    int n_2b = controls.TOT_SNUM;
    int n_3b = controls.NUM_3B_CHEBY;

    g_cached_trip_clusters.clear(); g_cached_trip_terms.clear();
    g_cached_quad_clusters.clear(); g_cached_quad_terms.clear();

    if (controls.USE_3B_CHEBY)
        build_cluster_tables(trips, 3, n_2b, g_cached_trip_clusters, g_cached_trip_terms);
    if (controls.USE_4B_CHEBY)
        build_cluster_tables(quads, 6, n_2b + n_3b, g_cached_quad_clusters, g_cached_quad_terms);

    fill_cluster_map(trips, LSQ_MAX_TRIP_MAP, 3, g_cached_trip_map, g_cached_trip_pi);
    fill_cluster_map(quads, LSQ_MAX_QUAD_MAP, 6, g_cached_quad_map, g_cached_quad_pi);

    if (!lsq_gpu_upload_static_tables(
            controls.NATMTYP, (int)ff.size(),
            g_cached_ipm.data(), g_cached_pp.data(),
            controls.USE_3B_CHEBY ? 1 : 0,
            g_cached_trip_map.data(), g_cached_trip_pi.data(),
            (int)g_cached_trip_clusters.size(), g_cached_trip_clusters.data(),
            (int)g_cached_trip_terms.size(), g_cached_trip_terms.data(),
            controls.USE_4B_CHEBY ? 1 : 0,
            g_cached_quad_map.data(), g_cached_quad_pi.data(),
            (int)g_cached_quad_clusters.size(), g_cached_quad_clusters.data(),
            (int)g_cached_quad_terms.size(), g_cached_quad_terms.data()))
        return false;

    g_static_ready = true;
    g_cached_natmtyp = controls.NATMTYP;
    return true;
}

bool lsq_gpu_deriv_cheby(Cheby &cheby, A_MAT &a_matrix,
                         CLUSTER_LIST &trips, CLUSTER_LIST &quads)
{
    JOB_CONTROL &controls = cheby.CONTROLS;
    FRAME &system = cheby.SYSTEM;
    NEIGHBORS &nlist = cheby.NEIGHBOR_LIST;
    vector<PAIRS> &ff = cheby.FF_2BODY;

    if (!lsq_gpu_available() || !lsq_gpu_is_initialized()) return false;
    if (controls.HIERARCHICAL_FIT || controls.FIT_COUL) return false;
    if (ff.empty() || ff[0].PAIRTYP != "CHEBYSHEV") return false;

    int max_order = 0;
    for (size_t i = 0; i < ff.size(); i++) {
        if (ff[i].SNUM > max_order) max_order = ff[i].SNUM;
        if (ff[i].SNUM_3B_CHEBY > max_order) max_order = ff[i].SNUM_3B_CHEBY;
        if (ff[i].SNUM_4B_CHEBY > max_order) max_order = ff[i].SNUM_4B_CHEBY;
    }
    if (max_order > LSQ_MAX_POLY_ORDER) return false;

    const int nparams = controls.TOT_SHORT_RANGE;
    const int natoms  = system.ATOMS;
    const double inv_vol = 1.0 / system.BOXDIM.VOL;

    int fit_stress = 0;
    if (controls.FIT_STRESS) fit_stress = 1;
    else if (controls.FIT_STRESS_ALL) fit_stress = 2;

    if (!ensure_static_tables(cheby, trips, quads)) return false;
    if (!upload_frame_geometry(system, nlist, controls)) return false;

    std::vector<double> h_fx(natoms * nparams, 0.0);
    std::vector<double> h_fy(natoms * nparams, 0.0);
    std::vector<double> h_fz(natoms * nparams, 0.0);
    std::vector<double> h_sxx(nparams, 0.0), h_sxy(nparams, 0.0), h_sxz(nparams, 0.0);
    std::vector<double> h_syy(nparams, 0.0), h_syz(nparams, 0.0), h_szz(nparams, 0.0);
    std::vector<double> h_ener(nparams, 0.0);

    if (!lsq_gpu_begin_frame_accum(nparams, natoms)) return false;

    int npairs = 0, ntrips = 0, nquads = 0;

    if (ff[0].SNUM > 0) {
        if (!lsq_gpu_enumerate_2b(&npairs)) return false;
        if (!lsq_gpu_launch_deriv_2b_device(npairs, nparams, natoms,
                nlist.PERM_SCALE[2], cheby.DERIV_CONST, fit_stress,
                controls.FIT_ENER ? 1 : 0))
            return false;
    }

    if (controls.USE_3B_CHEBY) {
        if (!lsq_gpu_enumerate_3b(&ntrips)) return false;
        if (!lsq_gpu_launch_deriv_3b_device(ntrips, nparams, natoms,
                nlist.PERM_SCALE[3], cheby.DERIV_CONST, fit_stress,
                controls.FIT_ENER ? 1 : 0))
            return false;
    }

    if (controls.USE_4B_CHEBY) {
        if (!lsq_gpu_enumerate_4b(&nquads)) return false;
        if (!lsq_gpu_launch_deriv_4b_device(nquads, nparams, natoms,
                nlist.PERM_SCALE[4], cheby.DERIV_CONST, fit_stress,
                controls.FIT_ENER ? 1 : 0))
            return false;
    }

    if (!lsq_gpu_finish_frame_accum(nparams, natoms,
            h_fx.data(), h_fy.data(), h_fz.data(),
            h_sxx.data(), h_sxy.data(), h_sxz.data(),
            h_syy.data(), h_syz.data(), h_szz.data(),
            h_ener.data()))
        return false;

    scatter_to_amat(a_matrix, controls, nparams, natoms, inv_vol, fit_stress,
                    h_fx, h_fy, h_fz, h_sxx, h_sxy, h_sxz, h_syy, h_syz, h_szz, h_ener);
    return true;
}

#endif // USE_CUDA
