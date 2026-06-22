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

static int cheby_type_to_int(Cheby_trans t)
{
    switch (t) {
    case Cheby_trans::MORSE:     return 0;
    case Cheby_trans::INVRSE_R:  return 1;
    default:                     return 2;
    }
}

static int pair_type_idx(FRAME &sys, JOB_CONTROL &ctrl, vector<int> &ipm,
                         int a1, int a2)
{
    return ipm[sys.ATOMTYPE_IDX[sys.PARENT[a1]] * ctrl.NATMTYP +
               sys.ATOMTYPE_IDX[sys.PARENT[a2]]];
}

static void fill_pair_params(vector<PAIRS> &ff, std::vector<LSQPairParams> &h_pp)
{
    h_pp.resize(ff.size());
    for (size_t p = 0; p < ff.size(); p++) {
        h_pp[p].snum        = ff[p].SNUM;
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

static void build_trips(Cheby &cheby, CLUSTER_LIST &trips,
                        std::vector<LSQTripGpu> &out)
{
    JOB_CONTROL &controls = cheby.CONTROLS;
    FRAME &system = cheby.SYSTEM;
    NEIGHBORS &nlist = cheby.NEIGHBOR_LIST;
    vector<PAIRS> &ff = cheby.FF_2BODY;
    double perm_scale = nlist.PERM_SCALE[3];

    vector<int> atom_type_index(3);
    XYZ rab[3];
    int natoms = system.ATOMS;

    for (int a1 = 0; a1 < natoms; a1++) {
        for (size_t a2idx = 0; a2idx < nlist.LIST_3B[a1].size(); a2idx++) {
            int a2 = nlist.LIST_3B[a1][a2idx];
            for (size_t a3idx = 0; a3idx < nlist.LIST_3B[a1].size(); a3idx++) {
                int a3 = nlist.LIST_3B[a1][a3idx];
                if (a3 == a2) continue;
                if (perm_scale == 1.0 && system.PARENT[a2] > system.PARENT[a3])
                    continue;

                atom_type_index[0] = system.get_atomtype_idx(a1);
                atom_type_index[1] = system.get_atomtype_idx(a2);
                atom_type_index[2] = system.get_atomtype_idx(a3);
                int tidx = trips.make_id_int(atom_type_index);
                int trip_idx = trips.INT_MAP[tidx];
                if (trip_idx < 0) continue;

                TRIPLETS &cl = trips.VEC[trip_idx];
                int pt_ij = pair_type_idx(system, controls, cheby.INT_PAIR_MAP, a1, a2);
                int pt_ik = pair_type_idx(system, controls, cheby.INT_PAIR_MAP, a1, a3);
                int pt_jk = pair_type_idx(system, controls, cheby.INT_PAIR_MAP, a2, a3);

                double rlen[3];
                rlen[0] = get_dist(system, rab[0], a1, a2);
                rlen[1] = get_dist(system, rab[1], a1, a3);
                rlen[2] = get_dist(system, rab[2], a2, a3);

                int pi[3] = {
                    trips.PAIR_INDICES[tidx][0],
                    trips.PAIR_INDICES[tidx][1],
                    trips.PAIR_INDICES[tidx][2]
                };

                if (!cl.FORCE_CUTOFF.PROCEED(rlen[0], cl.S_MINIM[pi[0]], cl.S_MAXIM[pi[0]]))
                    continue;
                if (!cl.FORCE_CUTOFF.PROCEED(rlen[1], cl.S_MINIM[pi[1]], cl.S_MAXIM[pi[1]]))
                    continue;
                if (!cl.FORCE_CUTOFF.PROCEED(rlen[2], cl.S_MINIM[pi[2]], cl.S_MAXIM[pi[2]]))
                    continue;

                if (cl.MIN_FOUND[0] == -1) {
                    cl.MIN_FOUND[pi[0]] = rlen[0];
                    cl.MIN_FOUND[pi[1]] = rlen[1];
                    cl.MIN_FOUND[pi[2]] = rlen[2];
                } else {
                    if (rlen[0] < cl.MIN_FOUND[pi[0]]) cl.MIN_FOUND[pi[0]] = rlen[0];
                    if (rlen[1] < cl.MIN_FOUND[pi[1]]) cl.MIN_FOUND[pi[1]] = rlen[1];
                    if (rlen[2] < cl.MIN_FOUND[pi[2]]) cl.MIN_FOUND[pi[2]] = rlen[2];
                }
                cl.N_CFG_CONTRIB++;

                LSQTripGpu tr = {};
                tr.a1 = a1;
                tr.a2 = system.PARENT[a2];
                tr.a3 = system.PARENT[a3];
                tr.cluster_idx = trip_idx;
                for (int k = 0; k < 3; k++) {
                    tr.pair_index[k] = pi[k];
                    tr.pt[k] = (k == 0) ? pt_ij : (k == 1) ? pt_ik : pt_jk;
                    tr.rlen[k] = rlen[k];
                    tr.lambda[k] = ff[tr.pt[k]].LAMBDA;
                    tr.cheby_type[k] = cheby_type_to_int(ff[tr.pt[k]].CHEBY_TYPE);
                    tr.snum[k] = ff[tr.pt[k]].SNUM_3B_CHEBY;
                    tr.rab[k*3+0] = rab[k].X;
                    tr.rab[k*3+1] = rab[k].Y;
                    tr.rab[k*3+2] = rab[k].Z;
                }
                out.push_back(tr);
            }
        }
    }
}

static void build_quads(Cheby &cheby, CLUSTER_LIST &quads,
                        std::vector<LSQQuadGpu> &out)
{
    JOB_CONTROL &controls = cheby.CONTROLS;
    FRAME &system = cheby.SYSTEM;
    NEIGHBORS &nlist = cheby.NEIGHBOR_LIST;
    vector<PAIRS> &ff = cheby.FF_2BODY;
    double perm_scale = nlist.PERM_SCALE[4];

    vector<int> atom_type_index(4);
    XYZ rab[6];
    int natoms = system.ATOMS;

    for (int a1 = 0; a1 < natoms; a1++) {
        for (size_t a2idx = 0; a2idx < nlist.LIST_4B[a1].size(); a2idx++) {
            int a2 = nlist.LIST_4B[a1][a2idx];
            for (size_t a3idx = 0; a3idx < nlist.LIST_4B[a1].size(); a3idx++) {
                int a3 = nlist.LIST_4B[a1][a3idx];
                if (a3 == a2) continue;
                if (perm_scale == 1.0 && system.PARENT[a2] > system.PARENT[a3])
                    continue;

                for (size_t a4idx = 0; a4idx < nlist.LIST_4B[a1].size(); a4idx++) {
                    int a4 = nlist.LIST_4B[a1][a4idx];
                    if (a2 == a4 || a3 == a4) continue;
                    if (perm_scale == 1.0 && system.PARENT[a3] > system.PARENT[a4])
                        continue;

                    atom_type_index[0] = system.get_atomtype_idx(a1);
                    atom_type_index[1] = system.get_atomtype_idx(a2);
                    atom_type_index[2] = system.get_atomtype_idx(a3);
                    atom_type_index[3] = system.get_atomtype_idx(a4);
                    int qid = quads.make_id_int(atom_type_index);
                    int quad_idx = quads.INT_MAP[qid];
                    if (quad_idx < 0) continue;

                    QUADRUPLETS &cl = quads.VEC[quad_idx];
                    int pt[6] = {
                        pair_type_idx(system, controls, cheby.INT_PAIR_MAP, a1, a2),
                        pair_type_idx(system, controls, cheby.INT_PAIR_MAP, a1, a3),
                        pair_type_idx(system, controls, cheby.INT_PAIR_MAP, a1, a4),
                        pair_type_idx(system, controls, cheby.INT_PAIR_MAP, a2, a3),
                        pair_type_idx(system, controls, cheby.INT_PAIR_MAP, a2, a4),
                        pair_type_idx(system, controls, cheby.INT_PAIR_MAP, a3, a4)
                    };

                    double rlen[6];
                    rlen[0] = get_dist(system, rab[0], a1, a2);
                    rlen[1] = get_dist(system, rab[1], a1, a3);
                    rlen[2] = get_dist(system, rab[2], a1, a4);
                    rlen[3] = get_dist(system, rab[3], a2, a3);
                    rlen[4] = get_dist(system, rab[4], a2, a4);
                    rlen[5] = get_dist(system, rab[5], a3, a4);

                    int pi[6];
                    for (int f = 0; f < 6; f++)
                        pi[f] = quads.PAIR_INDICES[qid][f];

                    bool ok = true;
                    for (int f = 0; f < 6; f++) {
                        if (!cl.FORCE_CUTOFF.PROCEED(rlen[f], cl.S_MINIM[pi[f]], cl.S_MAXIM[pi[f]])) {
                            ok = false;
                            break;
                        }
                    }
                    if (!ok) continue;

                    if (cl.MIN_FOUND[0] == -1) {
                        for (int f = 0; f < 6; f++)
                            cl.MIN_FOUND[pi[f]] = rlen[f];
                    } else {
                        for (int f = 0; f < 6; f++) {
                            if (rlen[f] < cl.MIN_FOUND[pi[f]])
                                cl.MIN_FOUND[pi[f]] = rlen[f];
                        }
                    }
                    cl.N_CFG_CONTRIB++;

                    LSQQuadGpu qd = {};
                    qd.a1 = a1;
                    qd.a2 = system.PARENT[a2];
                    qd.a3 = system.PARENT[a3];
                    qd.a4 = system.PARENT[a4];
                    qd.cluster_idx = quad_idx;
                    for (int f = 0; f < 6; f++) {
                        qd.pair_index[f] = pi[f];
                        qd.pt[f] = pt[f];
                        qd.rlen[f] = rlen[f];
                        qd.lambda[f] = ff[pt[f]].LAMBDA;
                        qd.cheby_type[f] = cheby_type_to_int(ff[pt[f]].CHEBY_TYPE);
                        qd.snum[f] = ff[pt[f]].SNUM_4B_CHEBY;
                        qd.rab[f*3+0] = rab[f].X;
                        qd.rab[f*3+1] = rab[f].Y;
                        qd.rab[f*3+2] = rab[f].Z;
                    }
                    out.push_back(qd);
                }
            }
        }
    }
}

static void build_2b_pairs(Cheby &cheby, std::vector<int> &h_a1,
                           std::vector<int> &h_a2, std::vector<int> &h_ptype,
                           std::vector<double> &h_rlen, std::vector<double> &h_rab)
{
    JOB_CONTROL &controls = cheby.CONTROLS;
    FRAME &system = cheby.SYSTEM;
    NEIGHBORS &nlist = cheby.NEIGHBOR_LIST;
    vector<PAIRS> &ff = cheby.FF_2BODY;
    XYZ rab;

    for (int a1 = 0; a1 < system.ATOMS; a1++) {
        for (size_t a2idx = 0; a2idx < nlist.LIST[a1].size(); a2idx++) {
            int a2 = nlist.LIST[a1][a2idx];
            int pt = pair_type_idx(system, controls, cheby.INT_PAIR_MAP, a1, a2);
            double rlen = get_dist(system, rab, a1, a2);

            if (rlen < ff[pt].MIN_FOUND_DIST)
                ff[pt].MIN_FOUND_DIST = rlen;

            if (rlen > ff[pt].S_MINIM && rlen < ff[pt].S_MAXIM) {
                ff[pt].N_CFG_CONTRIB++;
                h_a1.push_back(a1);
                h_a2.push_back(system.PARENT[a2]);
                h_ptype.push_back(pt);
                h_rlen.push_back(rlen);
                h_rab.push_back(rab.X);
                h_rab.push_back(rab.Y);
                h_rab.push_back(rab.Z);
            }
        }
    }
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

    int n_2b = controls.TOT_SNUM;
    int n_3b = controls.NUM_3B_CHEBY;

    std::vector<LSQPairParams> h_pp;
    fill_pair_params(ff, h_pp);

    std::vector<int> h_a1, h_a2, h_ptype;
    std::vector<double> h_rlen, h_rab2;
    if (ff[0].SNUM > 0)
        build_2b_pairs(cheby, h_a1, h_a2, h_ptype, h_rlen, h_rab2);

    std::vector<LSQClusterGpu> trip_clusters, quad_clusters;
    std::vector<LSQPowerTermGpu> trip_terms, quad_terms;
    if (controls.USE_3B_CHEBY)
        build_cluster_tables(trips, 3, n_2b, trip_clusters, trip_terms);
    if (controls.USE_4B_CHEBY)
        build_cluster_tables(quads, 6, n_2b + n_3b, quad_clusters, quad_terms);

    std::vector<LSQTripGpu> h_trips;
    std::vector<LSQQuadGpu> h_quads;
    if (controls.USE_3B_CHEBY)
        build_trips(cheby, trips, h_trips);
    if (controls.USE_4B_CHEBY)
        build_quads(cheby, quads, h_quads);

    std::vector<double> h_fx(natoms * nparams, 0.0);
    std::vector<double> h_fy(natoms * nparams, 0.0);
    std::vector<double> h_fz(natoms * nparams, 0.0);
    std::vector<double> h_sxx(nparams, 0.0), h_sxy(nparams, 0.0), h_sxz(nparams, 0.0);
    std::vector<double> h_syy(nparams, 0.0), h_syz(nparams, 0.0), h_szz(nparams, 0.0);
    std::vector<double> h_ener(nparams, 0.0);

    lsq_gpu_begin_frame_accum(nparams, natoms);

    if (ff[0].SNUM > 0 && !h_a1.empty()) {
        if (!lsq_gpu_launch_deriv_2b(
                (int)h_a1.size(), nparams, natoms, (int)ff.size(),
                h_a1.data(), h_a2.data(), h_ptype.data(),
                h_rlen.data(), h_rab2.data(), h_pp.data(),
                nlist.PERM_SCALE[2], cheby.DERIV_CONST, fit_stress,
                controls.FIT_ENER ? 1 : 0))
            return false;
    }

    if (controls.USE_3B_CHEBY && !h_trips.empty()) {
        if (!lsq_gpu_launch_deriv_3b(
                (int)h_trips.size(), nparams, natoms,
                h_trips.data(),
                trip_clusters.data(), (int)trip_clusters.size(),
                trip_terms.data(), (int)trip_terms.size(),
                nlist.PERM_SCALE[3], cheby.DERIV_CONST, fit_stress,
                controls.FIT_ENER ? 1 : 0))
            return false;
    }

    if (controls.USE_4B_CHEBY && !h_quads.empty()) {
        if (!lsq_gpu_launch_deriv_4b(
                (int)h_quads.size(), nparams, natoms,
                h_quads.data(),
                quad_clusters.data(), (int)quad_clusters.size(),
                quad_terms.data(), (int)quad_terms.size(),
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
