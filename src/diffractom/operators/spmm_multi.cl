// spmm_pf_adjoint_c for NRHS data arrays at once (SinglePhaseForwardOperator.adjoint_multi_cl): the CSR entries of a
// batch are read once for all of them. For every array, detector position and orientation the products are summed in
// the same order as spmm_pf_adjoint_c (the row's entries in CSR order), so every output equals that kernel's bitwise;
// TYM (detector positions per work-group) and the local size only change how the work is split.
//     s_n[r, y, k] = alpha * sum_j pf[r, k, j] * d_n[r, y, j]    for k < Kb, n < NRHS
__kernel void spmm_pf_adjoint_multi(
    __global const float *d0, __global const float *d1, __global const float *d2, __global const float *d3,
    __global const int *row_ptr, __global const ushort *col_j, __global const float *val,
    __global float *s0, __global float *s1, __global float *s2, __global float *s3,
    const int R, const int My, const int Kb, const int CP, const int Kstride, const float alpha,
    const int JC, __local float *ldata,   // NRHS * TYM * JC floats
    __global const int *yrange, const int use_yrange)   // (R, 2): detector positions that are needed (others left unwritten)
{
    const int lid = get_local_id(0), ls = get_local_size(0);
    const int k = get_group_id(0) * ls + lid;
    const int y0 = get_group_id(1) * TYM;
    const int r = get_group_id(2);
    if (use_yrange && (y0 > yrange[2 * r + 1] || y0 + TYM <= yrange[2 * r])) return;   // whole work-group
    const int ny = min(TYM, My - y0);
    const size_t line0 = (size_t)r * My + y0;
    __global const float *dd[4] = {d0, d1, d2, d3};
    __global float *ss[4] = {s0, s1, s2, s3};
    int i = 0, e = 0;
    if (k < Kb) { i = row_ptr[r * Kb + k]; e = row_ptr[r * Kb + k + 1]; }
    float acc[NRHS][TYM];
    #pragma unroll
    for (int n = 0; n < NRHS; ++n)
        #pragma unroll
        for (int t = 0; t < TYM; ++t) acc[n][t] = 0.0f;
    for (int j0 = 0; j0 < CP; j0 += JC) {
        const int jn = min(JC, CP - j0);
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int n = 0; n < NRHS; ++n)
            for (int t = 0; t < ny; ++t) {
                __global const float *src = dd[n] + (line0 + t) * CP + j0;
                for (int jj = lid; jj < jn; jj += ls) ldata[(n * TYM + t) * JC + jj] = src[jj];
            }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int j1 = j0 + jn;
        for (; i < e; ++i) {
            const int j = col_j[i];
            if (j >= j1) break;
            const float v = val[i];
            #pragma unroll
            for (int n = 0; n < NRHS; ++n)
                #pragma unroll
                for (int t = 0; t < TYM; ++t)
                    if (t < ny) acc[n][t] += v * ldata[(n * TYM + t) * JC + (j - j0)];
        }
    }
    if (k < Kb) {
        #pragma unroll
        for (int n = 0; n < NRHS; ++n)
            #pragma unroll
            for (int t = 0; t < TYM; ++t)
                if (t < ny) ss[n][(line0 + t) * Kstride + k] = alpha * acc[n][t];
    }
}
