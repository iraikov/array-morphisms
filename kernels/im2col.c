/* im2col.c -- C implementation of batched matmul-ready im2col for f32.
 *
 * Layout:
 *   src: NCHW row-major, shape [N, C, H, W]
 *   col: row-major, shape [N*OH*OW, C*KH*KW]
 *   For each output spatial location (n, oh, ow) the row index is
 *       m = n*OH*OW + oh*OW + ow
 *   and the columns are filled in order (c, kh, kw).
 */

#include <stdlib.h>
#include <string.h>

void im2col_batched_mr_f32(float *col,
                           const float *src,
                           int N, int C, int H, int W,
                           int KH, int KW, int SH, int SW,
                           int PH, int PW,
                           int OH, int OW)
{
    const int fan_in = C * KH * KW;
    const int C_H_W = C * H * W;
    const int H_W   = H * W;

    for (int n = 0; n < N; ++n) {
        const float *src_n = src + (long)n * C_H_W;
        for (int oh = 0; oh < OH; ++oh) {
            int ih0 = oh * SH - PH;
            for (int ow = 0; ow < OW; ++ow) {
                int iw0 = ow * SW - PW;
                float *row = col + ((long)n * OH * OW + (long)oh * OW + ow) * fan_in;
                int col_idx = 0;
                for (int c = 0; c < C; ++c) {
                    const float *src_c = src_n + (long)c * H_W;
                    for (int kh = 0; kh < KH; ++kh) {
                        int ih = ih0 + kh;
                        if (ih < 0 || ih >= H) {
                            col_idx += KW;
                            continue;
                        }
                        const float *src_h = src_c + (long)ih * W;
                        for (int kw = 0; kw < KW; ++kw) {
                            int iw = iw0 + kw;
                            row[col_idx++] = (iw >= 0 && iw < W) ? src_h[iw] : 0.0f;
                        }
                    }
                }
            }
        }
    }
}

void bias_add_f32(float *out, const float *b, int M, int out_ch)
{
    for (int m = 0; m < M; ++m) {
        float *row = out + (long)m * out_ch;
        for (int co = 0; co < out_ch; ++co) {
            row[co] += b[co];
        }
    }
}

/* col2im_batched_mr_f32 -- Scatter a [N*OH*OW, C*KH*KW] matmul-ready column
 * buffer back into an NCHW image.  This is the inverse of im2col_batched_mr_f32
 * and is used for conv backward-data.
 */
void col2im_batched_mr_f32(float *dx,
                           const float *col,
                           int N, int C, int H, int W,
                           int KH, int KW, int SH, int SW,
                           int PH, int PW,
                           int OH, int OW)
{
    const int fan_in = C * KH * KW;
    const int C_H_W = C * H * W;
    const int H_W   = H * W;

    for (int i = 0; i < N * C_H_W; ++i) dx[i] = 0.0f;

    for (int n = 0; n < N; ++n) {
        float *dx_n = dx + (long)n * C_H_W;
        for (int oh = 0; oh < OH; ++oh) {
            int ih0 = oh * SH - PH;
            for (int ow = 0; ow < OW; ++ow) {
                int iw0 = ow * SW - PW;
                const float *row = col + ((long)n * OH * OW + (long)oh * OW + ow) * fan_in;
                int col_idx = 0;
                for (int c = 0; c < C; ++c) {
                    float *dx_c = dx_n + (long)c * H_W;
                    for (int kh = 0; kh < KH; ++kh) {
                        int ih = ih0 + kh;
                        if (ih < 0 || ih >= H) {
                            col_idx += KW;
                            continue;
                        }
                        float *dx_h = dx_c + (long)ih * W;
                        for (int kw = 0; kw < KW; ++kw) {
                            int iw = iw0 + kw;
                            if (iw >= 0 && iw < W) {
                                dx_h[iw] += row[col_idx];
                            }
                            ++col_idx;
                        }
                    }
                }
            }
        }
    }
}
