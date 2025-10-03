#ifndef LTC_H
#define LTC_H
#include <algorithm>
#include "linalg.h"
#include "ltc.inc"
using namespace linalg::aliases;

/*
Credit:
    Original authors — credit is appreciated but not required:

        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
        2020, 2021, 2022, 2023 — Vladimír Vondruš <mosra@centrum.cz>
        2017 — Jonathan Hale <squareys@googlemail.com>, based on "Real-Time
            Polygonal-Light Shading with Linearly Transformed Cosines", by Eric
            Heitz et al, https://eheitzresearch.wordpress.com/415-2/

    This is free and unencumbered software released into the public domain.

    Anyone is free to copy, modify, publish, use, compile, sell, or distribute
    this software, either in source code form or as a compiled binary, for any
    purpose, commercial or non-commercial, and by any means.

    In jurisdictions that recognize copyright laws, the author or authors of
    this software dedicate any and all copyright interest in the software to
    the public domain. We make this dedication for the benefit of the public
    at large and to the detriment of our heirs and successors. We intend this
    dedication to be an overt act of relinquishment in perpetuity of all
    present and future rights to this software under copyright law.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Code slightly modified by Kevin Lee for CS488 Final Project
*/

#if 1
float3x3 M_GGX(const float theta, const float alpha)
{
    const int t = std::max<int>(0, std::min<int>(size-1, (int)floorf(theta / (0.5f*3.14159f) * size + 0.5f / size)));
    const int a = std::max<int>(0, std::min<int>(size-1, (int)floorf(sqrtf(alpha) * size + 0.5f / size)));
    const float* m = tabM[a + t*size];
    float3x3 res = {
        {m[0], m[1], m[2]},
        {m[3], m[4], m[5]},
        {m[6], m[7], m[8]}
    };
    return res;
}
float3x3 Minv_GGX(const float theta, const float alpha, bool antialias = true)
{
    float u = theta / (0.5f * 3.14159f) * size + 0.5f / size;
    float v = sqrtf(alpha) * size + 0.5f / size;
    const int t = std::max<int>(0, std::min<int>(size-1, (int)floorf(theta / (0.5f*3.14159f) * size + 0.5f/size)));
    const int a = std::max<int>(0, std::min<int>(size-1, (int)floorf(sqrtf(alpha) * size + 0.5f / size)));
    // printf("%d %d\n", t, a);
    const float* m = tabMinv[a + t*size];
    float3x3 res = {
        {m[0], m[1], m[2]},
        {m[3], m[4], m[5]},
        {m[6], m[7], m[8]}
    };
    if (!antialias) {
        return res;
    }
    if (0 < u && u < size - 1 && t < size - 1) {
        const float* m2 = tabMinv[a + (t + 1) * size];
        float3x3 res2 = {
            {m2[0], m2[1], m2[2]},
            {m2[3], m2[4], m2[5]},
            {m2[6], m2[7], m2[8]}
        };
        float c = u - t;
        return res * (1.0f - c) + res2 * (c);
    }
    return res;
}

float amplitude_GGX(const float theta, const float alpha)
{
    const int t = std::max<int>(0, std::min<int>(size-1, (int)floorf(theta / (0.5f*3.14159f) * size + 0.5f / size)));
    const int a = std::max<int>(0, std::min<int>(size-1, (int)floorf(sqrtf(alpha) * size + 0.5f / size)));

    return tabAmplitude[a + t*size];
}

/* Integrate between two edges on a clamped cosine distribution */
float integrateEdge(const float3 v1, const float3 v2) {
    float cosTheta = dot(v1, v2);
    cosTheta = cosTheta < -0.9999f ? -0.9999f : (cosTheta > 0.9999f ? 0.9999f : cosTheta);

    const float theta = acos(cosTheta);
    /* For theta <= 0.001 `theta/sin(theta)` is approximated as 1.0 */
    const float res = cross(v1, v2).z*((theta > 0.001f) ? theta/sin(theta) : 1.0f);
    return res;
}

int clipQuadToHorizon(float3 L[5]) {
    /* Detect clipping config */
    int config = 0;
    if(L[0].z > 0.0) config += 1;
    if(L[1].z > 0.0) config += 2;
    if(L[2].z > 0.0) config += 4;
    if(L[3].z > 0.0) config += 8;

    int n = 0;

    if(config == 0) {
        // clip all
    } else if(config == 1) { // V1 clip V2 V3 V4
        n = 3;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[3].z * L[0] + L[0].z * L[3];
    } else if(config == 2) { // V2 clip V1 V3 V4
        n = 3;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    } else if(config == 3) { // V1 V2 clip V3 V4
        n = 4;
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
        L[3] = -L[3].z * L[0] + L[0].z * L[3];
    } else if(config == 4) { // V3 clip V1 V2 V4
        n = 3;
        L[0] = -L[3].z * L[2] + L[2].z * L[3];
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
    } else if(config == 5) { // V1 V3 clip V2 V4, impossible
        n = 0;
    } else if(config == 6) { // V2 V3 clip V1 V4
        n = 4;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    } else if(config == 7) { // V1 V2 V3 clip V4
        n = 5;
        L[4] = -L[3].z * L[0] + L[0].z * L[3];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    } else if(config == 8) { // V4 clip V1 V2 V3
        n = 3;
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
        L[1] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] =  L[3];
    } else if(config == 9) { // V1 V4 clip V2 V3
        n = 4;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[2].z * L[3] + L[3].z * L[2];
    } else if(config == 10) { // V2 V4 clip V1 V3, impossible
        n = 0;
    } else if(config == 11) { // V1 V2 V4 clip V3
        n = 5;
        L[4] = L[3];
        L[3] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    } else if(config == 12) { // V3 V4 clip V1 V2
        n = 4;
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
    } else if(config == 13) { // V1 V3 V4 clip V2
        n = 5;
        L[4] = L[3];
        L[3] = L[2];
        L[2] = -L[1].z * L[2] + L[2].z * L[1];
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
    } else if(config == 14) { // V2 V3 V4 clip V1
        n = 5;
        L[4] = -L[0].z * L[3] + L[3].z * L[0];
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
    } else if(config == 15) { // V1 V2 V3 V4
        n = 4;
    }

    if(n == 3)
        L[3] = L[0];
    if(n == 4)
        L[4] = L[0];

    return n;
}

/*
 * Get intensity of light from the arealight given by `points` at the point `P`
 * with normal `N` when viewed from direction `P`.
 * @param N Normal
 * @param V View Direction
 * @param P Vertex Position
 * @param Minv Matrix to transform from BRDF distribution to clamped cosine distribution
 * @param points Light quad vertices
 * @param twoSided Whether the light is two sided
 */
float3 ltcEvaluate(float3 N, float3 V, float3 P, float3x3 Minv, float3 points[4], bool twoSided) {
    /* Construct orthonormal basis around N */
    const float3 T1 = normalize(V - N*dot(V, N));
    const float3 T2 = cross(N, T1);

    /* Rotate area light in (T1, T2, R) basis */
    Minv = mul(Minv, transpose(float3x3(T1, T2, N)));

    /* Allocate 5 vertices for polygon (one additional which may result from
     * clipping) */
    float3 L[5];
    L[0] = mul(Minv, (points[0] - P));
    L[1] = mul(Minv, (points[1] - P));
    L[2] = mul(Minv, (points[2] - P));
    L[3] = mul(Minv, (points[3] - P));

    /* Clip light quad so that the part behind the surface does not affect the
     * lighting of the point */
    int n = clipQuadToHorizon(L);
    if(n == 0)
        return float3(0.0f);

    // project onto sphere
    L[0] = normalize(L[0]);
    L[1] = normalize(L[1]);
    L[2] = normalize(L[2]);
    L[3] = normalize(L[3]);
    L[4] = normalize(L[4]);

    /* Integrate over the clamped cosine distribution in the domain of the
     * transformed light polygon */
    float sum = integrateEdge(L[0], L[1])
              + integrateEdge(L[1], L[2])
              + integrateEdge(L[2], L[3]);
    if(n >= 4)
        sum += integrateEdge(L[3], L[4]);
    if(n == 5)
        sum += integrateEdge(L[4], L[0]);

    /* Negated due to winding order */
    sum = twoSided ? abs(sum) : std::max(0.0f, -sum);

    return float3(sum);
}
#endif

#endif //LTC_H
