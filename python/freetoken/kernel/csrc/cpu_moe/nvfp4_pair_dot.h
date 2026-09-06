// Included beside the ordinary AVX-512 NVFP4 dot inside its ISA guard.
// Each input retains the ordinary dot's four accumulation chains and epilogue.
struct Nvfp4PairGroup {
  __m512 first;
  __m512 second;
};

__attribute__((target("avx512f,avx512bw,avx512vnni,avx2")))
static inline Nvfp4PairGroup nvfp4_i8_pair_grp4(
    const uint8_t* packed, const uint8_t* scale, const int8_t* a0,
    const int8_t* a1, const float* e4m3, const float* as0, const float* as1,
    int b, __m512i lut, __m512i idx, __m512i mask0F, __m512i idxsc) {
  const __m256i raw = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(packed + static_cast<size_t>(b) * 8));
  const __m512i src = _mm512_permutexvar_epi64(idx, _mm512_castsi256_si512(raw));
  const __m512i lo = _mm512_and_si512(src, mask0F);
  const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(src, 4), mask0F);
  const __m512i w = _mm512_shuffle_epi8(
      lut, _mm512_mask_blend_epi8(0xFF00FF00FF00FF00ULL, lo, hi));
  const __m512i aw = _mm512_abs_epi8(w);
  const __mmask64 neg = _mm512_movepi8_mask(w);
  int sc_raw;
  memcpy(&sc_raw, scale + b, 4);
  const __m128 ws = _mm_i32gather_ps(
      e4m3, _mm_cvtepu8_epi32(_mm_cvtsi32_si128(sc_raw)), 4);
  const __m512i x0 = _mm512_loadu_si512(
      reinterpret_cast<const __m512i*>(a0 + static_cast<size_t>(b) * 16));
  const __m512i x1 = _mm512_loadu_si512(
      reinterpret_cast<const __m512i*>(a1 + static_cast<size_t>(b) * 16));
  const __m512i d0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), aw,
      _mm512_mask_sub_epi8(x0, neg, _mm512_setzero_si512(), x0));
  const __m512i d1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), aw,
      _mm512_mask_sub_epi8(x1, neg, _mm512_setzero_si512(), x1));
  const __m512 s0 = _mm512_permutexvar_ps(idxsc,
      _mm512_castps128_ps512(_mm_mul_ps(ws, _mm_loadu_ps(as0 + b))));
  const __m512 s1 = _mm512_permutexvar_ps(idxsc,
      _mm512_castps128_ps512(_mm_mul_ps(ws, _mm_loadu_ps(as1 + b))));
  return {_mm512_mul_ps(_mm512_cvtepi32_ps(d0), s0),
          _mm512_mul_ps(_mm512_cvtepi32_ps(d1), s1)};
}

__attribute__((target("avx512f,avx512bw,avx512vnni,avx2")))
static void dot_nvfp4_i8_avx512vnni_pair(
    float* out, const uint8_t* packed, const uint8_t* scale, float global,
    const int8_t* a0, const int8_t* a1, int K, const float* e4m3,
    const float* as0, const float* as1) {
  const __m512i lut = _mm512_broadcast_i32x4(
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(kE2M1x2)));
  const __m512i idx = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);
  const __m512i mask0F = _mm512_set1_epi8(0x0F);
  const __m512i idxsc = _mm512_set_epi32(
      3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
  __m512 c02 = _mm512_setzero_ps(), c03 = _mm512_setzero_ps();
  __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
  __m512 c12 = _mm512_setzero_ps(), c13 = _mm512_setzero_ps();
  const int nb = K / 16;
  const int pfb = nvfp4_pf_blocks();
  const int pf = pfb < 0 ? std::min(512, 2 * nb) : pfb;
  int b = 0;
  for (; b + 16 <= nb; b += 16) {
    if (pf > 0) {
      _mm_prefetch(reinterpret_cast<const char*>(
          packed + (static_cast<size_t>(b) + pf) * 8), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(
          packed + (static_cast<size_t>(b) + pf) * 8 + 64), _MM_HINT_T0);
    }
    const auto g0 = nvfp4_i8_pair_grp4(
        packed, scale, a0, a1, e4m3, as0, as1, b, lut, idx, mask0F, idxsc);
    c00 = _mm512_add_ps(c00, g0.first);
    c10 = _mm512_add_ps(c10, g0.second);
    const auto g1 = nvfp4_i8_pair_grp4(
        packed, scale, a0, a1, e4m3, as0, as1, b + 4, lut, idx, mask0F, idxsc);
    c01 = _mm512_add_ps(c01, g1.first);
    c11 = _mm512_add_ps(c11, g1.second);
    const auto g2 = nvfp4_i8_pair_grp4(
        packed, scale, a0, a1, e4m3, as0, as1, b + 8, lut, idx, mask0F, idxsc);
    c02 = _mm512_add_ps(c02, g2.first);
    c12 = _mm512_add_ps(c12, g2.second);
    const auto g3 = nvfp4_i8_pair_grp4(
        packed, scale, a0, a1, e4m3, as0, as1, b + 12, lut, idx, mask0F, idxsc);
    c03 = _mm512_add_ps(c03, g3.first);
    c13 = _mm512_add_ps(c13, g3.second);
  }
  for (; b + 4 <= nb; b += 4) {
    const auto g = nvfp4_i8_pair_grp4(
        packed, scale, a0, a1, e4m3, as0, as1, b, lut, idx, mask0F, idxsc);
    c00 = _mm512_add_ps(c00, g.first);
    c10 = _mm512_add_ps(c10, g.second);
  }
  float s0 = _mm512_reduce_add_ps(
      _mm512_add_ps(_mm512_add_ps(c00, c01), _mm512_add_ps(c02, c03)));
  float s1 = _mm512_reduce_add_ps(
      _mm512_add_ps(_mm512_add_ps(c10, c11), _mm512_add_ps(c12, c13)));
  for (; b < nb; ++b) {
    const uint8_t* pk = packed + static_cast<size_t>(b) * 8;
    const int8_t* x0 = a0 + static_cast<size_t>(b) * 16;
    const int8_t* x1 = a1 + static_cast<size_t>(b) * 16;
    int i0 = 0, i1 = 0;
    for (int j = 0; j < 8; ++j) {
      i0 += (int)kE2M1x2[pk[j] & 0xF] * (int)x0[j]
          + (int)kE2M1x2[pk[j] >> 4] * (int)x0[8 + j];
      i1 += (int)kE2M1x2[pk[j] & 0xF] * (int)x1[j]
          + (int)kE2M1x2[pk[j] >> 4] * (int)x1[8 + j];
    }
    s0 += (e4m3[scale[b]] * as0[b]) * (float)i0;
    s1 += (e4m3[scale[b]] * as1[b]) * (float)i1;
  }
  out[0] = s0 * (0.5f * global);
  out[1] = s1 * (0.5f * global);
}
