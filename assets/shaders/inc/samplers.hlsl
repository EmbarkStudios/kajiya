#ifndef SAMPLERS_HLSL
#define SAMPLERS_HLSL

[[vk::binding(32)]] SamplerState sampler_lnc;
[[vk::binding(33)]] SamplerState sampler_llr;
[[vk::binding(34)]] SamplerState sampler_nnc;
[[vk::binding(35)]] SamplerState sampler_llc;

#endif