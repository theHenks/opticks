"""
Random number generation (minimal version).
"""

"""
    OpticalRNG

Simple random number generator for testing.
"""
mutable struct OpticalRNG
    seed::UInt64
    counter::UInt32
    
    function OpticalRNG(photon_idx::UInt32 = 0x00000000, event_idx::UInt32 = 0x00000000)
        seed = UInt64(42 + photon_idx + 10000 * event_idx)
        new(seed, 0x00000000)
    end
end

"""
    uniform(rng::OpticalRNG)

Generate uniform random float in [0,1).
"""
function uniform(rng::OpticalRNG)::Float32
    # Simple LCG for testing
    rng.counter += 1
    x = rng.seed + rng.counter
    x = x ⊻ (x >> 12)
    x = x ⊻ (x << 25)
    x = x ⊻ (x >> 27)
    return Float32((x * 0x2545F4914F6CDD1D) >> 32) / Float32(2^32)
end