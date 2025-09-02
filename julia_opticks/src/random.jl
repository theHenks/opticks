"""
Random number generation utilities for optical photon simulation.

Julia equivalent of qrng.h with CUDA random number generation.
"""

using CUDA
using Random

# Julia equivalent of curandState using CUDA.jl's RNG
const CUDARandomState = CUDA.RNG.Random123.PhiloxState

"""
    OpticalRNG

Random number generator state for optical photon simulation.
Manages CUDA random number generation with proper seeding and skip-ahead.
"""
mutable struct OpticalRNG
    state::CUDARandomState
    photon_idx::UInt32
    event_idx::UInt32
    skipahead_offset::UInt32
    
    function OpticalRNG(photon_idx::UInt32 = 0x00000000, event_idx::UInt32 = 0x00000000)
        # Default skipahead offset to prevent correlation between events
        skipahead_offset = parse(UInt32, get(ENV, "OPTICKS_EVENT_SKIPAHEAD", "10000"))
        
        # Initialize with Philox4x32 generator (similar to curand)
        seed = UInt64(42)  # Simulation seed constant
        subsequence = UInt64(photon_idx)
        offset = UInt64(skipahead_offset * event_idx)
        
        state = CUDA.RNG.Random123.PhiloxState(seed, subsequence, offset)
        
        new(state, photon_idx, event_idx, skipahead_offset)
    end
end

"""
    uniform(rng::OpticalRNG)

Generate uniform random float in [0,1).
"""
function uniform(rng::OpticalRNG)::Float32
    return Float32(CUDA.rand(rng.state))
end

"""
    uniform_sphere(u0::Float32, u1::Float32)

Generate uniform random direction on unit sphere using two uniform random numbers.
"""
function uniform_sphere(u0::Float32, u1::Float32)
    # Convert uniform randoms to spherical coordinates
    cos_theta = 2.0f0 * u0 - 1.0f0  # cos(θ) ∈ [-1,1]
    sin_theta = sqrt(1.0f0 - cos_theta * cos_theta)
    phi = 2.0f0 * π * u1  # φ ∈ [0,2π]
    
    cos_phi = cos(phi)
    sin_phi = sin(phi)
    
    return SVector{3,Float32}(
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta
    )
end

"""
    uniform_sphere(rng::OpticalRNG)

Generate uniform random direction on unit sphere.
"""
function uniform_sphere(rng::OpticalRNG)
    u0 = uniform(rng)
    u1 = uniform(rng)
    return uniform_sphere(u0, u1)
end

"""
    gauss_shoot(rng::OpticalRNG, mean::Float32, stddev::Float32)

Generate Gaussian distributed random number.
Julia equivalent of RandGaussQ_shoot from Geant4.
"""
function gauss_shoot(rng::OpticalRNG, mean::Float32, stddev::Float32)::Float32
    # Box-Muller transform
    u1 = uniform(rng)
    u2 = uniform(rng)
    
    # Ensure u1 > 0 to avoid log(0)
    while u1 <= 0.0f0
        u1 = uniform(rng)
    end
    
    z = sqrt(-2.0f0 * log(u1)) * cos(2.0f0 * π * u2)
    return mean + stddev * z
end

"""
    exponential_shoot(rng::OpticalRNG, mean::Float32)

Generate exponentially distributed random number.
"""
function exponential_shoot(rng::OpticalRNG, mean::Float32)::Float32
    u = uniform(rng)
    # Ensure u > 0 to avoid log(0)
    while u <= 0.0f0
        u = uniform(rng)
    end
    return -mean * log(u)
end

"""
    lambertian_direction!(dir::Ref{SVector{3,Float32}}, normal::SVector{3,Float32}, 
                         orient::Float32, rng::OpticalRNG)

Generate random direction with Lambertian (cosine) distribution.
Direction is relative to the given normal vector with specified orientation.
"""
function lambertian_direction!(dir::Ref{SVector{3,Float32}}, normal::SVector{3,Float32}, 
                              orient::Float32, rng::OpticalRNG)
    # Generate random direction in hemisphere
    u1 = uniform(rng)
    u2 = uniform(rng)
    
    # Cosine-weighted hemisphere sampling
    cos_theta = sqrt(u1)  # cos(θ) weighted by cos(θ)
    sin_theta = sqrt(1.0f0 - u1)
    phi = 2.0f0 * π * u2
    
    cos_phi = cos(phi)
    sin_phi = sin(phi)
    
    # Local hemisphere direction
    local_dir = SVector{3,Float32}(
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta * orient  # Apply orientation
    )
    
    # Transform from local coordinates aligned with normal to world coordinates
    # This requires building a coordinate frame from the normal
    # For simplicity, use Gram-Schmidt to build orthonormal basis
    
    # Find a vector not parallel to normal
    up = abs(normal[3]) < 0.999f0 ? SVector{3,Float32}(0,0,1) : SVector{3,Float32}(1,0,0)
    
    # Build orthonormal basis
    right = normalize(cross(up, normal))
    forward = cross(normal, right)
    
    # Transform to world coordinates
    world_dir = local_dir[1] * right + local_dir[2] * forward + local_dir[3] * normal
    
    dir[] = normalize(world_dir)
end

"""
    random_direction_marsaglia!(dir::Ref{SVector{3,Float32}}, rng::OpticalRNG)

Generate uniformly random direction using Marsaglia method.
"""
function random_direction_marsaglia!(dir::Ref{SVector{3,Float32}}, rng::OpticalRNG)
    # Marsaglia method for uniform sphere sampling
    while true
        x1 = 2.0f0 * uniform(rng) - 1.0f0
        x2 = 2.0f0 * uniform(rng) - 1.0f0
        s = x1*x1 + x2*x2
        
        if s < 1.0f0
            sqrt_term = sqrt(1.0f0 - s)
            dir[] = SVector{3,Float32}(
                2.0f0 * x1 * sqrt_term,
                2.0f0 * x2 * sqrt_term,
                1.0f0 - 2.0f0 * s
            )
            break
        end
    end
end

# CUDA kernel versions for GPU execution
"""
    init_rng_states_kernel!(states, photon_offset, event_idx, skipahead_offset)

CUDA kernel to initialize random number generator states for photons.
"""
function init_rng_states_kernel!(states::CuDeviceArray{OpticalRNG,1}, 
                                 photon_offset::UInt32, event_idx::UInt32,
                                 skipahead_offset::UInt32)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(states)
        photon_idx = photon_offset + UInt32(idx - 1)
        states[idx] = OpticalRNG(photon_idx, event_idx)
    end
    return nothing
end

"""
    init_rng_states!(states::CuArray{OpticalRNG,1}, photon_offset::UInt32, 
                     event_idx::UInt32, skipahead_offset::UInt32)

Initialize random number generator states for GPU simulation.
"""
function init_rng_states!(states::CuArray{OpticalRNG,1}, photon_offset::UInt32, 
                         event_idx::UInt32, skipahead_offset::UInt32 = 10000)
    if length(states) == 0
        return
    end
    
    threads = 256
    blocks = cld(length(states), threads)
    
    @cuda threads=threads blocks=blocks init_rng_states_kernel!(states, photon_offset, event_idx, skipahead_offset)
    CUDA.synchronize()
end