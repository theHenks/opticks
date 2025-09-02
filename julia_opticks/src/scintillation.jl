"""
Scintillation physics implementation.

Julia equivalent of QScint.cu and qscint.h for generating
scintillation photons in optical media.
"""

using CUDA
using LinearAlgebra

"""
    ScintillationGenstep

Generation step information for scintillation.
Equivalent to sscint struct.
"""
struct ScintillationGenstep
    # Control parameters
    gentype::UInt32      # Generation type identifier
    trackid::UInt32      # Parent track ID
    matline::UInt32      # Material line index
    numphoton::UInt32    # Number of photons to generate
    
    # Position and timing
    pos::SVector{3,Float32}     # Position
    time::Float32               # Time
    
    # Step information
    delta_position::SVector{3,Float32}  # Step displacement vector
    step_length::Float32                # Step length
    
    # Particle properties
    code::Int32          # Particle code
    charge::Float32      # Particle charge
    weight::Float32      # Statistical weight
    
    # Scintillation-specific
    scintillation_time::Float32  # Characteristic scintillation time
    yield_ratio::Float32         # Slow/fast component ratio
    
    function ScintillationGenstep()
        new(0, 0, 0, 0,
            SVector{3,Float32}(0,0,0), 0.0f0,
            SVector{3,Float32}(0,0,0), 0.0f0,
            0, 0.0f0, 1.0f0,
            1.0f0, 1.0f0)
    end
end

"""
    ScintillationProperties

Properties for scintillation materials including emission spectrum
and timing characteristics.
"""
struct ScintillationProperties
    emission_spectrum::Vector{Float32}    # Emission probability vs wavelength
    wavelengths::Vector{Float32}          # Wavelength grid
    fast_time_constant::Float32           # Fast component time constant (ns)
    slow_time_constant::Float32           # Slow component time constant (ns)
    yield_ratio::Float32                  # Slow/fast yield ratio
    birks_constant::Float32               # Birks quenching constant
    
    function ScintillationProperties(emission::Vector{Float32}, wavelengths::Vector{Float32},
                                   fast_time::Float32 = 1.0f0, slow_time::Float32 = 10.0f0,
                                   yield_ratio::Float32 = 1.0f0, birks::Float32 = 0.01f0)
        new(emission, wavelengths, fast_time, slow_time, yield_ratio, birks)
    end
end

"""
    QScint

Scintillation photon generation on GPU.
"""
struct QScint
    # Material scintillation properties (on GPU)
    emission_spectra::CuArray{Float32,2}    # [material, wavelength]
    wavelengths::CuArray{Float32,1}         # Wavelength grid
    cumulative_spectra::CuArray{Float32,2}  # Cumulative distribution for sampling
    
    # Timing properties
    fast_time_constants::CuArray{Float32,1}  # Per material
    slow_time_constants::CuArray{Float32,1}  # Per material
    yield_ratios::CuArray{Float32,1}         # Per material
    
    function QScint(properties::Vector{ScintillationProperties})
        num_materials = length(properties)
        if num_materials == 0
            error("Must provide at least one scintillation material")
        end
        
        # Assume all materials use same wavelength grid
        wavelengths = properties[1].wavelengths
        num_wavelengths = length(wavelengths)
        
        # Build emission spectra matrix
        emission_matrix = zeros(Float32, num_materials, num_wavelengths)
        fast_times = zeros(Float32, num_materials)
        slow_times = zeros(Float32, num_materials)
        yields = zeros(Float32, num_materials)
        
        for (i, prop) in enumerate(properties)
            emission_matrix[i, :] = prop.emission_spectrum
            fast_times[i] = prop.fast_time_constant
            slow_times[i] = prop.slow_time_constant
            yields[i] = prop.yield_ratio
        end
        
        # Compute cumulative distributions
        cumulative = compute_cumulative_emission(emission_matrix)
        
        # Upload to GPU
        gpu_emission = CuArray(emission_matrix)
        gpu_wavelengths = CuArray(wavelengths)
        gpu_cumulative = CuArray(cumulative)
        gpu_fast_times = CuArray(fast_times)
        gpu_slow_times = CuArray(slow_times)
        gpu_yields = CuArray(yields)
        
        new(gpu_emission, gpu_wavelengths, gpu_cumulative,
            gpu_fast_times, gpu_slow_times, gpu_yields)
    end
end

"""
    compute_cumulative_emission(emission_spectra)

Compute cumulative distribution functions for emission spectra.
"""
function compute_cumulative_emission(emission_spectra::Array{Float32,2})
    num_materials, num_wavelengths = size(emission_spectra)
    cumulative = zeros(Float32, num_materials, num_wavelengths)
    
    for i in 1:num_materials
        cumulative[i, 1] = 0.0f0
        for j in 2:num_wavelengths
            cumulative[i, j] = cumulative[i, j-1] + emission_spectra[i, j]
        end
        
        # Normalize to [0,1]
        if cumulative[i, end] > 0.0f0
            cumulative[i, :] ./= cumulative[i, end]
        end
    end
    
    return cumulative
end

"""
    sample_scintillation_wavelength(rng::OpticalRNG, scint::QScint, material::Int32)

Sample wavelength from scintillation emission spectrum.
"""
function sample_scintillation_wavelength(rng::OpticalRNG, scint::QScint, material::Int32)::Float32
    xi = uniform(rng)
    
    # Binary search in cumulative distribution
    low = 1
    high = length(scint.wavelengths)
    
    while high - low > 1
        mid = (low + high) ÷ 2
        if scint.cumulative_spectra[material, mid] < xi
            low = mid
        else
            high = mid
        end
    end
    
    # Linear interpolation
    if high <= length(scint.wavelengths) && low >= 1
        frac = (xi - scint.cumulative_spectra[material, low]) / 
               (scint.cumulative_spectra[material, high] - scint.cumulative_spectra[material, low])
        return scint.wavelengths[low] + frac * (scint.wavelengths[high] - scint.wavelengths[low])
    else
        return scint.wavelengths[low]
    end
end

"""
    sample_scintillation_time(rng::OpticalRNG, scint::QScint, material::Int32)

Sample scintillation photon emission time using exponential decay.
"""
function sample_scintillation_time(rng::OpticalRNG, scint::QScint, material::Int32)::Float32
    # Choose between fast and slow component
    yield_ratio = scint.yield_ratios[material]
    xi_component = uniform(rng)
    
    if xi_component < yield_ratio / (1.0f0 + yield_ratio)
        # Fast component
        time_constant = scint.fast_time_constants[material]
    else
        # Slow component
        time_constant = scint.slow_time_constants[material]
    end
    
    # Sample from exponential distribution
    return exponential_shoot(rng, time_constant)
end

"""
    generate_scintillation_photon!(photon::SPhoton, rng::OpticalRNG, 
                                  genstep::ScintillationGenstep, scint::QScint,
                                  photon_idx::Int32, genstep_id::Int32)

Generate a single scintillation photon.
"""
function generate_scintillation_photon!(photon::SPhoton, rng::OpticalRNG,
                                       genstep::ScintillationGenstep, scint::QScint,
                                       photon_idx::Int32, genstep_id::Int32)
    # Sample wavelength from emission spectrum
    wavelength = sample_scintillation_wavelength(rng, scint, Int32(genstep.matline))
    
    # Sample emission time
    emission_time = sample_scintillation_time(rng, scint, Int32(genstep.matline))
    
    # Sample position uniformly along step
    xi_pos = uniform(rng)
    pos = genstep.pos + xi_pos * genstep.delta_position
    time = genstep.time + emission_time
    
    # Generate isotropic direction
    direction = uniform_sphere(rng)
    
    # Generate random polarization perpendicular to direction
    # Find a vector not parallel to direction
    if abs(direction[3]) < 0.9f0
        perp1 = normalize(cross(direction, SVector{3,Float32}(0, 0, 1)))
    else
        perp1 = normalize(cross(direction, SVector{3,Float32}(1, 0, 0)))
    end
    perp2 = cross(direction, perp1)
    
    # Random polarization angle
    pol_angle = 2.0f0 * π * uniform(rng)
    polarization = cos(pol_angle) * perp1 + sin(pol_angle) * perp2
    
    # Set photon properties
    photon.pos = pos
    photon.time = time
    photon.mom = direction
    photon.pol = polarization
    photon.wavelength = wavelength
    
    # Set photon index and clear flags
    set_photon_idx!(photon, UInt32(photon_idx))
    photon.boundary_flag = 0x00000000
    photon.identity = 0x00000000
    photon.flagmask = 0x00000000
    photon.iindex = 0x00000000
end

"""
    reemit_scintillation_photon!(photon::SPhoton, rng::OpticalRNG, scint::QScint,
                                 material::Int32, scintillation_time::Float32)

Reemit a scintillation photon (for wavelength shifting, etc.).
"""
function reemit_scintillation_photon!(photon::SPhoton, rng::OpticalRNG, scint::QScint,
                                     material::Int32, scintillation_time::Float32)
    # Keep current position and add scintillation delay
    photon.time += scintillation_time
    
    # Sample new wavelength from emission spectrum
    photon.wavelength = sample_scintillation_wavelength(rng, scint, material)
    
    # Generate new isotropic direction
    photon.mom = uniform_sphere(rng)
    
    # Generate new random polarization
    if abs(photon.mom[3]) < 0.9f0
        perp1 = normalize(cross(photon.mom, SVector{3,Float32}(0, 0, 1)))
    else
        perp1 = normalize(cross(photon.mom, SVector{3,Float32}(1, 0, 0)))
    end
    perp2 = cross(photon.mom, perp1)
    
    pol_angle = 2.0f0 * π * uniform(rng)
    photon.pol = cos(pol_angle) * perp1 + sin(pol_angle) * perp2
end

"""
    scintillation_generation_kernel!(photons, rngs, gensteps, scint, photon_offset)

CUDA kernel for generating scintillation photons.
"""
function scintillation_generation_kernel!(photons::CuDeviceArray{SPhoton,1},
                                         rngs::CuDeviceArray{OpticalRNG,1},
                                         gensteps::CuDeviceArray{ScintillationGenstep,1},
                                         scint::QScint,
                                         photon_offset::Int32)
    photon_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if photon_idx <= length(photons)
        # Find which genstep this photon belongs to
        cumulative = 0
        genstep_id = 1
        for i in 1:length(gensteps)
            if photon_idx <= cumulative + gensteps[i].numphoton
                genstep_id = i
                break
            end
            cumulative += gensteps[i].numphoton
        end
        
        if genstep_id <= length(gensteps)
            generate_scintillation_photon!(photons[photon_idx], rngs[photon_idx],
                                          gensteps[genstep_id], scint,
                                          Int32(photon_idx + photon_offset), Int32(genstep_id))
        end
    end
    return nothing
end

"""
    generate_scintillation_photons!(photons::CuArray{SPhoton,1}, rngs::CuArray{OpticalRNG,1},
                                   gensteps::CuArray{ScintillationGenstep,1}, scint::QScint,
                                   photon_offset::Int32 = 0)

Generate scintillation photons from generation steps on GPU.
"""
function generate_scintillation_photons!(photons::CuArray{SPhoton,1}, rngs::CuArray{OpticalRNG,1},
                                        gensteps::CuArray{ScintillationGenstep,1}, scint::QScint,
                                        photon_offset::Int32 = 0)
    if length(photons) == 0
        return
    end
    
    threads = 256
    blocks = cld(length(photons), threads)
    
    @cuda threads=threads blocks=blocks scintillation_generation_kernel!(photons, rngs, gensteps, scint, photon_offset)
    CUDA.synchronize()
end