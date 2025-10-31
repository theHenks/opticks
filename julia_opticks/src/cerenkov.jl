"""
Cerenkov radiation physics implementation.

Julia equivalent of QCerenkov.cu and qcerenkov.h for generating
Cerenkov photons in optical media.
"""

using CUDA
using LinearAlgebra

"""
    CerenkovGenstep

Generation step information for Cerenkov radiation.
Equivalent to scerenkov struct.
"""
struct CerenkovGenstep
    # Control parameters
    gentype::UInt32      # Generation type identifier
    trackid::UInt32      # Parent track ID
    matline::UInt32      # Material line index for boundary lookup  
    numphoton::UInt32    # Number of photons to generate
    
    # Position and timing
    pos::SVector{3,Float32}     # Initial position
    time::Float32               # Initial time
    
    # Step information
    delta_position::SVector{3,Float32}  # Step displacement vector
    step_length::Float32                # Step length
    
    # Particle properties
    code::Int32          # Particle code
    charge::Float32      # Particle charge
    weight::Float32      # Statistical weight
    pre_velocity::Float32 # Particle velocity before step
    
    # Cerenkov-specific
    beta_inverse::Float32 # 1/β where β = v/c
    
    function CerenkovGenstep()
        new(0, 0, 0, 0,
            SVector{3,Float32}(0,0,0), 0.0f0,
            SVector{3,Float32}(0,0,0), 0.0f0,
            0, 0.0f0, 1.0f0, 0.0f0,
            1.0f0)
    end
    
    function CerenkovGenstep(gentype::UInt32, trackid::UInt32, matline::UInt32, numphoton::UInt32,
                            pos::SVector{3,Float32}, time::Float32,
                            delta_position::SVector{3,Float32}, step_length::Float32,
                            code::Int32, charge::Float32, weight::Float32, pre_velocity::Float32,
                            beta_inverse::Float32)
        new(gentype, trackid, matline, numphoton,
            pos, time, delta_position, step_length,
            code, charge, weight, pre_velocity, beta_inverse)
    end
end

"""
    QCerenkov

Cerenkov photon generation on GPU.
Contains material properties and methods for photon generation.
"""
struct QCerenkov
    # Material properties (should be on GPU)
    rindex::CuArray{Float32,2}      # Refractive index [material, wavelength]
    absorption::CuArray{Float32,2}   # Absorption length [material, wavelength]  
    wavelengths::CuArray{Float32,1}  # Wavelength grid
    
    # Cerenkov integral tables for efficient sampling
    cerenkov_integral::CuArray{Float32,2}  # Integrated Cerenkov spectrum
    
    function QCerenkov(rindex::Array{Float32,2}, absorption::Array{Float32,2}, 
                       wavelengths::Array{Float32,1})
        # Upload to GPU
        gpu_rindex = CuArray(rindex)
        gpu_absorption = CuArray(absorption) 
        gpu_wavelengths = CuArray(wavelengths)
        
        # Compute Cerenkov integral table
        gpu_integral = compute_cerenkov_integral(gpu_rindex, gpu_wavelengths)
        
        new(gpu_rindex, gpu_absorption, gpu_wavelengths, gpu_integral)
    end
end

"""
    compute_cerenkov_integral(rindex, wavelengths)

Compute the integrated Cerenkov spectrum for efficient wavelength sampling.
"""
function compute_cerenkov_integral(rindex::CuArray{Float32,2}, wavelengths::CuArray{Float32,1})
    num_materials, num_wavelengths = size(rindex)
    integral = CUDA.zeros(Float32, num_materials, num_wavelengths)
    
    # Use CUDA kernel to compute integrals
    threads = (16, 16)
    blocks = (cld(num_materials, threads[1]), cld(num_wavelengths, threads[2]))
    
    @cuda threads=threads blocks=blocks cerenkov_integral_kernel!(integral, rindex, wavelengths)
    CUDA.synchronize()
    
    return integral
end

"""
    cerenkov_integral_kernel!(integral, rindex, wavelengths)

CUDA kernel to compute Cerenkov integral table.
"""
function cerenkov_integral_kernel!(integral::CuDeviceArray{Float32,2},
                                   rindex::CuDeviceArray{Float32,2},
                                   wavelengths::CuDeviceArray{Float32,1})
    mat_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    wl_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if mat_idx <= size(integral, 1) && wl_idx <= size(integral, 2)
        if wl_idx == 1
            integral[mat_idx, wl_idx] = 0.0f0
        else
            # Cerenkov spectrum ∝ 1/λ² * sin²(θ_c)
            # where sin²(θ_c) = 1 - 1/(n*β)²
            
            wl = wavelengths[wl_idx]
            n = rindex[mat_idx, wl_idx]
            
            # For now assume β ≈ 1 (ultrarelativistic)
            # sin²(θ_c) = 1 - 1/n² (for β = 1)
            sin2_theta_c = max(0.0f0, 1.0f0 - 1.0f0/(n*n))
            
            # Cerenkov yield ∝ sin²(θ_c) / λ²
            spectrum_value = sin2_theta_c / (wl * wl)
            
            # Cumulative integral (trapezoidal rule)
            prev_wl = wavelengths[wl_idx - 1]
            prev_n = rindex[mat_idx, wl_idx - 1]
            prev_sin2 = max(0.0f0, 1.0f0 - 1.0f0/(prev_n*prev_n))
            prev_spectrum = prev_sin2 / (prev_wl * prev_wl)
            
            dwl = wl - prev_wl
            integral[mat_idx, wl_idx] = integral[mat_idx, wl_idx - 1] + 
                                        0.5f0 * (spectrum_value + prev_spectrum) * dwl
        end
    end
    return nothing
end

"""
    sample_cerenkov_wavelength(rng::OpticalRNG, cerenkov::QCerenkov, material::Int32, 
                               beta_inverse::Float32)

Sample wavelength from Cerenkov spectrum using inverse transform sampling.
"""
function sample_cerenkov_wavelength(rng::OpticalRNG, cerenkov::QCerenkov, 
                                   material::Int32, beta_inverse::Float32)::Float32
    # Get random number for sampling
    xi = uniform(rng)
    
    # Get maximum integral value for normalization
    max_integral = cerenkov.cerenkov_integral[material, end]
    target = xi * max_integral
    
    # Binary search for wavelength
    low = 1
    high = length(cerenkov.wavelengths)
    
    while high - low > 1
        mid = (low + high) ÷ 2
        if cerenkov.cerenkov_integral[material, mid] < target
            low = mid
        else
            high = mid
        end
    end
    
    # Linear interpolation between low and high
    if high <= length(cerenkov.wavelengths) && low >= 1
        frac = (target - cerenkov.cerenkov_integral[material, low]) / 
               (cerenkov.cerenkov_integral[material, high] - cerenkov.cerenkov_integral[material, low])
        return cerenkov.wavelengths[low] + frac * (cerenkov.wavelengths[high] - cerenkov.wavelengths[low])
    else
        return cerenkov.wavelengths[low]
    end
end

"""
    cerenkov_angle(n::Float32, beta_inverse::Float32)

Calculate Cerenkov angle cos(θ) = 1/(n*β).
"""
function cerenkov_angle(n::Float32, beta_inverse::Float32)::Float32
    cos_theta = beta_inverse / n
    return min(cos_theta, 1.0f0)  # Ensure physical values
end

"""
    generate_cerenkov_photon!(photon::SPhoton, rng::OpticalRNG, genstep::CerenkovGenstep,
                             cerenkov::QCerenkov, photon_idx::Int32, genstep_id::Int32)

Generate a single Cerenkov photon from a generation step.
"""
function generate_cerenkov_photon!(photon::SPhoton, rng::OpticalRNG, 
                                  genstep::CerenkovGenstep, cerenkov::QCerenkov,
                                  photon_idx::Int32, genstep_id::Int32)
    # Sample wavelength from Cerenkov spectrum
    wavelength = sample_cerenkov_wavelength(rng, cerenkov, Int32(genstep.matline), genstep.beta_inverse)
    
    # Get refractive index at this wavelength
    wl_idx = searchsortedfirst(cerenkov.wavelengths, wavelength)
    wl_idx = clamp(wl_idx, 1, length(cerenkov.wavelengths))
    n = cerenkov.rindex[genstep.matline, wl_idx]
    
    # Calculate Cerenkov angle
    cos_theta_c = cerenkov_angle(n, genstep.beta_inverse)
    sin_theta_c = sqrt(max(0.0f0, 1.0f0 - cos_theta_c * cos_theta_c))
    
    # Sample position along step
    xi = uniform(rng)
    pos = genstep.pos + xi * genstep.delta_position
    time = genstep.time + xi * genstep.step_length / (3e8 / genstep.beta_inverse)  # c/β
    
    # Direction of parent particle (normalized)
    particle_dir = normalize(genstep.delta_position)
    
    # Sample azimuthal angle φ uniformly
    phi = 2.0f0 * π * uniform(rng)
    cos_phi = cos(phi)
    sin_phi = sin(phi)
    
    # Build coordinate system with particle direction as z-axis
    # Find perpendicular vectors
    abs_z = abs(particle_dir[3])
    if abs_z < 0.9f0
        perp1 = normalize(cross(particle_dir, SVector{3,Float32}(0, 0, 1)))
    else
        perp1 = normalize(cross(particle_dir, SVector{3,Float32}(1, 0, 0)))
    end
    perp2 = cross(particle_dir, perp1)
    
    # Photon direction in Cerenkov cone
    photon_dir = sin_theta_c * cos_phi * perp1 + 
                 sin_theta_c * sin_phi * perp2 + 
                 cos_theta_c * particle_dir
    
    # Polarization perpendicular to photon direction (random azimuth)
    pol_phi = 2.0f0 * π * uniform(rng)
    # Use cross product to get perpendicular vector
    if abs(photon_dir[3]) < 0.9f0
        pol_perp = normalize(cross(photon_dir, SVector{3,Float32}(0, 0, 1)))
    else
        pol_perp = normalize(cross(photon_dir, SVector{3,Float32}(1, 0, 0)))
    end
    pol_perp2 = cross(photon_dir, pol_perp)
    polarization = cos(pol_phi) * pol_perp + sin(pol_phi) * pol_perp2
    
    # Set photon properties
    photon.pos = pos
    photon.time = time
    photon.mom = photon_dir
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
    cerenkov_generation_kernel!(photons, rngs, gensteps, cerenkov, photon_offset)

CUDA kernel for generating Cerenkov photons.
"""
function cerenkov_generation_kernel!(photons::CuDeviceArray{SPhoton,1},
                                    rngs::CuDeviceArray{OpticalRNG,1},
                                    gensteps::CuDeviceArray{CerenkovGenstep,1},
                                    cerenkov::QCerenkov,
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
            generate_cerenkov_photon!(photons[photon_idx], rngs[photon_idx], 
                                     gensteps[genstep_id], cerenkov,
                                     Int32(photon_idx + photon_offset), Int32(genstep_id))
        end
    end
    return nothing
end

"""
    generate_cerenkov_photons!(photons::CuArray{SPhoton,1}, rngs::CuArray{OpticalRNG,1},
                              gensteps::CuArray{CerenkovGenstep,1}, cerenkov::QCerenkov,
                              photon_offset::Int32 = 0)

Generate Cerenkov photons from generation steps on GPU.
"""
function generate_cerenkov_photons!(photons::CuArray{SPhoton,1}, rngs::CuArray{OpticalRNG,1},
                                   gensteps::CuArray{CerenkovGenstep,1}, cerenkov::QCerenkov,
                                   photon_offset::Int32 = 0)
    if length(photons) == 0
        return
    end
    
    threads = 256
    blocks = cld(length(photons), threads)
    
    @cuda threads=threads blocks=blocks cerenkov_generation_kernel!(photons, rngs, gensteps, cerenkov, photon_offset)
    CUDA.synchronize()
end