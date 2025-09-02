"""
Photomultiplier tube (PMT) simulation implementation.

Julia equivalent of QPMT.cu and qpmt.h for simulating
photon detection and response in PMTs.
"""

using CUDA
using LinearAlgebra

"""
    PMTProperties

Properties for a photomultiplier tube including quantum efficiency,
timing response, and geometric parameters.
"""
struct PMTProperties
    name::String
    pmt_type::Int32                  # PMT type identifier
    
    # Quantum efficiency vs wavelength
    quantum_efficiency::Vector{Float32}
    wavelengths::Vector{Float32}
    
    # Timing response parameters
    transit_time::Float32            # Mean transit time (ns)
    transit_time_spread::Float32     # Transit time spread (ns)
    rise_time::Float32               # Rise time (ns)
    fall_time::Float32               # Fall time (ns)
    
    # Geometric parameters
    cathode_radius::Float32          # Photocathode radius (mm)
    cathode_thickness::Float32       # Photocathode thickness (mm)
    
    # Additional response parameters
    dark_rate::Float32               # Dark count rate (Hz)
    afterpulsing_prob::Float32       # Afterpulsing probability
    
    function PMTProperties(name::String, pmt_type::Int32, wavelengths::Vector{Float32})
        n_wl = length(wavelengths)
        new(name, pmt_type,
            zeros(Float32, n_wl),    # Default zero QE
            wavelengths,
            10.0f0, 2.0f0, 1.0f0, 5.0f0,  # Default timing
            50.0f0, 3.0f0,                 # Default geometry
            100.0f0, 0.01f0)               # Default noise parameters
    end
end

"""
    PMTHit

Information about a photon hit on a PMT.
"""
mutable struct PMTHit
    pmt_id::UInt32                   # PMT identifier
    photon_id::UInt32               # Original photon ID
    wavelength::Float32             # Photon wavelength
    hit_time::Float32               # Hit time
    hit_position::SVector{3,Float32} # Hit position on cathode
    
    # Detection response
    detected::Bool                   # Whether photon was detected
    detection_time::Float32          # Actual detection time (including transit time)
    pulse_amplitude::Float32         # Pulse amplitude
    
    function PMTHit()
        new(0, 0, 0.0f0, 0.0f0, SVector{3,Float32}(0,0,0),
            false, 0.0f0, 0.0f0)
    end
end

"""
    QPMT

GPU-based PMT simulation system.
"""
struct QPMT
    # PMT quantum efficiency data on GPU
    quantum_efficiency::CuArray{Float32,2}  # [pmt_type, wavelength]
    wavelengths::CuArray{Float32,1}
    
    # PMT timing parameters on GPU  
    transit_times::CuArray{Float32,1}        # [pmt_type]
    transit_spreads::CuArray{Float32,1}
    rise_times::CuArray{Float32,1}
    fall_times::CuArray{Float32,1}
    
    # PMT geometry on GPU
    cathode_radii::CuArray{Float32,1}        # [pmt_type]
    cathode_thicknesses::CuArray{Float32,1}
    
    # PMT noise parameters
    dark_rates::CuArray{Float32,1}           # [pmt_type]
    afterpulsing_probs::CuArray{Float32,1}
    
    function QPMT(pmt_properties::Vector{PMTProperties})
        if length(pmt_properties) == 0
            error("Must provide at least one PMT type")
        end
        
        # Assume all PMTs use same wavelength grid
        wavelengths = pmt_properties[1].wavelengths
        n_wl = length(wavelengths)
        n_pmt_types = length(pmt_properties)
        
        # Build property matrices
        qe_matrix = zeros(Float32, n_pmt_types, n_wl)
        transit_times = zeros(Float32, n_pmt_types)
        transit_spreads = zeros(Float32, n_pmt_types)
        rise_times = zeros(Float32, n_pmt_types)
        fall_times = zeros(Float32, n_pmt_types)
        cathode_radii = zeros(Float32, n_pmt_types)
        cathode_thicknesses = zeros(Float32, n_pmt_types)
        dark_rates = zeros(Float32, n_pmt_types)
        afterpulsing_probs = zeros(Float32, n_pmt_types)
        
        for (i, pmt) in enumerate(pmt_properties)
            qe_matrix[i, :] = pmt.quantum_efficiency
            transit_times[i] = pmt.transit_time
            transit_spreads[i] = pmt.transit_time_spread
            rise_times[i] = pmt.rise_time
            fall_times[i] = pmt.fall_time
            cathode_radii[i] = pmt.cathode_radius
            cathode_thicknesses[i] = pmt.cathode_thickness
            dark_rates[i] = pmt.dark_rate
            afterpulsing_probs[i] = pmt.afterpulsing_prob
        end
        
        # Upload to GPU
        new(CuArray(qe_matrix), CuArray(wavelengths),
            CuArray(transit_times), CuArray(transit_spreads),
            CuArray(rise_times), CuArray(fall_times),
            CuArray(cathode_radii), CuArray(cathode_thicknesses),
            CuArray(dark_rates), CuArray(afterpulsing_probs))
    end
end

"""
    get_quantum_efficiency(pmt::QPMT, pmt_type::Int32, wavelength::Float32)

Get quantum efficiency for PMT type at given wavelength.
"""
function get_quantum_efficiency(pmt::QPMT, pmt_type::Int32, wavelength::Float32)::Float32
    if pmt_type < 1 || pmt_type > size(pmt.quantum_efficiency, 1)
        return 0.0f0
    end
    
    return interpolate_property(pmt.quantum_efficiency, pmt.wavelengths, pmt_type, wavelength)
end

"""
    sample_transit_time(rng::OpticalRNG, pmt::QPMT, pmt_type::Int32)

Sample transit time for PMT response using Gaussian distribution.
"""
function sample_transit_time(rng::OpticalRNG, pmt::QPMT, pmt_type::Int32)::Float32
    if pmt_type < 1 || pmt_type > length(pmt.transit_times)
        return 0.0f0
    end
    
    mean_time = pmt.transit_times[pmt_type]
    spread = pmt.transit_spreads[pmt_type]
    
    return gauss_shoot(rng, mean_time, spread)
end

"""
    check_geometric_acceptance(hit_pos::SVector{3,Float32}, pmt::QPMT, pmt_type::Int32)

Check if photon hit position is within PMT cathode acceptance.
"""
function check_geometric_acceptance(hit_pos::SVector{3,Float32}, pmt::QPMT, pmt_type::Int32)::Bool
    if pmt_type < 1 || pmt_type > length(pmt.cathode_radii)
        return false
    end
    
    # Simple cylindrical acceptance check
    radius_sq = hit_pos[1]^2 + hit_pos[2]^2
    cathode_radius = pmt.cathode_radii[pmt_type]
    
    return radius_sq <= cathode_radius^2
end

"""
    simulate_pmt_response!(hit::PMTHit, photon::SPhoton, rng::OpticalRNG, 
                          pmt::QPMT, pmt_type::Int32, pmt_id::UInt32)

Simulate PMT response to a photon hit.
"""
function simulate_pmt_response!(hit::PMTHit, photon::SPhoton, rng::OpticalRNG,
                               pmt::QPMT, pmt_type::Int32, pmt_id::UInt32)
    # Set basic hit information
    hit.pmt_id = pmt_id
    hit.photon_id = photon_idx(photon)
    hit.wavelength = photon.wavelength
    hit.hit_time = photon.time
    hit.hit_position = photon.pos
    
    # Check geometric acceptance
    if !check_geometric_acceptance(photon.pos, pmt, pmt_type)
        hit.detected = false
        return
    end
    
    # Get quantum efficiency at photon wavelength
    qe = get_quantum_efficiency(pmt, pmt_type, photon.wavelength)
    
    # Sample detection probability
    xi_detect = uniform(rng)
    if xi_detect > qe
        hit.detected = false
        return
    end
    
    # Photon detected - calculate response
    hit.detected = true
    
    # Sample transit time
    transit_time = sample_transit_time(rng, pmt, pmt_type)
    hit.detection_time = hit.hit_time + transit_time
    
    # Simple amplitude model (could be more sophisticated)
    # Include collection efficiency variation, gain fluctuations, etc.
    base_amplitude = 1.0f0
    
    # Add gain fluctuations (simplified Gaussian)
    gain_fluctuation = gauss_shoot(rng, 1.0f0, 0.1f0)  # 10% gain variation
    hit.pulse_amplitude = base_amplitude * max(0.1f0, gain_fluctuation)
end

"""
    generate_dark_hits!(dark_hits::CuArray{PMTHit,1}, rngs::CuArray{OpticalRNG,1},
                       pmt::QPMT, pmt_types::CuArray{Int32,1}, pmt_ids::CuArray{UInt32,1},
                       time_window::Float32)

Generate dark noise hits for PMTs during time window.
"""
function generate_dark_hits_kernel!(dark_hits::CuDeviceArray{PMTHit,1},
                                   rngs::CuDeviceArray{OpticalRNG,1},
                                   pmt::QPMT, pmt_types::CuDeviceArray{Int32,1},
                                   pmt_ids::CuDeviceArray{UInt32,1},
                                   time_window::Float32)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(dark_hits)
        pmt_type = pmt_types[idx]
        pmt_id = pmt_ids[idx]
        
        if pmt_type >= 1 && pmt_type <= length(pmt.dark_rates)
            dark_rate = pmt.dark_rates[pmt_type]
            expected_hits = dark_rate * time_window * 1e-9f0  # Convert ns to s
            
            # Sample number of dark hits (Poisson approximation)
            xi = uniform(rngs[idx])
            if xi < expected_hits  # Simplified for low rates
                # Generate a dark hit
                hit = dark_hits[idx]
                hit.pmt_id = pmt_id
                hit.photon_id = 0x00000000  # Mark as dark hit
                hit.wavelength = 400.0f0   # Arbitrary wavelength for dark hits
                hit.hit_time = uniform(rngs[idx]) * time_window
                hit.hit_position = SVector{3,Float32}(0, 0, 0)
                hit.detected = true
                hit.detection_time = hit.hit_time + sample_transit_time(rngs[idx], pmt, pmt_type)
                hit.pulse_amplitude = gauss_shoot(rngs[idx], 1.0f0, 0.1f0)
            else
                # No dark hit
                dark_hits[idx].detected = false
            end
        else
            dark_hits[idx].detected = false
        end
    end
    return nothing
end

"""
    pmt_detection_kernel!(hits, photons, rngs, pmt, pmt_types, pmt_ids)

CUDA kernel for PMT detection simulation.
"""
function pmt_detection_kernel!(hits::CuDeviceArray{PMTHit,1},
                              photons::CuDeviceArray{SPhoton,1},
                              rngs::CuDeviceArray{OpticalRNG,1},
                              pmt::QPMT,
                              pmt_types::CuDeviceArray{Int32,1},
                              pmt_ids::CuDeviceArray{UInt32,1})
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(hits) && idx <= length(photons)
        pmt_type = pmt_types[idx]
        pmt_id = pmt_ids[idx]
        
        if pmt_type >= 1 && pmt_type <= size(pmt.quantum_efficiency, 1)
            simulate_pmt_response!(hits[idx], photons[idx], rngs[idx], 
                                  pmt, pmt_type, pmt_id)
        else
            hits[idx].detected = false
        end
    end
    return nothing
end

"""
    simulate_pmt_detection!(hits::CuArray{PMTHit,1}, photons::CuArray{SPhoton,1},
                           rngs::CuArray{OpticalRNG,1}, pmt::QPMT,
                           pmt_types::CuArray{Int32,1}, pmt_ids::CuArray{UInt32,1})

Simulate PMT detection for array of photons on GPU.
"""
function simulate_pmt_detection!(hits::CuArray{PMTHit,1}, photons::CuArray{SPhoton,1},
                                 rngs::CuArray{OpticalRNG,1}, pmt::QPMT,
                                 pmt_types::CuArray{Int32,1}, pmt_ids::CuArray{UInt32,1})
    if length(hits) == 0 || length(photons) == 0
        return
    end
    
    threads = 256
    blocks = cld(min(length(hits), length(photons)), threads)
    
    @cuda threads=threads blocks=blocks pmt_detection_kernel!(hits, photons, rngs, pmt, pmt_types, pmt_ids)
    CUDA.synchronize()
end

"""
    filter_detected_hits(hits::CuArray{PMTHit,1})

Filter array to only include detected hits.
"""
function filter_detected_hits(hits::CuArray{PMTHit,1})
    # Copy to CPU for filtering (could be done more efficiently on GPU)
    cpu_hits = Array(hits)
    detected_hits = filter(h -> h.detected, cpu_hits)
    return CuArray(detected_hits)
end

"""
    calculate_pmt_efficiency(detected_hits::Vector{PMTHit}, total_photons::Int)

Calculate overall PMT detection efficiency.
"""
function calculate_pmt_efficiency(detected_hits::Vector{PMTHit}, total_photons::Int)
    if total_photons == 0
        return 0.0f0
    end
    
    num_detected = count(h -> h.detected, detected_hits)
    return Float32(num_detected) / Float32(total_photons)
end