"""
Main simulation interface for Opticks.jl

Julia equivalent of QSim.cu and QSim.hh, providing the main interface
for optical photon simulation combining generation, propagation, and detection.
"""

using CUDA
using LinearAlgebra

"""
    SimulationContext

Context for a photon simulation including all physics components.
"""
mutable struct SimulationContext
    # Physics components
    optical::QOptical              # Optical properties and boundaries
    cerenkov::QCerenkov           # Cerenkov generation
    scint::QScint                 # Scintillation generation  
    pmt::QPMT                     # PMT detection
    
    # Simulation parameters
    max_bounces::Int32            # Maximum bounces per photon
    max_steps::Int32              # Maximum propagation steps
    time_window::Float32          # Simulation time window (ns)
    wavelength_min::Float32       # Minimum wavelength (nm)
    wavelength_max::Float32       # Maximum wavelength (nm)
    
    # Random number generation
    event_idx::UInt32             # Current event index
    
    function SimulationContext(optical::QOptical, cerenkov::QCerenkov, 
                              scint::QScint, pmt::QPMT)
        new(optical, cerenkov, scint, pmt,
            1000, 10000, 1000.0f0, 200.0f0, 800.0f0,  # Default limits
            0x00000000)  # Start with event 0
    end
end

"""
    PropagationState

Working state for a single photon during propagation.
"""
mutable struct PropagationState
    position::SVector{3,Float32}
    direction::SVector{3,Float32}
    polarization::SVector{3,Float32}
    wavelength::Float32
    time::Float32
    
    current_material::Int32
    distance_to_boundary::Float32
    absorption_length::Float32
    scattering_length::Float32
    
    bounces::Int32
    steps::Int32
    status::Int32  # 0=propagating, 1=detected, 2=absorbed, 3=escaped
    
    function PropagationState()
        new(SVector{3,Float32}(0,0,0), SVector{3,Float32}(0,0,1), SVector{3,Float32}(1,0,0),
            400.0f0, 0.0f0,
            1, 1e6f0, 1e6f0, 1e6f0,
            0, 0, 0)
    end
end

"""
    init_propagation_state!(state::PropagationState, photon::SPhoton, ctx::SimulationContext)

Initialize propagation state from photon.
"""
function init_propagation_state!(state::PropagationState, photon::SPhoton, ctx::SimulationContext)
    state.position = photon.pos
    state.direction = photon.mom
    state.polarization = photon.pol
    state.wavelength = photon.wavelength
    state.time = photon.time
    
    state.current_material = 1  # Default to first material
    state.distance_to_boundary = 1e6f0
    state.absorption_length = get_absorption_length(ctx.optical, state.current_material, state.wavelength)
    state.scattering_length = get_scattering_length(ctx.optical, state.current_material, state.wavelength)
    
    state.bounces = 0
    state.steps = 0
    state.status = 0  # Propagating
end

"""
    update_photon_from_state!(photon::SPhoton, state::PropagationState)

Update photon from propagation state.
"""
function update_photon_from_state!(photon::SPhoton, state::PropagationState)
    photon.pos = state.position
    photon.mom = state.direction
    photon.pol = state.polarization
    photon.wavelength = state.wavelength
    photon.time = state.time
end

"""
    propagate_step!(state::PropagationState, rng::OpticalRNG, ctx::SimulationContext)

Perform one propagation step for a photon.
"""
function propagate_step!(state::PropagationState, rng::OpticalRNG, ctx::SimulationContext)
    if state.status != 0  # Not propagating
        return
    end
    
    # Sample step length
    xi_abs = uniform(rng)
    xi_scat = uniform(rng)
    
    step_abs = -state.absorption_length * log(max(1e-10f0, xi_abs))
    step_scat = -state.scattering_length * log(max(1e-10f0, xi_scat))
    step_boundary = state.distance_to_boundary
    
    # Find shortest step
    step_length = min(step_abs, step_scat, step_boundary)
    
    # Move photon
    state.position += step_length * state.direction
    state.time += step_length / 2.998e8f0  # c in mm/ns
    state.steps += 1
    
    # Check what happened
    if step_length == step_abs
        # Absorbed
        state.status = 2
    elseif step_length == step_scat
        # Rayleigh scattered
        old_direction = state.direction
        new_direction = Ref(SVector{3,Float32}(0,0,0))
        rayleigh_scatter_direction!(new_direction, old_direction, rng)
        state.direction = new_direction[]
        
        # Update material properties for new step
        state.distance_to_boundary = 1e6f0  # Simplified - should calculate actual distance
        state.absorption_length = get_absorption_length(ctx.optical, state.current_material, state.wavelength)
        state.scattering_length = get_scattering_length(ctx.optical, state.current_material, state.wavelength)
    else
        # Hit boundary
        state.bounces += 1
        
        # Simplified boundary interaction (should use proper geometry)
        boundary_idx = 1  # Simplified
        from_outside = true
        interaction = boundary_interaction!(
            SPhoton(),  # Dummy photon for interface
            rng, ctx.optical, boundary_idx, from_outside
        )
        
        if interaction == 0  # Transmitted
            # Update material
            state.current_material = (state.current_material == 1) ? 2 : 1
            state.absorption_length = get_absorption_length(ctx.optical, state.current_material, state.wavelength)
            state.scattering_length = get_scattering_length(ctx.optical, state.current_material, state.wavelength)
        elseif interaction == 1  # Reflected
            # Direction already updated by boundary_interaction!
            state.direction = SVector{3,Float32}(state.direction[1], state.direction[2], -state.direction[3])
        else  # Absorbed
            state.status = 2
        end
        
        state.distance_to_boundary = 1e6f0  # Reset
    end
    
    # Check termination conditions
    if state.bounces >= ctx.max_bounces || state.steps >= ctx.max_steps
        state.status = 3  # Escaped/terminated
    elseif state.time > ctx.time_window
        state.status = 3  # Time cutoff
    end
end

"""
    propagate_photon!(photon::SPhoton, rng::OpticalRNG, ctx::SimulationContext)

Propagate a single photon through the geometry.
"""
function propagate_photon!(photon::SPhoton, rng::OpticalRNG, ctx::SimulationContext)
    state = PropagationState()
    init_propagation_state!(state, photon, ctx)
    
    while state.status == 0  # Still propagating
        propagate_step!(state, rng, ctx)
    end
    
    # Update photon with final state
    update_photon_from_state!(photon, state)
    
    # Set status flags
    if state.status == 1
        set_flag!(photon, 0x0001)  # Detected
    elseif state.status == 2
        set_flag!(photon, 0x0002)  # Absorbed
    else
        set_flag!(photon, 0x0004)  # Escaped
    end
end

"""
    propagation_kernel!(photons, rngs, ctx)

CUDA kernel for photon propagation.
"""
function propagation_kernel!(photons::CuDeviceArray{SPhoton,1},
                            rngs::CuDeviceArray{OpticalRNG,1},
                            ctx::SimulationContext)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(photons)
        propagate_photon!(photons[idx], rngs[idx], ctx)
    end
    return nothing
end

"""
    propagate_photons!(photons::CuArray{SPhoton,1}, rngs::CuArray{OpticalRNG,1},
                      ctx::SimulationContext)

Propagate array of photons on GPU.
"""
function propagate_photons!(photons::CuArray{SPhoton,1}, rngs::CuArray{OpticalRNG,1},
                           ctx::SimulationContext)
    if length(photons) == 0
        return
    end
    
    threads = 256
    blocks = cld(length(photons), threads)
    
    @cuda threads=threads blocks=blocks propagation_kernel!(photons, rngs, ctx)
    CUDA.synchronize()
end

"""
    simulate_event!(event::OpticalEvent, cerenkov_gensteps::CuArray{CerenkovGenstep,1},
                   scint_gensteps::CuArray{ScintillationGenstep,1}, ctx::SimulationContext)

Simulate a complete optical event including generation, propagation, and detection.
"""
function simulate_event!(event::OpticalEvent, 
                        cerenkov_gensteps::CuArray{CerenkovGenstep,1},
                        scint_gensteps::CuArray{ScintillationGenstep,1}, 
                        ctx::SimulationContext)
    
    # Count total photons to generate
    total_cerenkov = sum(gs -> gs.numphoton, Array(cerenkov_gensteps))
    total_scint = sum(gs -> gs.numphoton, Array(scint_gensteps))
    total_photons = total_cerenkov + total_scint
    
    if total_photons == 0
        return
    end
    
    # Ensure event arrays are large enough
    if length(event.photons) < total_photons
        resize!(event.photons, total_photons)
        resize!(event.seeds, total_photons)
    end
    
    # Initialize random number generators
    init_rng_states!(view(event.seeds, 1:total_photons), UInt32(0), ctx.event_idx)
    rngs = reinterpret(OpticalRNG, event.seeds)
    
    # Generate Cerenkov photons
    if total_cerenkov > 0
        cerenkov_photons = view(event.photons, 1:total_cerenkov)
        cerenkov_rngs = view(rngs, 1:total_cerenkov)
        generate_cerenkov_photons!(cerenkov_photons, cerenkov_rngs, cerenkov_gensteps, ctx.cerenkov)
    end
    
    # Generate scintillation photons
    if total_scint > 0
        scint_photons = view(event.photons, (total_cerenkov+1):(total_cerenkov+total_scint))
        scint_rngs = view(rngs, (total_cerenkov+1):(total_cerenkov+total_scint))
        generate_scintillation_photons!(scint_photons, scint_rngs, scint_gensteps, ctx.scint, Int32(total_cerenkov))
    end
    
    # Propagate all photons
    active_photons = view(event.photons, 1:total_photons)
    active_rngs = view(rngs, 1:total_photons)
    propagate_photons!(active_photons, active_rngs, ctx)
    
    event.num_photons = Int32(total_photons)
    ctx.event_idx += 1  # Increment for next event
end

"""
    simulate_optical_photons(materials::Vector{MaterialProperties},
                           surfaces::Vector{SurfaceProperties},
                           boundaries::Vector{BoundaryProperties},
                           pmt_properties::Vector{PMTProperties},
                           cerenkov_props::Vector{ScintillationProperties},
                           scint_props::Vector{ScintillationProperties})

Main entry point for optical photon simulation setup.
"""
function simulate_optical_photons(materials::Vector{MaterialProperties},
                                 surfaces::Vector{SurfaceProperties},
                                 boundaries::Vector{BoundaryProperties},
                                 pmt_properties::Vector{PMTProperties},
                                 cerenkov_props::Vector{MaterialProperties},
                                 scint_props::Vector{ScintillationProperties})
    
    # Create physics components
    optical = QOptical(materials, surfaces, boundaries)
    
    # For Cerenkov, we need material properties
    rindex_matrix = zeros(Float32, length(materials), length(materials[1].wavelengths))
    absorption_matrix = zeros(Float32, length(materials), length(materials[1].wavelengths))
    for (i, mat) in enumerate(materials)
        rindex_matrix[i, :] = mat.rindex
        absorption_matrix[i, :] = mat.absorption
    end
    cerenkov = QCerenkov(rindex_matrix, absorption_matrix, materials[1].wavelengths)
    
    scint = QScint(scint_props)
    pmt = QPMT(pmt_properties)
    
    # Create simulation context
    ctx = SimulationContext(optical, cerenkov, scint, pmt)
    
    return ctx
end

"""
    run_simulation(ctx::SimulationContext, cerenkov_gensteps::Vector{CerenkovGenstep},
                  scint_gensteps::Vector{ScintillationGenstep}, max_photons::Int = 1000000)

Run a complete optical photon simulation.
"""
function run_simulation(ctx::SimulationContext, 
                       cerenkov_gensteps::Vector{CerenkovGenstep},
                       scint_gensteps::Vector{ScintillationGenstep},
                       max_photons::Int = 1000000)
    
    # Create event
    event = OpticalEvent(max_photons)
    
    # Upload gensteps to GPU
    gpu_cerenkov_gensteps = CuArray(cerenkov_gensteps)
    gpu_scint_gensteps = CuArray(scint_gensteps)
    
    # Run simulation
    simulate_event!(event, gpu_cerenkov_gensteps, gpu_scint_gensteps, ctx)
    
    # Return results (could add hit collection, analysis, etc.)
    return event
end