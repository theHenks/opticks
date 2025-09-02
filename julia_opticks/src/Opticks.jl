"""
    Opticks.jl

Julia implementation of Opticks GPU-accelerated optical photon simulation.
This package converts the CUDA-based Opticks library to Julia using CUDA.jl
and replaces Geant4 dependencies with Geant4.jl.
"""
module Opticks

using CUDA
using LinearAlgebra
using Random
using StaticArrays

# Export main types and functions
export SPhoton, QOptical, QCerenkov, QScint, QPMT
export OpticalEvent, SimulationContext, CerenkovGenstep, ScintillationGenstep
export MaterialProperties, SurfaceProperties, BoundaryProperties, PMTProperties, ScintillationProperties
export simulate_optical_photons, run_simulation, simulate_event!
export generate_cerenkov_photons!, generate_scintillation_photons!, propagate_photons!

# Include core modules
include("types.jl")
include("photon.jl") 
include("random.jl")
include("cerenkov.jl")
include("scintillation.jl")
include("optics.jl")
include("pmt.jl")
include("simulation.jl")

end # module