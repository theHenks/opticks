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
export SPhoton, QSim, QCerenkov, QScint, QPMT
export simulate_photons, cerenkov_generation, scintillation_generation
export optical_propagate, boundary_physics

# Include core modules
include("types.jl")
include("photon.jl") 
include("simulation.jl")
include("cerenkov.jl")
include("scintillation.jl")
include("optics.jl")
include("pmt.jl")
include("random.jl")

end # module