"""
    Opticks.jl

Julia implementation of Opticks GPU-accelerated optical photon simulation.
This package converts the CUDA-based Opticks library to Julia using CUDA.jl
and replaces Geant4 dependencies with Geant4.jl.
"""
module Opticks

using LinearAlgebra
using Random
using StaticArrays

# Conditional CUDA support
const CUDA_AVAILABLE = try
    using CUDA
    CUDA.functional()
catch
    false
end

if CUDA_AVAILABLE
    using CUDA
    export CuArray
end

# Export main types and functions
export SPhoton, OpticalRNG, uniform
export MaterialProperties, SurfaceProperties, BoundaryProperties
export photon_idx, orientation, set_photon_idx!, set_orientation!
export zero_photon!, set_photon_position!, set_photon_momentum!

# Include core modules
include("types_minimal.jl")
include("photon_minimal.jl") 
include("random_minimal.jl")

end # module