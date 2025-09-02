"""
Core types for Opticks Julia implementation (minimal version without CUDA).
"""

using StaticArrays

"""
    SPhoton

Represents a single optical photon with position, momentum, polarization,
timing, and physics state information.
"""
mutable struct SPhoton
    pos::SVector{3,Float32}
    time::Float32
    mom::SVector{3,Float32}
    iindex::UInt32
    pol::SVector{3,Float32}
    wavelength::Float32
    boundary_flag::UInt32
    identity::UInt32
    orient_idx::UInt32
    flagmask::UInt32
    
    function SPhoton()
        new(
            SVector{3,Float32}(0, 0, 0), 0.0f0,
            SVector{3,Float32}(0, 0, 0), 0x00000000,
            SVector{3,Float32}(0, 0, 0), 0.0f0,
            0x00000000, 0x00000000, 0x00000000, 0x00000000
        )
    end
end

# Accessor methods for packed fields
"""Get photon index (lower 31 bits of orient_idx)"""
photon_idx(p::SPhoton) = p.orient_idx & 0x7fffffff

"""Get orientation (-1.0 or +1.0 based on MSB of orient_idx)"""
orientation(p::SPhoton) = (p.orient_idx & 0x80000000) != 0 ? -1.0f0 : 1.0f0

"""Set photon index while preserving orientation bit"""
function set_photon_idx!(p::SPhoton, idx::UInt32)
    p.orient_idx = (p.orient_idx & 0x80000000) | (idx & 0x7fffffff)
end

"""Set orientation while preserving photon index"""
function set_orientation!(p::SPhoton, orient::Float32)
    orient_bit = orient < 0.0f0 ? 0x80000000 : 0x00000000
    p.orient_idx = (p.orient_idx & 0x7fffffff) | orient_bit
end

"""
    MaterialProperties

Container for optical material properties.
"""
struct MaterialProperties
    name::String
    rindex::Vector{Float32}
    absorption::Vector{Float32}
    scattering::Vector{Float32}
    wavelengths::Vector{Float32}
    
    function MaterialProperties(name::String, wavelengths::Vector{Float32})
        n_wl = length(wavelengths)
        new(name, 
            ones(Float32, n_wl),
            fill(1e6f0, n_wl),
            fill(1e6f0, n_wl),
            wavelengths)
    end
end

"""
    SurfaceProperties

Optical surface properties.
"""
struct SurfaceProperties
    name::String
    transmittance::Vector{Float32}
    reflectance::Vector{Float32}
    wavelengths::Vector{Float32}
    
    function SurfaceProperties(name::String, wavelengths::Vector{Float32})
        n_wl = length(wavelengths)
        new(name,
            zeros(Float32, n_wl),
            ones(Float32, n_wl),
            wavelengths)
    end
end

"""
    BoundaryProperties

Properties at boundary between materials.
"""
struct BoundaryProperties
    material1_idx::Int32
    material2_idx::Int32
    
    function BoundaryProperties(mat1::Int32, mat2::Int32)
        new(mat1, mat2)
    end
end