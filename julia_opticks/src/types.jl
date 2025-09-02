"""
Core types for Opticks Julia implementation.

Converts C++ structs from the original Opticks to Julia equivalents.
"""

using CUDA
using StaticArrays

# Equivalent to sphoton struct from sysrap/sphoton.h
"""
    SPhoton

Represents a single optical photon with position, momentum, polarization,
timing, and physics state information.

# Fields
- `pos::SVector{3,Float32}`: Position (x,y,z)
- `time::Float32`: Time
- `mom::SVector{3,Float32}`: Momentum direction (normalized)
- `iindex::UInt32`: Instance index of intersected geometry
- `pol::SVector{3,Float32}`: Polarization vector
- `wavelength::Float32`: Wavelength in nm
- `boundary_flag::UInt32`: Combined boundary (upper 16) and flag (lower 16) bits
- `identity::UInt32`: Sensor identifier + 1
- `orient_idx::UInt32`: Orientation bit (MSB) + photon index (lower 31 bits)
- `flagmask::UInt32`: Accumulated history flags
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
            SVector{3,Float32}(0, 0, 0), 0.0f0,    # pos, time
            SVector{3,Float32}(0, 0, 0), 0x00000000,  # mom, iindex
            SVector{3,Float32}(0, 0, 0), 0.0f0,    # pol, wavelength
            0x00000000, 0x00000000, 0x00000000, 0x00000000  # boundary_flag, identity, orient_idx, flagmask
        )
    end
end

# Accessor methods for packed fields
"""Get photon index (lower 31 bits of orient_idx)"""
photon_idx(p::SPhoton) = p.orient_idx & 0x7fffffff

"""Get orientation (-1.0 or +1.0 based on MSB of orient_idx)"""
orientation(p::SPhoton) = (p.orient_idx & 0x80000000) != 0 ? -1.0f0 : 1.0f0

"""Get boundary index (upper 16 bits of boundary_flag)"""
boundary(p::SPhoton) = (p.boundary_flag >> 16) & 0xffff

"""Get history flag (lower 16 bits of boundary_flag)"""
flag(p::SPhoton) = p.boundary_flag & 0xffff

"""Set photon index while preserving orientation bit"""
function set_photon_idx!(p::SPhoton, idx::UInt32)
    p.orient_idx = (p.orient_idx & 0x80000000) | (idx & 0x7fffffff)
end

"""Set orientation while preserving photon index"""
function set_orientation!(p::SPhoton, orient::Float32)
    orient_bit = orient < 0.0f0 ? 0x80000000 : 0x00000000
    p.orient_idx = (p.orient_idx & 0x7fffffff) | orient_bit
end

"""Set boundary index while preserving flags"""
function set_boundary!(p::SPhoton, boundary::UInt32)
    p.boundary_flag = (p.boundary_flag & 0x0000ffff) | ((boundary & 0xffff) << 16)
end

"""Set history flag while preserving boundary, also add to flagmask"""
function set_flag!(p::SPhoton, flag_val::UInt32)
    p.boundary_flag = (p.boundary_flag & 0xffff0000) | (flag_val & 0xffff)
    p.flagmask |= flag_val
end

# Optical properties structure
"""
    OpticalProperties

Container for optical material properties including refractive index,
absorption, scattering coefficients, etc.
"""
struct OpticalProperties
    rindex::Vector{Float32}      # Refractive index vs wavelength
    absorption::Vector{Float32}   # Absorption length vs wavelength  
    scattering::Vector{Float32}  # Scattering length vs wavelength
    reemission::Vector{Float32}  # Re-emission probability vs wavelength
    wavelengths::Vector{Float32} # Wavelength grid
end

# Boundary properties between materials
"""
    BoundaryProperties
    
Properties at interface between two optical materials including
transmission, reflection, surface properties.
"""
struct BoundaryProperties
    material1::Int32
    material2::Int32
    surface1::Int32
    surface2::Int32
    transmittance::Vector{Float32}
    reflectance::Vector{Float32}
    efficiency::Vector{Float32}
    wavelengths::Vector{Float32}
end

# Event structure to hold generated photons and results
"""
    OpticalEvent
    
Container for photon simulation event including input gensteps
and output photon hits.
"""
mutable struct OpticalEvent
    gensteps::CuArray{Float32,2}    # Generation steps [num_steps, 6]
    photons::CuArray{SPhoton,1}     # Current photon states
    hits::CuArray{SPhoton,1}        # Final photon hits
    seeds::CuArray{UInt64,1}        # Random number seeds
    num_photons::Int32
    num_hits::Int32
    
    function OpticalEvent(max_photons::Int)
        new(
            CuArray{Float32,2}(undef, 0, 6),
            CuArray{SPhoton,1}(undef, max_photons),
            CuArray{SPhoton,1}(undef, max_photons),
            CuArray{UInt64,1}(undef, max_photons),
            0, 0
        )
    end
end