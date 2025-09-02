"""
Photon utility functions and basic operations.

Julia equivalent of photon manipulation functions from sphoton.h
"""

using CUDA
using Random

"""
    zero_photon!(p::SPhoton)

Initialize photon to zero state.
"""
function zero_photon!(p::SPhoton)
    p.pos = SVector{3,Float32}(0, 0, 0)
    p.time = 0.0f0
    p.mom = SVector{3,Float32}(0, 0, 0)
    p.iindex = 0x00000000
    p.pol = SVector{3,Float32}(0, 0, 0)
    p.wavelength = 0.0f0
    p.boundary_flag = 0x00000000
    p.identity = 0x00000000
    p.orient_idx = 0x00000000
    p.flagmask = 0x00000000
    return p
end

"""
    set_photon_position!(p::SPhoton, x::Float32, y::Float32, z::Float32, t::Float32)

Set photon position and time.
"""
function set_photon_position!(p::SPhoton, x::Float32, y::Float32, z::Float32, t::Float32)
    p.pos = SVector{3,Float32}(x, y, z)
    p.time = t
end

"""
    set_photon_momentum!(p::SPhoton, px::Float32, py::Float32, pz::Float32)

Set photon momentum direction (should be normalized).
"""
function set_photon_momentum!(p::SPhoton, px::Float32, py::Float32, pz::Float32)
    p.mom = SVector{3,Float32}(px, py, pz)
end

"""
    set_photon_polarization!(p::SPhoton, polx::Float32, poly::Float32, polz::Float32)

Set photon polarization vector.
"""
function set_photon_polarization!(p::SPhoton, polx::Float32, poly::Float32, polz::Float32)
    p.pol = SVector{3,Float32}(polx, poly, polz)
end

"""
    set_photon_wavelength!(p::SPhoton, wl::Float32)

Set photon wavelength in nanometers.
"""
function set_photon_wavelength!(p::SPhoton, wl::Float32)
    p.wavelength = wl
end

"""
    normalize_momentum!(p::SPhoton)

Normalize the photon momentum vector to unit length.
"""
function normalize_momentum!(p::SPhoton)
    norm = sqrt(p.mom[1]^2 + p.mom[2]^2 + p.mom[3]^2)
    if norm > 0.0f0
        p.mom = p.mom / norm
    end
end

"""
    normalize_polarization!(p::SPhoton)

Normalize the photon polarization vector and ensure it's perpendicular to momentum.
"""
function normalize_polarization!(p::SPhoton)
    # Remove component parallel to momentum
    dot_prod = p.pol[1]*p.mom[1] + p.pol[2]*p.mom[2] + p.pol[3]*p.mom[3]
    p.pol = p.pol - dot_prod * p.mom
    
    # Normalize
    norm = sqrt(p.pol[1]^2 + p.pol[2]^2 + p.pol[3]^2)
    if norm > 0.0f0
        p.pol = p.pol / norm
    end
end

"""
    copy_photon(p::SPhoton)

Create a copy of a photon.
"""
function copy_photon(p::SPhoton)
    return SPhoton(
        p.pos, p.time, p.mom, p.iindex, p.pol, p.wavelength,
        p.boundary_flag, p.identity, p.orient_idx, p.flagmask
    )
end

# CUDA kernel for photon operations
"""
    zero_photons_kernel!(photons)

CUDA kernel to initialize an array of photons to zero state.
"""
function zero_photons_kernel!(photons::CuDeviceArray{SPhoton,1})
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(photons)
        zero_photon!(photons[idx])
    end
    return nothing
end

"""
    zero_photons!(photons::CuArray{SPhoton,1})

GPU function to zero an array of photons.
"""
function zero_photons!(photons::CuArray{SPhoton,1})
    if length(photons) == 0
        return
    end
    
    threads = 256
    blocks = cld(length(photons), threads)
    
    @cuda threads=threads blocks=blocks zero_photons_kernel!(photons)
    CUDA.synchronize()
end

"""
    photon_energy(wl::Float32)

Convert wavelength in nm to photon energy in eV.
hc = 1239.84198 eV⋅nm
"""
function photon_energy(wl::Float32)::Float32
    return 1239.84198f0 / wl
end

"""
    photon_wavelength(energy::Float32)

Convert photon energy in eV to wavelength in nm.
"""
function photon_wavelength(energy::Float32)::Float32
    return 1239.84198f0 / energy
end