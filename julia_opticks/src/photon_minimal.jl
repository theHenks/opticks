"""
Photon utility functions (minimal version).
"""

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
    set_photon_position!(p::SPhoton, x, y, z, t)

Set photon position and time.
"""
function set_photon_position!(p::SPhoton, x::Float32, y::Float32, z::Float32, t::Float32)
    p.pos = SVector{3,Float32}(x, y, z)
    p.time = t
end

"""
    set_photon_momentum!(p::SPhoton, px, py, pz)

Set photon momentum direction.
"""
function set_photon_momentum!(p::SPhoton, px::Float32, py::Float32, pz::Float32)
    p.mom = SVector{3,Float32}(px, py, pz)
end