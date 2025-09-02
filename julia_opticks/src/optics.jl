"""
Optical properties and boundary physics implementation.

Julia equivalent of QOptical.cu, qprop.h, and qbnd.h for handling
optical material properties and boundary interactions.
"""

using CUDA
using LinearAlgebra

"""
    MaterialProperties

Complete optical properties for a single material.
"""
struct MaterialProperties
    name::String
    rindex::Vector{Float32}          # Refractive index vs wavelength
    absorption::Vector{Float32}      # Absorption length (mm) vs wavelength
    scattering::Vector{Float32}      # Rayleigh scattering length (mm) vs wavelength
    reemission::Vector{Float32}      # Re-emission probability vs wavelength
    wavelengths::Vector{Float32}     # Wavelength grid (nm)
    
    function MaterialProperties(name::String, wavelengths::Vector{Float32})
        n_wl = length(wavelengths)
        new(name, 
            ones(Float32, n_wl),      # Default n=1 (vacuum)
            fill(1e6f0, n_wl),        # Default very long absorption
            fill(1e6f0, n_wl),        # Default very long scattering  
            zeros(Float32, n_wl),     # Default no re-emission
            wavelengths)
    end
end

"""
    SurfaceProperties

Optical surface properties for boundary interactions.
"""
struct SurfaceProperties
    name::String
    model::String                    # "glisur", "unified", etc.
    finish::String                   # "polished", "ground", etc.
    type::String                     # "dielectric_metal", "dielectric_dielectric", etc.
    
    # Surface optical parameters
    transmittance::Vector{Float32}   # Transmittance vs wavelength
    reflectance::Vector{Float32}     # Reflectance vs wavelength  
    efficiency::Vector{Float32}      # Detection efficiency vs wavelength
    wavelengths::Vector{Float32}     # Wavelength grid
    
    # Surface roughness parameters
    sigma_alpha::Float32             # Surface roughness angle (radians)
    polish::Float32                  # Polish parameter [0,1]
    
    function SurfaceProperties(name::String, wavelengths::Vector{Float32})
        n_wl = length(wavelengths)
        new(name, "unified", "polished", "dielectric_dielectric",
            zeros(Float32, n_wl),     # Default transmittance
            ones(Float32, n_wl),      # Default full reflectance
            zeros(Float32, n_wl),     # Default no detection
            wavelengths,
            0.0f0, 1.0f0)            # Default smooth surface
    end
end

"""
    BoundaryProperties

Properties at the boundary between two materials.
"""
struct BoundaryProperties
    material1_idx::Int32             # Index into material array
    material2_idx::Int32  
    surface1_idx::Int32              # Index into surface array (-1 if none)
    surface2_idx::Int32
    
    function BoundaryProperties(mat1::Int32, mat2::Int32, surf1::Int32 = -1, surf2::Int32 = -1)
        new(mat1, mat2, surf1, surf2)
    end
end

"""
    QOptical

GPU-based optical properties manager.
Handles material properties, surface properties, and boundary interactions.
"""
struct QOptical
    # Material properties on GPU
    material_rindex::CuArray{Float32,2}      # [material, wavelength]
    material_absorption::CuArray{Float32,2}  
    material_scattering::CuArray{Float32,2}
    material_reemission::CuArray{Float32,2}
    
    # Surface properties on GPU
    surface_transmittance::CuArray{Float32,2}  # [surface, wavelength]
    surface_reflectance::CuArray{Float32,2}
    surface_efficiency::CuArray{Float32,2}
    surface_sigma_alpha::CuArray{Float32,1}    # [surface]
    surface_polish::CuArray{Float32,1}
    
    # Boundary lookup table
    boundary_materials::CuArray{Int32,2}       # [boundary, 2] -> (mat1, mat2)
    boundary_surfaces::CuArray{Int32,2}        # [boundary, 2] -> (surf1, surf2)
    
    # Wavelength grid
    wavelengths::CuArray{Float32,1}
    
    function QOptical(materials::Vector{MaterialProperties}, 
                     surfaces::Vector{SurfaceProperties},
                     boundaries::Vector{BoundaryProperties})
        
        if length(materials) == 0
            error("Must provide at least one material")
        end
        
        # Assume all use same wavelength grid
        wavelengths = materials[1].wavelengths
        n_wl = length(wavelengths)
        n_mat = length(materials)
        n_surf = length(surfaces)
        n_bnd = length(boundaries)
        
        # Build material property matrices
        mat_rindex = zeros(Float32, n_mat, n_wl)
        mat_absorption = zeros(Float32, n_mat, n_wl)
        mat_scattering = zeros(Float32, n_mat, n_wl)
        mat_reemission = zeros(Float32, n_mat, n_wl)
        
        for (i, mat) in enumerate(materials)
            mat_rindex[i, :] = mat.rindex
            mat_absorption[i, :] = mat.absorption
            mat_scattering[i, :] = mat.scattering
            mat_reemission[i, :] = mat.reemission
        end
        
        # Build surface property matrices
        surf_transmittance = zeros(Float32, max(n_surf, 1), n_wl)
        surf_reflectance = zeros(Float32, max(n_surf, 1), n_wl)
        surf_efficiency = zeros(Float32, max(n_surf, 1), n_wl)
        surf_sigma_alpha = zeros(Float32, max(n_surf, 1))
        surf_polish = ones(Float32, max(n_surf, 1))
        
        for (i, surf) in enumerate(surfaces)
            surf_transmittance[i, :] = surf.transmittance
            surf_reflectance[i, :] = surf.reflectance
            surf_efficiency[i, :] = surf.efficiency
            surf_sigma_alpha[i] = surf.sigma_alpha
            surf_polish[i] = surf.polish
        end
        
        # Build boundary lookup tables
        bnd_materials = zeros(Int32, max(n_bnd, 1), 2)
        bnd_surfaces = fill(Int32(-1), max(n_bnd, 1), 2)
        
        for (i, bnd) in enumerate(boundaries)
            bnd_materials[i, 1] = bnd.material1_idx
            bnd_materials[i, 2] = bnd.material2_idx
            bnd_surfaces[i, 1] = bnd.surface1_idx
            bnd_surfaces[i, 2] = bnd.surface2_idx
        end
        
        # Upload to GPU
        new(CuArray(mat_rindex), CuArray(mat_absorption), CuArray(mat_scattering), CuArray(mat_reemission),
            CuArray(surf_transmittance), CuArray(surf_reflectance), CuArray(surf_efficiency),
            CuArray(surf_sigma_alpha), CuArray(surf_polish),
            CuArray(bnd_materials), CuArray(bnd_surfaces),
            CuArray(wavelengths))
    end
end

"""
    interpolate_property(props::CuDeviceArray{Float32,2}, wavelengths::CuDeviceArray{Float32,1},
                        material_idx::Int32, wavelength::Float32)

Interpolate a material property at given wavelength.
"""
function interpolate_property(props::CuDeviceArray{Float32,2}, 
                             wavelengths::CuDeviceArray{Float32,1},
                             material_idx::Int32, wavelength::Float32)::Float32
    if material_idx < 1 || material_idx > size(props, 1)
        return 0.0f0
    end
    
    # Find wavelength bracket
    n_wl = length(wavelengths)
    if wavelength <= wavelengths[1]
        return props[material_idx, 1]
    elseif wavelength >= wavelengths[n_wl]
        return props[material_idx, n_wl]
    end
    
    # Binary search for bracket
    low = 1
    high = n_wl
    while high - low > 1
        mid = (low + high) ÷ 2
        if wavelengths[mid] < wavelength
            low = mid
        else
            high = mid
        end
    end
    
    # Linear interpolation
    wl_low = wavelengths[low]
    wl_high = wavelengths[high]
    frac = (wavelength - wl_low) / (wl_high - wl_low)
    
    return props[material_idx, low] + frac * (props[material_idx, high] - props[material_idx, low])
end

"""
    get_refractive_index(optical::QOptical, material_idx::Int32, wavelength::Float32)

Get refractive index for material at wavelength.
"""
function get_refractive_index(optical::QOptical, material_idx::Int32, wavelength::Float32)::Float32
    return interpolate_property(optical.material_rindex, optical.wavelengths, material_idx, wavelength)
end

"""
    get_absorption_length(optical::QOptical, material_idx::Int32, wavelength::Float32)

Get absorption length for material at wavelength.
"""
function get_absorption_length(optical::QOptical, material_idx::Int32, wavelength::Float32)::Float32
    return interpolate_property(optical.material_absorption, optical.wavelengths, material_idx, wavelength)
end

"""
    get_scattering_length(optical::QOptical, material_idx::Int32, wavelength::Float32)

Get Rayleigh scattering length for material at wavelength.
"""
function get_scattering_length(optical::QOptical, material_idx::Int32, wavelength::Float32)::Float32
    return interpolate_property(optical.material_scattering, optical.wavelengths, material_idx, wavelength)
end

"""
    fresnel_reflection(n1::Float32, n2::Float32, cos_theta1::Float32)

Calculate Fresnel reflection coefficient at interface.
Returns (Rs, Rp, cos_theta2) for s-polarized, p-polarized light and transmitted angle.
"""
function fresnel_reflection(n1::Float32, n2::Float32, cos_theta1::Float32)
    sin_theta1_sq = 1.0f0 - cos_theta1 * cos_theta1
    sin_theta1 = sqrt(max(0.0f0, sin_theta1_sq))
    
    # Snell's law: n1*sin(θ1) = n2*sin(θ2)
    sin_theta2 = (n1 / n2) * sin_theta1
    
    # Check for total internal reflection
    if sin_theta2 >= 1.0f0
        return (1.0f0, 1.0f0, 0.0f0)  # Total reflection
    end
    
    cos_theta2 = sqrt(1.0f0 - sin_theta2 * sin_theta2)
    
    # Fresnel equations
    rs_num = n1 * cos_theta1 - n2 * cos_theta2
    rs_den = n1 * cos_theta1 + n2 * cos_theta2
    rs = (rs_num / rs_den)^2
    
    rp_num = n1 * cos_theta2 - n2 * cos_theta1  
    rp_den = n1 * cos_theta2 + n2 * cos_theta1
    rp = (rp_num / rp_den)^2
    
    return (rs, rp, cos_theta2)
end

"""
    rayleigh_scatter_direction!(new_direction::Ref{SVector{3,Float32}}, 
                               old_direction::SVector{3,Float32}, rng::OpticalRNG)

Sample new direction for Rayleigh scattering.
Uses the 1 + cos²θ angular distribution.
"""
function rayleigh_scatter_direction!(new_direction::Ref{SVector{3,Float32}},
                                    old_direction::SVector{3,Float32}, rng::OpticalRNG)
    # Sample scattering angle from Rayleigh distribution: (1 + cos²θ)
    # Use rejection sampling
    while true
        xi1 = uniform(rng)
        xi2 = uniform(rng)
        
        cos_theta = 2.0f0 * xi1 - 1.0f0  # cos θ ∈ [-1,1]
        prob = 0.75f0 * (1.0f0 + cos_theta * cos_theta)  # Normalized to max value of 1.5
        
        if xi2 * 1.5f0 <= prob
            sin_theta = sqrt(1.0f0 - cos_theta * cos_theta)
            phi = 2.0f0 * π * uniform(rng)
            
            # Build coordinate system with old direction as z-axis
            if abs(old_direction[3]) < 0.9f0
                perp1 = normalize(cross(old_direction, SVector{3,Float32}(0, 0, 1)))
            else
                perp1 = normalize(cross(old_direction, SVector{3,Float32}(1, 0, 0)))
            end
            perp2 = cross(old_direction, perp1)
            
            # New direction in spherical coordinates
            new_direction[] = sin_theta * cos(phi) * perp1 + 
                             sin_theta * sin(phi) * perp2 + 
                             cos_theta * old_direction
            break
        end
    end
end

"""
    boundary_interaction!(photon::SPhoton, rng::OpticalRNG, optical::QOptical,
                         boundary_idx::Int32, from_outside::Bool)

Handle photon interaction at a boundary between materials.
Updates photon state for transmission, reflection, or absorption.
Returns interaction type: 0=transmitted, 1=reflected, 2=absorbed
"""
function boundary_interaction!(photon::SPhoton, rng::OpticalRNG, optical::QOptical,
                              boundary_idx::Int32, from_outside::Bool)::Int32
    if boundary_idx < 1 || boundary_idx > size(optical.boundary_materials, 1)
        return 2  # Absorb if invalid boundary
    end
    
    # Get material indices
    mat1_idx = optical.boundary_materials[boundary_idx, 1]
    mat2_idx = optical.boundary_materials[boundary_idx, 2]
    
    # Determine which material we're coming from/going to
    if from_outside
        from_mat = mat1_idx
        to_mat = mat2_idx
    else
        from_mat = mat2_idx  
        to_mat = mat1_idx
    end
    
    # Get refractive indices
    n1 = get_refractive_index(optical, from_mat, photon.wavelength)
    n2 = get_refractive_index(optical, to_mat, photon.wavelength)
    
    # Calculate angle of incidence (assuming normal is +z)
    cos_theta1 = abs(photon.mom[3])  # Simplified - should use actual surface normal
    
    # Get Fresnel reflection coefficients
    rs, rp, cos_theta2 = fresnel_reflection(n1, n2, cos_theta1)
    
    # Average for unpolarized light (could use actual polarization)
    reflectance = 0.5f0 * (rs + rp)
    
    # Check surface properties if present
    surf1_idx = optical.boundary_surfaces[boundary_idx, 1]
    surf2_idx = optical.boundary_surfaces[boundary_idx, 2]
    
    if surf1_idx >= 1 && surf1_idx <= size(optical.surface_reflectance, 1)
        surface_reflect = interpolate_property(optical.surface_reflectance, optical.wavelengths, surf1_idx, photon.wavelength)
        reflectance *= surface_reflect
    end
    
    # Sample interaction
    xi = uniform(rng)
    
    if xi < reflectance
        # Reflect - flip momentum component normal to surface
        photon.mom = SVector{3,Float32}(photon.mom[1], photon.mom[2], -photon.mom[3])
        return 1
    else
        # Transmit - update momentum direction for refraction
        if cos_theta2 > 0.0f0
            # Apply Snell's law (simplified for normal incidence)
            scale = n1 / n2
            photon.mom = SVector{3,Float32}(scale * photon.mom[1], scale * photon.mom[2], 
                                           sign(photon.mom[3]) * cos_theta2)
            normalize_momentum!(photon)
            return 0
        else
            # Total internal reflection  
            photon.mom = SVector{3,Float32}(photon.mom[1], photon.mom[2], -photon.mom[3])
            return 1
        end
    end
end