using Test
using Opticks
using CUDA
using LinearAlgebra
using StaticArrays

@testset "Opticks.jl Comprehensive Tests" begin
    
    @testset "SPhoton Type and Bit Packing" begin
        # Test SPhoton creation
        photon = SPhoton()
        @test photon.pos == SVector{3,Float32}(0.0f0, 0.0f0, 0.0f0)
        @test photon.time == 0.0f0
        @test photon.mom == SVector{3,Float32}(0.0f0, 0.0f0, 0.0f0)
        @test photon.pol == SVector{3,Float32}(0.0f0, 0.0f0, 0.0f0)
        @test photon.wavelength == 0.0f0
        @test photon_idx(photon) == 0x00000000
        @test orientation(photon) == 1.0f0
        
        # Test photon index bit packing
        set_photon_idx!(photon, 0x12345678)
        @test photon_idx(photon) == 0x12345678
        @test orientation(photon) == 1.0f0  # Should be preserved
        
        # Test orientation bit packing
        set_orientation!(photon, -1.0f0)
        @test orientation(photon) == -1.0f0
        @test photon_idx(photon) == 0x12345678  # Should be preserved
        
        # Test boundary and flag bit packing
        set_boundary!(photon, 0x00AB)
        @test boundary(photon) == 0x00AB
        
        set_flag!(photon, 0x000F)
        @test flag(photon) == 0x000F
        @test photon.flagmask == 0x000F
        
        # Test multiple flag accumulation
        set_flag!(photon, 0x00F0)
        @test photon.flagmask == 0x00FF  # Should accumulate
        
        # Test boundary preserved when setting flag
        set_flag!(photon, 0x0001)
        @test boundary(photon) == 0x00AB
    end
    
    @testset "Photon Utility Functions" begin
        # Test zero_photon!
        photon = SPhoton()
        photon.time = 10.0f0
        photon.wavelength = 500.0f0
        zero_photon!(photon)
        @test photon.time == 0.0f0
        @test photon.wavelength == 0.0f0
        @test photon_idx(photon) == 0x00000000
        
        # Test set_photon_position!
        set_photon_position!(photon, 1.0f0, 2.0f0, 3.0f0, 10.0f0)
        @test photon.pos ≈ SVector{3,Float32}(1.0f0, 2.0f0, 3.0f0)
        @test photon.time ≈ 10.0f0
        
        # Test set_photon_momentum!
        set_photon_momentum!(photon, 0.0f0, 0.0f0, 1.0f0)
        @test photon.mom ≈ SVector{3,Float32}(0.0f0, 0.0f0, 1.0f0)
        
        # Test set_photon_polarization!
        set_photon_polarization!(photon, 1.0f0, 0.0f0, 0.0f0)
        @test photon.pol ≈ SVector{3,Float32}(1.0f0, 0.0f0, 0.0f0)
        
        # Test set_photon_wavelength!
        set_photon_wavelength!(photon, 420.0f0)
        @test photon.wavelength ≈ 420.0f0
        
        # Test normalize_momentum!
        set_photon_momentum!(photon, 3.0f0, 4.0f0, 0.0f0)
        normalize_momentum!(photon)
        @test norm(photon.mom) ≈ 1.0f0
        @test photon.mom ≈ SVector{3,Float32}(0.6f0, 0.8f0, 0.0f0)
        
        # Test normalize_polarization!
        set_photon_momentum!(photon, 0.0f0, 0.0f0, 1.0f0)
        set_photon_polarization!(photon, 1.0f0, 1.0f0, 1.0f0)
        normalize_polarization!(photon)
        # Should be perpendicular to momentum
        @test abs(dot(photon.pol, photon.mom)) < 1e-6
        @test norm(photon.pol) ≈ 1.0f0
    end
    
    @testset "Random Number Generation" begin
        # Test OpticalRNG initialization
        rng = OpticalRNG(0x00000001, 0x00000000)
        @test rng.photon_idx == 0x00000001
        @test rng.event_idx == 0x00000000
        @test rng.counter == 0x00000000
        
        # Test uniform generation bounds
        for _ in 1:100
            u = uniform(rng)
            @test 0.0f0 <= u <= 1.0f0
        end
        
        # Test uniform generates different values
        rng1 = OpticalRNG(0x00000001, 0x00000000)
        rng2 = OpticalRNG(0x00000001, 0x00000000)
        u1 = uniform(rng1)
        u2 = uniform(rng1)
        @test u1 != u2
        
        # Test different seeds generate different sequences
        rng1 = OpticalRNG(0x00000001, 0x00000000)
        rng2 = OpticalRNG(0x00000002, 0x00000000)
        vals1 = [uniform(rng1) for _ in 1:10]
        vals2 = [uniform(rng2) for _ in 1:10]
        @test vals1 != vals2
        
        # Test uniform_sphere generates unit vectors
        rng = OpticalRNG(0x00000001, 0x00000000)
        for _ in 1:100
            dir = uniform_sphere(rng)
            @test length(dir) == 3
            @test norm(dir) ≈ 1.0f0 atol=1e-5
        end
        
        # Test uniform_sphere coverage
        rng = OpticalRNG(0x00000001, 0x00000000)
        dirs = [uniform_sphere(rng) for _ in 1:1000]
        # Check that we get both positive and negative components
        @test any(d[1] > 0.5 for d in dirs)
        @test any(d[1] < -0.5 for d in dirs)
        @test any(d[2] > 0.5 for d in dirs)
        @test any(d[2] < -0.5 for d in dirs)
        @test any(d[3] > 0.5 for d in dirs)
        @test any(d[3] < -0.5 for d in dirs)
        
        # Test gauss_shoot
        rng = OpticalRNG(0x00000001, 0x00000000)
        mean = 5.0f0
        stddev = 2.0f0
        samples = [gauss_shoot(rng, mean, stddev) for _ in 1:1000]
        sample_mean = sum(samples) / length(samples)
        sample_std = sqrt(sum((s - sample_mean)^2 for s in samples) / (length(samples) - 1))
        @test abs(sample_mean - mean) < 0.2  # Within 10% of expected
        @test abs(sample_std - stddev) < 0.3  # Within 15% of expected
        
        # Test exponential_shoot
        rng = OpticalRNG(0x00000001, 0x00000000)
        tau = 10.0f0
        samples = [exponential_shoot(rng, tau) for _ in 1:1000]
        sample_mean = sum(samples) / length(samples)
        @test abs(sample_mean - tau) < 1.0  # Within 10% of expected mean
        @test all(s >= 0.0 for s in samples)  # All non-negative
    end
    
    @testset "Material Properties" begin
        # Test MaterialProperties creation
        wavelengths = Float32[300, 400, 500, 600, 700]
        mat = MaterialProperties("water", wavelengths)
        
        @test mat.name == "water"
        @test length(mat.rindex) == 5
        @test length(mat.absorption) == 5
        @test length(mat.scattering) == 5
        @test length(mat.reemission) == 5
        @test length(mat.wavelengths) == 5
        @test mat.wavelengths == wavelengths
        
        # Test default values
        @test all(mat.rindex .== 1.0f0)  # Default vacuum
        @test all(mat.absorption .== 1e6f0)  # Very long
        @test all(mat.scattering .== 1e6f0)  # Very long
        @test all(mat.reemission .== 0.0f0)  # No re-emission
        
        # Test setting custom values
        mat.rindex .= 1.33f0
        @test all(mat.rindex .== 1.33f0)
    end
    
    @testset "Cerenkov Physics" begin
        # Test Cerenkov angle calculation
        n = 1.33f0  # Water
        beta_inv = 1.0f0  # v = c
        cos_theta = cerenkov_angle(n, beta_inv)
        expected_cos_theta = beta_inv / n
        @test cos_theta ≈ expected_cos_theta
        
        # Test physical limits (cos_theta <= 1)
        beta_inv = 0.5f0  # v = 2c (unphysical, but test clamp)
        cos_theta = cerenkov_angle(n, beta_inv)
        @test cos_theta <= 1.0f0
        
        # Test CerenkovGenstep creation
        genstep = CerenkovGenstep()
        @test genstep.gentype == 0
        @test genstep.numphoton == 0
        @test genstep.pos == SVector{3,Float32}(0, 0, 0)
        @test genstep.time == 0.0f0
        @test genstep.beta_inverse == 1.0f0
    end
    
    @testset "Scintillation Physics" begin
        # Test ScintillationProperties creation
        wavelengths = Float32[300, 400, 500, 600, 700]
        emission = ones(Float32, length(wavelengths))
        scint_props = ScintillationProperties(emission, wavelengths)
        
        @test length(scint_props.emission_spectrum) == 5
        @test length(scint_props.wavelengths) == 5
        @test scint_props.fast_time_constant == 1.0f0
        @test scint_props.slow_time_constant == 10.0f0
        @test scint_props.yield_ratio == 1.0f0
        @test scint_props.birks_constant == 0.01f0
        
        # Test compute_cumulative_emission
        emission_matrix = ones(Float32, 2, 5)
        cumulative = compute_cumulative_emission(emission_matrix)
        @test size(cumulative) == (2, 5)
        @test cumulative[1, 1] == 0.0f0
        @test cumulative[1, end] == 1.0f0  # Normalized
        @test all(diff(cumulative[1, :]) .>= 0)  # Monotonic
        
        # Test ScintillationGenstep creation
        genstep = ScintillationGenstep()
        @test genstep.gentype == 0
        @test genstep.numphoton == 0
        @test genstep.pos == SVector{3,Float32}(0, 0, 0)
        @test genstep.time == 0.0f0
        @test genstep.scintillation_time == 1.0f0
        @test genstep.yield_ratio == 1.0f0
    end
    
    @testset "PMT Properties" begin
        # Test PMTProperties creation
        wavelengths = Float32[300, 400, 500, 600, 700]
        pmt = PMTProperties("test_pmt", Int32(1), wavelengths)
        
        @test pmt.name == "test_pmt"
        @test pmt.pmt_type == 1
        @test length(pmt.quantum_efficiency) == 5
        @test length(pmt.wavelengths) == 5
        @test all(pmt.quantum_efficiency .== 0.0f0)  # Default zero
        @test pmt.transit_time == 10.0f0
        @test pmt.transit_time_spread == 2.0f0
        @test pmt.rise_time == 1.0f0
        @test pmt.fall_time == 5.0f0
        @test pmt.cathode_radius == 50.0f0
        @test pmt.cathode_thickness == 3.0f0
        @test pmt.dark_rate == 100.0f0
        @test pmt.afterpulsing_prob == 0.01f0
        
        # Test PMTHit creation
        hit = PMTHit()
        @test hit.pmt_id == 0
        @test hit.photon_id == 0
        @test hit.detected == false
        @test hit.detection_time == 0.0f0
    end
    
    @testset "Surface and Boundary Properties" begin
        # Test SurfaceProperties creation
        wavelengths = Float32[300, 400, 500, 600, 700]
        surf = SurfaceProperties("test_surface", wavelengths)
        
        @test surf.name == "test_surface"
        @test surf.model == "unified"
        @test surf.finish == "polished"
        @test surf.type == "dielectric_dielectric"
        @test length(surf.transmittance) == 5
        @test length(surf.reflectance) == 5
        @test all(surf.reflectance .== 1.0f0)  # Default full reflection
        @test surf.sigma_alpha == 0.0f0
        @test surf.polish == 1.0f0
        
        # Test BoundaryProperties creation
        boundary = BoundaryProperties(Int32(1), Int32(2))
        @test boundary.material1_idx == 1
        @test boundary.material2_idx == 2
        @test boundary.surface1_idx == -1
        @test boundary.surface2_idx == -1
        
        boundary2 = BoundaryProperties(Int32(1), Int32(2), Int32(3), Int32(4))
        @test boundary2.surface1_idx == 3
        @test boundary2.surface2_idx == 4
    end
    
    if CUDA.functional()
        @testset "CUDA Basic Operations" begin
            # Test photon array on GPU
            n_photons = 100
            photons = CuArray{SPhoton}(undef, n_photons)
            
            # Test zeroing photons on GPU
            zero_photons!(photons)
            
            # Copy back and check
            cpu_photons = Array(photons)
            @test all(p -> p.time == 0.0f0, cpu_photons)
            @test all(p -> photon_idx(p) == 0x00000000, cpu_photons)
            @test all(p -> p.wavelength == 0.0f0, cpu_photons)
        end
        
        @testset "Cerenkov Generation on GPU" begin
            # Setup materials with refractive index
            wavelengths = Float32[300:50:700;]
            num_wl = length(wavelengths)
            
            # Create material properties (water)
            rindex_matrix = ones(Float32, 1, num_wl) .* 1.33f0
            absorption_matrix = ones(Float32, 1, num_wl) .* 1e6f0
            
            # Create QCerenkov
            qcerenkov = QCerenkov(rindex_matrix, absorption_matrix, wavelengths)
            
            # Create genstep with proper constructor
            genstep = CerenkovGenstep(
                UInt32(0), UInt32(1), UInt32(1), UInt32(10),  # gentype, trackid, matline, numphoton
                SVector{3,Float32}(0.0f0, 0.0f0, 0.0f0), 0.0f0,  # pos, time
                SVector{3,Float32}(0.0f0, 0.0f0, 10.0f0), 10.0f0,  # delta_position, step_length
                Int32(0), 0.0f0, 1.0f0, 1.0f0,  # code, charge, weight, pre_velocity
                1.0f0  # beta_inverse
            )
            
            gensteps_gpu = CuArray([genstep])
            
            # Create photon and RNG arrays
            n_photons = 10
            photons_gpu = CuArray{SPhoton}(undef, n_photons)
            rngs_gpu = CuArray([OpticalRNG(UInt32(i), 0x00000000) for i in 1:n_photons])
            
            # Generate Cerenkov photons
            generate_cerenkov_photons!(photons_gpu, rngs_gpu, gensteps_gpu, qcerenkov)
            
            # Copy back and verify
            photons_cpu = Array(photons_gpu)
            @test all(p -> p.wavelength > 0.0f0, photons_cpu)
            @test all(p -> p.wavelength >= 300.0f0, photons_cpu)
            @test all(p -> p.wavelength <= 700.0f0, photons_cpu)
            @test all(p -> norm(p.mom) ≈ 1.0f0, photons_cpu)
            @test all(p -> norm(p.pol) ≈ 1.0f0, photons_cpu)
            # Check photons are along the step
            @test all(p -> 0.0f0 <= p.pos[3] <= 10.0f0, photons_cpu)
        end
        
        @testset "Scintillation Generation on GPU" begin
            # Setup scintillation properties
            wavelengths = Float32[300:50:700;]
            emission = ones(Float32, length(wavelengths))
            scint_props = [ScintillationProperties(emission, wavelengths, 1.0f0, 10.0f0)]
            
            # Create QScint
            qscint = QScint(scint_props)
            
            # Create genstep with proper constructor
            genstep = ScintillationGenstep(
                UInt32(0), UInt32(1), UInt32(1), UInt32(10),  # gentype, trackid, matline, numphoton
                SVector{3,Float32}(0.0f0, 0.0f0, 0.0f0), 0.0f0,  # pos, time
                SVector{3,Float32}(0.0f0, 0.0f0, 10.0f0), 10.0f0,  # delta_position, step_length
                Int32(0), 0.0f0, 1.0f0,  # code, charge, weight
                1.0f0, 1.0f0  # scintillation_time, yield_ratio
            )
            
            gensteps_gpu = CuArray([genstep])
            
            # Create photon and RNG arrays
            n_photons = 10
            photons_gpu = CuArray{SPhoton}(undef, n_photons)
            rngs_gpu = CuArray([OpticalRNG(UInt32(i), 0x00000000) for i in 1:n_photons])
            
            # Generate scintillation photons
            generate_scintillation_photons!(photons_gpu, rngs_gpu, gensteps_gpu, qscint)
            
            # Copy back and verify
            photons_cpu = Array(photons_gpu)
            @test all(p -> p.wavelength > 0.0f0, photons_cpu)
            @test all(p -> p.wavelength >= 300.0f0, photons_cpu)
            @test all(p -> p.wavelength <= 700.0f0, photons_cpu)
            @test all(p -> norm(p.mom) ≈ 1.0f0, photons_cpu)
            @test all(p -> norm(p.pol) ≈ 1.0f0, photons_cpu)
            # Check isotropic emission
            @test length(unique(p.mom for p in photons_cpu)) > 5
        end
        
        @testset "QOptical GPU Operations" begin
            # Test QOptical creation
            wavelengths = Float32[300:50:700;]
            
            water = MaterialProperties("water", wavelengths)
            water.rindex .= 1.33f0
            water.absorption .= 1000.0f0
            
            air = MaterialProperties("air", wavelengths)
            air.rindex .= 1.0f0
            
            materials = [air, water]
            surfaces = SurfaceProperties[]
            boundaries = BoundaryProperties[]
            
            qoptical = QOptical(materials, surfaces, boundaries)
            
            # Verify GPU arrays were created
            @test size(qoptical.material_rindex) == (2, length(wavelengths))
            @test size(qoptical.material_absorption) == (2, length(wavelengths))
            
            # Copy back and verify values
            rindex_cpu = Array(qoptical.material_rindex)
            @test all(rindex_cpu[1, :] .≈ 1.0f0)  # Air
            @test all(rindex_cpu[2, :] .≈ 1.33f0)  # Water
        end
        
        @testset "QPMT GPU Operations" begin
            # Test QPMT creation
            wavelengths = Float32[300:50:700;]
            
            pmt1 = PMTProperties("pmt1", 1, wavelengths)
            pmt1.quantum_efficiency .= 0.2f0
            
            pmt2 = PMTProperties("pmt2", 2, wavelengths)
            pmt2.quantum_efficiency .= 0.3f0
            
            pmts = [pmt1, pmt2]
            qpmt = QPMT(pmts)
            
            # Verify GPU arrays were created
            @test size(qpmt.quantum_efficiency) == (2, length(wavelengths))
            
            # Copy back and verify values
            qe_cpu = Array(qpmt.quantum_efficiency)
            @test all(qe_cpu[1, :] .≈ 0.2f0)
            @test all(qe_cpu[2, :] .≈ 0.3f0)
        end
        
    else
        @warn "CUDA not functional, skipping GPU tests"
    end
    
    @testset "Integration Tests" begin
        # Test that components work together
        wavelengths = Float32[300:50:700;]
        
        # Create materials
        water = MaterialProperties("water", wavelengths)
        water.rindex .= 1.33f0
        water.absorption .= 5000.0f0
        
        air = MaterialProperties("air", wavelengths)
        air.rindex .= 1.0f0
        
        # Create photon and test propagation through materials
        photon = SPhoton()
        set_photon_position!(photon, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
        set_photon_momentum!(photon, 0.0f0, 0.0f0, 1.0f0)
        set_photon_wavelength!(photon, 420.0f0)
        
        @test photon.pos[3] == 0.0f0
        @test photon.mom[3] == 1.0f0
        @test photon.wavelength == 420.0f0
    end
end