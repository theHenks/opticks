using Test
using Opticks
using CUDA

@testset "Opticks.jl Tests" begin
    
    @testset "Basic Types" begin
        # Test SPhoton creation and manipulation
        photon = SPhoton()
        @test photon.pos == [0.0f0, 0.0f0, 0.0f0]
        @test photon.time == 0.0f0
        @test photon_idx(photon) == 0x00000000
        @test orientation(photon) == 1.0f0
        
        # Test setting properties
        set_photon_idx!(photon, 0x12345678)
        @test photon_idx(photon) == 0x12345678
        @test orientation(photon) == 1.0f0  # Should be preserved
        
        set_orientation!(photon, -1.0f0)
        @test orientation(photon) == -1.0f0
        @test photon_idx(photon) == 0x12345678  # Should be preserved
    end
    
    @testset "Photon Utilities" begin
        # Test photon manipulation functions
        photon = SPhoton()
        zero_photon!(photon)
        
        set_photon_position!(photon, 1.0f0, 2.0f0, 3.0f0, 10.0f0)
        @test photon.pos ≈ [1.0f0, 2.0f0, 3.0f0]
        @test photon.time ≈ 10.0f0
        
        set_photon_momentum!(photon, 0.0f0, 0.0f0, 1.0f0)
        @test photon.mom ≈ [0.0f0, 0.0f0, 1.0f0]
        
        set_photon_wavelength!(photon, 420.0f0)
        @test photon.wavelength ≈ 420.0f0
    end
    
    @testset "Material Properties" begin
        # Test material property creation
        wavelengths = Float32[300, 400, 500, 600, 700]
        mat = MaterialProperties("water", wavelengths)
        
        @test mat.name == "water"
        @test length(mat.rindex) == 5
        @test length(mat.wavelengths) == 5
        @test mat.wavelengths == wavelengths
    end
    
    @testset "Random Number Generation" begin
        # Test RNG functionality
        rng = OpticalRNG(0x00000001, 0x00000000)
        
        # Test uniform generation
        u1 = uniform(rng)
        u2 = uniform(rng)
        @test 0.0f0 <= u1 <= 1.0f0
        @test 0.0f0 <= u2 <= 1.0f0
        @test u1 != u2  # Should be different
        
        # Test sphere sampling
        dir = uniform_sphere(rng)
        @test length(dir) == 3
        norm = sqrt(dir[1]^2 + dir[2]^2 + dir[3]^2)
        @test norm ≈ 1.0f0 atol=1e-6
    end
    
    if CUDA.functional()
        @testset "CUDA Functionality" begin
            # Test basic CUDA operations
            n_photons = 100
            photons = CuArray{SPhoton}(undef, n_photons)
            
            # Test zeroing photons on GPU
            zero_photons!(photons)
            
            # Copy back and check
            cpu_photons = Array(photons)
            @test all(p -> p.time == 0.0f0, cpu_photons)
            @test all(p -> photon_idx(p) == 0x00000000, cpu_photons)
        end
        
        @testset "Physics Simulation" begin
            # Test basic physics setup
            wavelengths = Float32[200:50:800;]
            
            # Create simple materials
            water = MaterialProperties("water", wavelengths)
            water.rindex .= 1.33f0  # Constant refractive index
            
            air = MaterialProperties("air", wavelengths)
            air.rindex .= 1.0f0
            
            materials = [air, water]
            surfaces = SurfaceProperties[]
            boundaries = [BoundaryProperties(1, 2)]
            
            # Create simple PMT
            pmt_props = [PMTProperties("test_pmt", 1, wavelengths)]
            pmt_props[1].quantum_efficiency .= 0.2f0  # 20% QE
            
            # Create simple scintillator
            scint_props = [ScintillationProperties(ones(Float32, length(wavelengths)), wavelengths)]
            
            # Test simulation context creation
            ctx = simulate_optical_photons(materials, surfaces, boundaries, 
                                         pmt_props, materials, scint_props)
            
            @test ctx isa SimulationContext
            @test ctx.max_bounces > 0
            @test ctx.time_window > 0.0f0
        end
    else
        @warn "CUDA not functional, skipping GPU tests"
    end
end