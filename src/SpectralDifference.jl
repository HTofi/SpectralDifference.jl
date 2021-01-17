module SpectralDifference

    include("InputParameters.jl")
    include("BasicRoutines.jl")
    include("ArtificialViscosity1D.jl")
    include("DataIO.jl")
    include("FluxReconstruction1D.jl")
    include("Initializer1D.jl")
    include("Meshing1D.jl")
    include("RiemannSolver.jl")
    include("TimeMarching1D.jl")

end
