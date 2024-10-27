## Setup the environment for the exercise by activating the local environment ##
using Pkg;
cd(@__DIR__);
pwd()
Pkg.activate(".");
Pkg.instantiate();
################################################################################


## Add the necessary packages ##
using Plots;


struct Function1D <: Function
    f::Function
end

function (Ω::Function1D)(x::Real)
    return Ω.f(x)
end

abstract type  Domain end

struct Domain1D <: Domain
    a::Real
    b::Real
end

abstract type Poisson1D end

abstract type NumericalPoisson1D end

struct AnalyticalPoisson1D <: Poisson1D
    u::Function1D
    Ω::Domain1D
end


abstract type BoundaryCondition end
struct Neumann <: BoundaryCondition
    gN::Real
end

struct Dirichlet <: BoundaryCondition
    gD::Real  
end

struct FiniteVolumePoisson1D <: NumericalPoisson1D
    S::Function1D # Source term
    Ω::Domain1D
    BC::NamedTuple{(:a, :b), Tuple{BoundaryCondition, BoundaryCondition}}
end


function (u::AnalyticalPoisson1D)(x::Real)
    # check if x is in the domain
    if x < u.Ω.a || x > u.Ω.b
        error("x is not in the domain")
    end
    return u.u(x)
end

function solve(p::AnalyticalPoisson1D,N::Int)
    Ω = p.Ω
    h = (Ω.b - Ω.a) / N
    x = range(Ω.a + h/2, stop=Ω.b-h/2, length=N)
    u = zeros(N)
    for i in 1:N
        u[i] = p(x[i])
    end
    return Solution(u, x)
end


function solve(p::FiniteVolumePoisson1D,N::Int)
    # We need to construct AQ = S
    Ω = p.Ω
    source = p.S
    h = (Ω.b - Ω.a) / N
    x = range(Ω.a+h/2, stop=Ω.b - h/2, length=N)
    A = zeros(N, N)
    Q = zeros(N)
    S = x .|> ((x) -> source(x))
    BC = p.BC
    a = BC.a
    b = BC.b
    # Left boundary
    if(a isa Neumann)
       A[1, 1] = 1/h^2
       A[1, 2] = -1/h^2
       S[1] += -a.gN/h
    elseif(a isa Dirichlet)
        A[1, 1] = 3/h^2
        A[1, 2] = -1/h^2
        S[1] += 2*a.gD/h^2
    end
    for i in 2:N-1
        A[i, i-1] = -1/h^2
        A[i, i] = +2/h^2
        A[i, i+1] = -1/h^2
    end
    # Right boundary
    if(b isa Neumann)
       A[N, N] = 1/h^2
       A[N, N-1] = -1/h^2
       S[N] += b.gN/h
    elseif(b isa Dirichlet)
        A[N, N] = 3/h^2
        A[N, N-1] = -1/h^2
        S[N] += 2*b.gD/h^2
    end
    Q = A \ S
    return Solution(Q,x)
end

struct Solution
    u::Vector{Real}
    x::Vector{Real}
end

struct ErrorEstimator
    exact::AnalyticalPoisson1D
    numerical::NumericalPoisson1D
    hs::Vector{Real}
end

struct Error
    hs::Vector{Real}
    error::Vector{Real}
end

function error(estimator::ErrorEstimator)
    exact = estimator.exact
    numerical = estimator.numerical
    hs = estimator.hs
    error = zeros(length(hs))
    for i in 1:length(hs)
        h_error =  0.0;
        N = round(Int, (exact.Ω.b - exact.Ω.a) / hs[i])
        u_exact = solve(exact, N)
        u_numerical = solve(numerical, N)
        for j in 1:length(hs)
            e =  abs(u_exact.u[j] - u_numerical.u[j])
            h_error  = max(h_error, e)
        end
        error[i] = h_error
    end
    return Error(hs, error)
    
end


## Solve Task C ##

Ω = Domain1D(0.0, 1.0)
u_case1 = AnalyticalPoisson1D(Function1D(x -> sin(π*x)), Ω)
u_case2 = AnalyticalPoisson1D(Function1D(x -> cos(π*x)), Ω)
u_case1_fv = FiniteVolumePoisson1D(Function1D(x -> π^2 *sin(π*x)), Ω, (Dirichlet(0.0), Dirichlet(0.0)))
u_case2_fv = FiniteVolumePoisson1D(Function1D(x -> π^2 *cos(π*x)), Ω, (Dirichlet(1.0), Neumann(0.0)))

sol_anal_case1 =  solve(u_case1, 50)
sol_anal_case2 = solve(u_case2, 50)
sol_fv_case1 = solve(u_case1_fv, 50)
sol_fv_case2 = solve(u_case2_fv, 50)

plot(sol_anal_case1.x, sol_anal_case1.u, label="sin(πx)", xlabel="x", ylabel="u(x)", title="Analytical Solution")
plot!(sol_anal_case2.x, sol_anal_case2.u, label="cos(πx)")
## Plot the solution of the finite volume method (scatter)##
scatter!(sol_fv_case1.x, sol_fv_case1.u, label="Finite Volume Method")
scatter!(sol_fv_case2.x, sol_fv_case2.u, label="Finite Volume Method")

## Solve Task D ##
## Compute the error estimator for the finite volume method ##
i = 5:15 .|> Float64 
hs = 2 .^ (- i)
error_case1  = ErrorEstimator(u_case1, u_case1_fv, hs) |> error
error_case2 = ErrorEstimator(u_case2, u_case2_fv, hs) |> error

## Plot the error estimator for the finite volume method (log scale)##
plot(hs, error_case1.error, label="sin(πx)", xlabel="h", ylabel="Error", title="Error Estimator", yscale=:log2)
plot!(hs, error_case2.error, label="cos(πx)")

