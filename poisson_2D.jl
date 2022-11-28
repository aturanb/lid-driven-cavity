using SparseArrays
using LinearAlgebra
import Plots
import PlotlyJS
using Statistics

# -------------------------------Preliminary functions----------------------------------------------------
function RMSE(field_1, field_2)
    return sqrt(mean( (field_1 - field_2).^2 ))
end

function sparse_laplace_stencil(N)
    I1 = collect(1:N)
    J1 = copy(I1)
    V1 = -2 * ones(N)

    I2 = collect(2:N)
    J2 = collect(1:N-1)
    V2 = ones(N-1)

    A = sparse(I1, J1, V1)  + sparse(I2, J2, V2, N, N) + sparse(J2, I2, V2, N, N)
    return A
end

function sparse_identity(N)
    I = J = collect(1:N)
    V = ones(N)
    return sparse(I, J, V)
end
# --------------------------------------------------------------------------------------------------------




# -------------------------------------------Solver-------------------------------------------------------
"""
    poisson_solve_2D(source_func, Nx, Ny; left_bound, right_bound, top_bound, bottom_bound)

Solve Poisson's Equation in 2 dimensions ∇^2 p = f(x, y). Return vectors `x` and `y` and 2D array `p`.

# Arguments
- `source_func::Function`: the source function on the RHS, f(x, y).
    Written to do everything element-wise. In practice, I feed it a vector `x` and a scalar `y` and it
    returns a vector with values at `x[2:Nx]` and `y[j]` for some `j` (see line 93).
- `Nx::Integer`: sets the grid spacing in the x-direction. 
    There will be Nx + 1 total nodes in the x-direction, and Nx - 1 interior nodes in the x-direction.
- `Ny::Integer`: same as `Nx`, but for the y-direction. Issues arise when Nx != Ny that I have not yet been 
    able to resolve (different from issues with nonzero boundary conditions).
- `xf::Float64`: the final x position. `x[end] = xf`.
- `yf::Float64`: same as `xf`, but for y.
- `[left/right/etc.]_bound::Vector{Float64}`: vectors containing the boundary values for the output field 
    (be careful!).
"""
function poisson_solve_2D(source_func, Nx, Ny, xf, yf; 
    left_bound, right_bound, top_bound, bottom_bound)

    # make physical space
    dx = xf/Nx
    dy = yf/Ny
        
    x = collect(0:dx:xf)
    y = collect(0:dy:yf)

    # differential operators
    Dxx = sparse_laplace_stencil(Nx-1) / (dx^2)
    Dyy = sparse_laplace_stencil(Ny-1) / (dy^2)

    Ix = sparse_identity(Nx-1)
    Iy = sparse_identity(Ny-1)

    A = kron(Iy, Dxx) + kron(Dyy, Ix)

    # SCALAR FIELD TO SOLVE FOR
    p = zeros(Nx+1, Ny+1) 

    # fill in boundary conditions
    p[1, :]   .= left_bound    # x=0
    p[:, 1]   .= bottom_bound  # y=0
    p[end, :] .= right_bound   # x=xf
    p[:, end] .= top_bound     # y=yf

    function b_j(x, y, j, p; dx=dx, dy=dy)
        # jth vector of vector `b` where "Ap = b", and p is a stacked vector of the interior nodes 
        # of a 2D field.
        Nx, Ny = size(p) .- 1
        extra_vec = zeros(Nx-1)
        if j == 2
            extra_vec = -p[2:Nx, 1] / dy^2
        elseif j == Ny
            extra_vec = -p[2:Nx, Ny+1] / dy^2
        end
        
        bj = source_func(x[2:Nx], y[j])
        bj[1] -= p[1, j]/dx^2
        bj[end] -= p[Nx+1, j]/dx^2
        bj .+= extra_vec
        return bj
    end

    # fill in b vector
    b = zeros((Nx-1) * (Ny-1))
    for j = 2:Ny
        start_index = 1 + (j-2)*(Nx-1)
        end_index = (j-1)*(Nx-1)
        b[start_index:end_index] .= b_j(x, y, j, p, dx=dx, dy=dy)
    end

    # solve A * p_int_vec = b, where p_int_vec contains all of the interior values of p
    p_int_vec = A \ b

    # interior values of p reshaped into a matrix. be sure to transpose it.
    p_int = reshape(p_int_vec, Nx-1, Ny-1)  # note the ' character to take adjoint

    # fill in main matrix for p
    p[2:end-1, 2:end-1] .= p_int

    p = p

    return (x, y, p)

end
# --------------------------------------------------------------------------------------------------------




# -----------------------Exact Soln and Source Functions---------------------------------------------------
function p_true_nonzero_bnds(x, y)
    # inputs are 1-D vectors. function will create meshgrid for them.
    x_mesh = x * ones(length(y))'
    y_mesh = ones(length(x)) * y'
    p = sin.(π*x_mesh) .* cos.(2*π*y_mesh)

    return p
end

function p_true_zero_bnds(x, y)
    # inputs are 1-D vectors. function will create meshgrid for them.
    x_mesh = x * ones(length(y))'
    y_mesh = ones(length(x)) * y'
    p = sin.(π*x_mesh) .* sin.(2*π*y_mesh)

    return p
end

function source_nonzero_bnds(x, y)
    # RHS of Poisson equation ∇^2 P = f(x, y)
    # in this case, determined by manufacturing a solution for P 
    return -5 * π^2 * sin.(π*x) .* cos.(2*π*y)
end

function source_zero_bnds(x, y)
    # RHS of Poisson equation ∇^2 P = f(x, y)
    # in this case, determined by manufacturing a solution for P 
    return -5 * π^2 * sin.(π*x) .* sin.(2*π*y)
end
# --------------------------------------------------------------------------------------------------------




# ------------------------------------------Set spatial grid----------------------------------------------
Nx = 32
Ny = 64
xf = 1.
yf = 2.
# --------------------------------------------------------------------------------------------------------




# -------------------Use this for the `nonzero_bnds` exact solution and source function-------------------
dx = xf/Nx
x_vals = collect(0:dx:xf)
top = bottom = sin.(π*x_vals)

x, y, p = poisson_solve_2D(source_nonzero_bnds, Nx, Ny, xf, yf; 
        left_bound=zeros(Ny+1), right_bound=zeros(Ny+1), top_bound=top, bottom_bound=bottom)

exact_soln = p_true_nonzero_bnds(x, y)
# --------------------------------------------------------------------------------------------------------




# -------------------Use this for the `zero_bnds` exact solution and source function----------------------
# top = bottom = zeros(Nx+1)
# x, y, p = poisson_solve_2D(source_zero_bnds, Nx, Ny, xf, yf; 
#         left_bound=zeros(Ny+1), right_bound=zeros(Ny+1), top_bound=top, bottom_bound=bottom)

# exact_soln = p_true_zero_bnds(x, y)
# --------------------------------------------------------------------------------------------------------




# If you want interactive plots, you must add package "PlotlyJS"  & uncomment `import PlotlyJS` at the top.
# (It's very nice). Otherwise, use the code after the block below to plot with the "Plots" package.

# --------------------------------------------------------------------------------------------------------
fig = PlotlyJS.make_subplots(
        rows=1, cols=2,
        specs=fill(PlotlyJS.Spec(kind="scene"), 1, 2),
        column_titles=["exact solution", "numerical solution"]
)
PlotlyJS.add_trace!(fig, PlotlyJS.surface(x=x, y=y, z=exact_soln, 
    showscale=false), row=1, col=1)
PlotlyJS.add_trace!(fig, PlotlyJS.surface(x=x, y=y, z=p, 
    showscale=false), row=1, col=2)
fig
# --------------------------------------------------------------------------------------------------------




# For "Plots" I need to transpose the solutions for the x-, y-axes to make sense, 
# and for "PlotlyJS" I don't. Weird.

# --------------------------------------------------------------------------------------------------------
# p1 = Plots.surface(x, y, exact_soln', title="Exact Solution", cbar=false)
# p2 = Plots.surface(x, y, p', title="Numerical Solution", cbar=false)
# # Plots.plot(p1)
# Plots.plot(p1, p2, layout=(1, 2))
# Plots.xlabel!("x")
# Plots.ylabel!("y")
# --------------------------------------------------------------------------------------------------------