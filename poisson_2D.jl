using SparseArrays
using LinearAlgebra
import Plots
import PlotlyJS
using Statistics

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

# issues when Nx != Ny
Nx = 16
Ny = 16

xf = 1.
yf = 1.

dx = xf/Nx
dy = yf/Ny

x = collect(0:dx:xf)
y = collect(0:dy:yf)

Dxx = sparse_laplace_stencil(Nx-1) / (dx^2)
Dyy = sparse_laplace_stencil(Ny-1) / (dy^2)

Ix = sparse_identity(Nx-1)
Iy = sparse_identity(Ny-1)

A = kron(Dxx, Iy) + kron(Ix, Dyy)


p = zeros(Nx+1, Ny+1)
# fill in boundary conditions
p[1, :]   .= 0  # x=0
p[:, 1]   .= 0  # y=0
p[end, :] .= 0  # x=xf
p[:, end] .= 0  # y=yf


function f(x, y)
    # RHS of Poisson equation ∇^2 P = f(x, y)
    # in this case, determined by manufacturing a solution for P 
    return -5 * π^2 * sin.(π*x) .* sin.(2*π*y)
end

function b_j(x, y, j; p_mat=p, dx=dx, dy=dy)
    # jth vector of vector `b` where "Ax = b", and x is a stacked vector of the interior nodes of a 2D field
    Nx, Ny = size(p_mat) .- 1
    extra_vec = zeros(Nx-1)
    if j == 2
        extra_vec = -p_mat[2:Nx, 1] / dy^2
    elseif j == Ny
        extra_vec = -p_mat[2:Nx, Ny+1] / dy^2
    end
    bj = f(x[2:Nx], y[j])
    bj[1] -= p_mat[1, j]/dx^2
    bj[end] -= p_mat[Nx+1, j]/dx^2
    bj .+= extra_vec
    return bj
end


# fill in b vector
b = zeros((Nx-1) * (Ny-1))
for j = 2:Ny
    start_index = 1 + (j-2)*(Nx-1)
    end_index = (j-1)*(Nx-1)
    b[start_index:end_index] .= b_j(x, y, j, p_mat=p, dx=dx, dy=dy)
end

# solve A * p_int_vec = b, where p_int_vec contains all of the interior values of p
p_int_vec = A \ b

# interior values of p reshaped into a matrix. be sure to transpose it.
p_int = reshape(p_int_vec, Ny-1, Nx-1)'  # note the ' character to take adjoint

# fill in main matrix for p
p[2:end-1, 2:end-1] .= p_int


function p_true(x, y)
    # inputs are 1-D vectors. function will create meshgrid for them.
    x_mesh = x * ones(length(y))'
    y_mesh = ones(length(x)) * y'
    p = sin.(π*x_mesh) .* sin.(2*π*y_mesh)

    # enforce boundary conditions
    p[1, :]   .= 0  # x=0
    p[:, 1]   .= 0  # y=0
    p[end, :] .= 0  # x=xf
    p[:, end] .= 0  # y=yf

    return p
end

exact_soln = p_true(x, y)'


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

# p1 = Plots.surface(x, y, exact_soln, title="Exact Solution", cbar=false)
# p2 = Plots.surface(x, y, p, title="Numerical Solution", cbar=false)
# # Plots.plot(p1)
# Plots.plot(p1, p2, layout=(1, 2))
# Plots.xlabel!("x")
# Plots.ylabel!("y")

# write a function that generalizes this solver for arbitrary number of nodes, source function (f), and BCs
"""
    poisson_solve_2D(source_func, Nx, Ny; left_bound, right_bound, top_bound, bottom_bound)

Solve Poisson's Equation in 2 dimensions ∇^2 p = f(x, y). Return vectors `x` and `y` and 2D array `p`.

# Arguments
- `source_func::Function`: the source function on the RHS, f(x, y).
    Takes in vectors `x` and `y` and returns 2D array with values for each pair `x[i]` and `y[j]`
- `Nx::Integer`: sets the grid spacing in the x-direction. 
    There will be Nx + 1 total nodes in the x-direction, and Nx - 1 interior nodes in the x-direction.
- `Ny::Integer`: same as `Nx`, but for the y-direction. Issues arise when Nx != Ny that I have not yet been 
    able to resolve.
- `xf::Float64`: the final x position. `x[end] = xf`.
- `yf::Float64`: same as `xf`, but for y.
- `[left/right/etc.]_bound::Vector{Float64}`: vectors containing the boundary values for the output field 
    (be careful!).
"""
function poisson_solve_2D(source_func, Nx, Ny, xf, yf; 
    left_bound, right_bound, top_bound, bottom_bound)

    dx = xf/Nx
    dy = yf/Ny
        
    x = collect(0:dx:xf)
    y = collect(0:dy:yf)

    Dxx = sparse_laplace_stencil(Nx-1) / (dx^2)
    Dyy = sparse_laplace_stencil(Ny-1) / (dy^2)

    Ix = sparse_identity(Nx-1)
    Iy = sparse_identity(Ny-1)

    A = kron(Dxx, Iy) + kron(Ix, Dyy)

    p = zeros(Nx+1, Ny+1)
    # fill in boundary conditions
    p[1, :]   .= left_bound    # x=0
    p[:, 1]   .= bottom_bound  # y=0
    p[end, :] .= right_bound   # x=xf
    p[:, end] .= top_bound     # y=yf

    function b_j(x, y, j; p_mat=p, dx=dx, dy=dy)
        # jth vector of vector `b` where "Ax = b", and x is a stacked vector of the interior nodes of a 2D field
        Nx, Ny = size(p_mat) .- 1
        extra_vec = zeros(Nx-1)
        if j == 2
            extra_vec = -p_mat[2:Nx, 1] / dy^2
        elseif j == Ny
            extra_vec = -p_mat[2:Nx, Ny+1] / dy^2
        end
        bj = source_func(x[2:Nx], y[j])
        bj[1] -= p_mat[1, j]/dx^2
        bj[end] -= p_mat[Nx+1, j]/dx^2
        bj .+= extra_vec
        return bj
    end

    # fill in b vector
    b = zeros((Nx-1) * (Ny-1))
    for j = 2:Ny
        start_index = 1 + (j-2)*(Nx-1)
        end_index = (j-1)*(Nx-1)
        b[start_index:end_index] .= b_j(x, y, j, p_mat=p, dx=dx, dy=dy)
    end

    # solve A * p_int_vec = b, where p_int_vec contains all of the interior values of p
    p_int_vec = A \ b

    # interior values of p reshaped into a matrix. be sure to transpose it.
    p_int = reshape(p_int_vec, Ny-1, Nx-1)'  # note the ' character to take adjoint

    # fill in main matrix for p
    p[2:end-1, 2:end-1] .= p_int

    return (x, y, p)

end