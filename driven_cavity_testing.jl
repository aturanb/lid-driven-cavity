# Driven Cavity by +e MAC Me+od
using Plots

# Initialize Grid Size
xGrid = 32
yGrid = 32

# Initialize Computational Domain
Lx = 1.0
Ly = 1.0

MaxStep = 500
Visc = 0.1 
rho = 1.0

# Initialize Parameters for SOR
MaxIt = 100
Beta = 1.2
MaxErr = 0.001

# Dimensions of the control volume in x and y directions
Δx  = Lx / xGrid
Δy  = Ly / yGrid

# Boundary conditions
un = 1 # u-north 
us = 0 # u-south
ve = 0 # v-east 
vw = 0 # v-west

# Initialize time variables
time = 0.0
Δt = 0.002

u = zeros(MaxStep, xGrid + 1, yGrid + 2)
v = zeros(MaxStep, xGrid + 2, yGrid + 1)
uu = zeros(MaxStep, xGrid + 1, yGrid + 1)
vv = zeros(MaxStep, xGrid + 1, yGrid + 1)
p = zeros(MaxStep, xGrid + 2, yGrid + 2)
ut = zeros(MaxStep, xGrid + 1, yGrid + 2)
vt = zeros(MaxStep, xGrid + 2, yGrid + 1)
pold = zeros(xGrid + 2, yGrid + 2)

# initial conditions for pressure
c = zeros(xGrid + 1, yGrid + 2) .+ 1 / (2 / Δx ^2 + 2 / Δy ^2)
c[2, 3:yGrid] .= 1 / (1 / Δx ^2 + 2 / Δy ^2)
c[xGrid+1, 3:yGrid] .= 1 / (1 / Δx ^2 + 2 / Δy ^2)
c[3:xGrid, 2] .= 1 / (1 / Δx ^2 + 2 / Δy ^2)
c[3:xGrid, yGrid+1] .= 1 / (1 / Δx ^2 + 2 / Δy ^2)
c[2, 2] = 1 / (1 / Δx ^2 + 1 / Δy ^2)
c[2, yGrid+1] = 1 / (1 / Δx ^2 + 1 / Δy ^2)
c[xGrid+1, 2] = 1 / (1 / Δx ^2 + 1 / Δy ^2)
c[xGrid+1, yGrid+1] = 1 / (1 / Δx ^2 + 1 / Δy ^2)

# grid points
x = zeros(Float64, xGrid + 1, yGrid + 1)
y = zeros(Float64, xGrid + 1, yGrid + 1)

# Fill in the grid points 
for i = 1:xGrid+1
    for j = 1:yGrid+1,
        x[i, j] = Δx  * (i - 1)
        y[i, j] = Δy  * (j - 1)
    end
end

function main()
    for is = 1:MaxStep
        @show is
        # assign boundary conditions
        u[is, 1:xGrid+1, 1] .= (2 * us .- u[is, 1:xGrid+1, 2])
        u[is, 1:xGrid+1, yGrid+2] .= (2 * un .- u[is, 1:xGrid+1, yGrid+1])
        v[is, 1, 1:yGrid+1] .= (2 * vw .- v[is, 2, 1:yGrid+1])
        v[is, xGrid+2, 1:yGrid+1] .= (2 * ve .- v[is, xGrid+1, 1:yGrid+1])

        for i = 2:xGrid # temporary u-velocity
            for j = 2:yGrid+1
                ut[is, i, j] = u[is, i, j] + Δt * (-0.25 * (
                    ((u[is, i+1, j] + u[is, i, j])^2 - (u[is, i, j] + u[is, i-1, j])^2) / Δx  + ((u[is, i, j+1] + u[is, i, j]) * (v[is, i+1, j] + v[is, i, j]) - (u[is, i, j] + u[is, i, j-1]) * (v[is, i+1, j-1] + v[is, i, j-1])) / Δy ) + Visc * ((u[is, i+1, j] + u[is, i-1, j] - 2 * u[is, i, j]) / Δx ^2 + (u[is, i, j+1] + u[is, i, j-1] - 2 * u[is, i, j]) / Δy ^2))
            end
        end

        for i = 2:xGrid+1 # temporary v-velocity
            for j = 2:yGrid
                vt[is, i, j] = v[is, i, j] + Δt * (-0.25 * (((u[is, i, j+1] + u[is, i, j]) * (v[is, i+1, j] + v[is, i, j]) - (u[is, i-1, j+1] + u[is, i-1, j]) * (v[is, i, j] + v[is, i-1, j])) / Δx  + ((v[is, i, j+1] + v[is, i, j])^2 - (v[is, i, j] + v[is, i, j-1])^2) / Δy ) + Visc * ((v[is, i+1, j] + v[is, i-1, j] - 2 * v[is, i, j]) / Δx ^2 + (v[is, i, j+1] + v[is, i, j-1] - 2 * v[is, i, j]) / Δy ^2))
            end
        end

        for it = 1:MaxIt # solve for pressure
            global pold = p[is, :, :]

            for i = 2:xGrid+1
                for j = 2:yGrid+1
                    p[is, i, j] = Beta * c[i, j] * ((p[is, i+1, j] + p[is, i-1, j]) / Δx ^2 + (p[is, i, j+1] + p[is, i, j-1]) / Δy ^2 - (rho / Δt) * ((ut[is, i, j] - ut[is, i-1, j]) / Δx  + (vt[is, i, j] - vt[is, i, j-1]) / Δy )) + (1 - Beta) * p[is, i, j]
                end
            end

            Err = 0.0 # check error
            for i = 2:xGrid+1
                for j = 2:yGrid+1,
                    Err = Err + abs(pold[i, j] - p[is, i, j])

                end
            end

            if Err <= MaxErr
                break # stop if converged
            end
        end

        # correct +e velocity
        u[is, 2:xGrid, 2:yGrid+1] = ut[is, 2:xGrid, 2:yGrid+1] - (Δt / Δx) * (p[is, 3:xGrid+1, 2:yGrid+1] - p[is, 2:xGrid, 2:yGrid+1]) #FIXME 
        v[is, 2:xGrid+1, 2:yGrid] = vt[is, 2:xGrid+1, 2:yGrid] - (Δt / Δy) * (p[is, 2:xGrid+1, 3:yGrid+1] - p[is, 2:xGrid+1, 2:yGrid])

        global time += Δt # plot +e results
        uu[is, 1:xGrid+1, 1:yGrid+1] = 0.5 * (u[is, 1:xGrid+1, 2:yGrid+2] + u[is, 1:xGrid+1, 1:yGrid+1])
        vv[is, 1:xGrid+1, 1:yGrid+1] = 0.5 * (v[is, 2:xGrid+2, 1:yGrid+1] + v[is, 1:xGrid+1, 1:yGrid+1])
    end

    anim = @animate for is = 1:MaxStep
        #Plot it
        global plt = quiver!(x, y, quiver=(uu[is, :, :], vv[is, :, :]))
        #display(plt)
    end
    gif(anim, "driven_cavity_anim_solution.gif", fps=15)
    savefig("driven cavity with UN at a tenth.png")
end

main()