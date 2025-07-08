import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Parameters ---
N_POINTS = 129
DOMAIN_SIZE = 1.0
N_ITERATIONS = 20000
TIME_STEP_LENGTH = 0.001
HORIZONTAL_VELOCITY_TOP = 1.0
DENSITY = 1.0
Re = 400
KINEMATIC_VISCOSITY = 1.0 / Re
N_PRESSURE_POISSON_ITERATIONS = 200
STABILITY_SAFETY_FACTOR = 0.25

def simulate_lid_driven_cavity():
    dx = DOMAIN_SIZE / (N_POINTS - 1)
    x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    X, Y = np.meshgrid(x, y, indexing='xy')

    # Initialize fields
    u = np.zeros_like(X)
    v = np.zeros_like(X)
    p = np.zeros_like(X)

    # Precompute coefficients
    inv_dx = 1.0 / dx
    inv_dx2 = inv_dx * inv_dx
    dt_nu = TIME_STEP_LENGTH * KINEMATIC_VISCOSITY
    dt_rho = TIME_STEP_LENGTH / DENSITY

    # Vectorized derivative functions
    def central_diff_x(f):
        diff = np.zeros_like(f)
        diff[:, 1:-1] = (f[:, 2:] - f[:, :-2]) * 0.5 * inv_dx
        return diff

    def central_diff_y(f):
        diff = np.zeros_like(f)
        diff[1:-1, :] = (f[2:, :] - f[:-2, :]) * 0.5 * inv_dx
        return diff

    def laplacian(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (f[1:-1, :-2] + f[:-2, 1:-1] +
                             f[1:-1, 2:] + f[2:, 1:-1] - 4 * f[1:-1, 1:-1]) * inv_dx2
        return diff

    # Main simulation loop
    for it in tqdm(range(N_ITERATIONS), desc="Simulating"):
        u_old = u.copy()
        v_old = v.copy()

        # Predictor step (vectorized)
        conv_u = u * central_diff_x(u) + v * central_diff_y(u)
        diff_u = laplacian(u)
        u_star = u + TIME_STEP_LENGTH * (-conv_u + KINEMATIC_VISCOSITY * diff_u)

        conv_v = u * central_diff_x(v) + v * central_diff_y(v)
        diff_v = laplacian(v)
        v_star = v + TIME_STEP_LENGTH * (-conv_v + KINEMATIC_VISCOSITY * diff_v)

        # Apply boundary conditions
        u_star[0, :] = 0.0                     # Bottom wall
        u_star[-1, :] = HORIZONTAL_VELOCITY_TOP  # Top wall
        u_star[:, [0, -1]] = 0.0                # Left/right walls
        v_star[[0, -1], :] = 0.0                # Bottom/top walls
        v_star[:, [0, -1]] = 0.0                # Left/right walls

        # Fix corner values
        u_star[0, 0] = 0; u_star[0, -1] = 0
        u_star[-1, 0] = 0; u_star[-1, -1] = 0
        v_star[0, 0] = 0; v_star[0, -1] = 0
        v_star[-1, 0] = 0; v_star[-1, -1] = 0

        # Pressure Poisson equation
        div_u = central_diff_x(u_star)
        div_v = central_diff_y(v_star)
        rhs = (div_u + div_v) / TIME_STEP_LENGTH

        # Solve with Jacobi method
        p_prev = p.copy()
        for _ in range(N_PRESSURE_POISSON_ITERATIONS):
            p_new = p.copy()
            p_new[1:-1, 1:-1] = 0.25 * (
                p_prev[1:-1, :-2] + p_prev[1:-1, 2:] +
                p_prev[:-2, 1:-1] + p_prev[2:, 1:-1] -
                dx**2 * rhs[1:-1, 1:-1]
            )
            # Pressure BCs (Neumann)
            p_new[:, 0] = p_new[:, 1]           # Left
            p_new[:, -1] = p_new[:, -2]          # Right
            p_new[0, :] = p_new[1, :]            # Bottom
            p_new[-1, :] = p_new[-2, :]          # Top
            p_new[0, 0] = 0.0                    # Reference point

            # Under-relaxation
            p = p_prev + 0.5 * (p_new - p_prev)
            p_prev = p.copy()

        # Corrector step
        dp_dx = central_diff_x(p)
        dp_dy = central_diff_y(p)
        u = u_star - dt_rho * dp_dx
        v = v_star - dt_rho * dp_dy

        # Reapply velocity BCs
        u[0, :] = 0.0; u[-1, :] = HORIZONTAL_VELOCITY_TOP
        u[:, [0, -1]] = 0.0
        v[[0, -1], :] = 0.0; v[:, [0, -1]] = 0.0
        u[0, 0] = 0; u[0, -1] = 0; u[-1, 0] = 0; u[-1, -1] = 0
        v[0, 0] = 0; v[0, -1] = 0; v[-1, 0] = 0; v[-1, -1] = 0

        # Convergence check
        if it % 1000 == 0:
            du = np.max(np.abs(u - u_old))
            dv = np.max(np.abs(v - v_old))
            if du < 1e-6 and dv < 1e-6:
                print(f"Converged after {it} iterations")
                break

    return u, v, p, X, Y, dx

def compute_vorticity(u, v, dx):
    """Vorticity calculation with boundary corrections for Re=400"""
    dv_dx = np.zeros_like(v)
    du_dy = np.zeros_like(u)

    # Central differences for interior points
    dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
    du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)

    # Second-order differences for boundaries
    # Left boundary
    dv_dx[:, 0] = (-3*v[:, 0] + 4*v[:, 1] - v[:, 2]) / (2 * dx)
    # Right boundary
    dv_dx[:, -1] = (3*v[:, -1] - 4*v[:, -2] + v[:, -3]) / (2 * dx)
    # Bottom boundary
    du_dy[0, :] = (-3*u[0, :] + 4*u[1, :] - u[2, :]) / (2 * dx)
    # Top boundary
    du_dy[-1, :] = (3*u[-1, :] - 4*u[-2, :] + u[-3, :]) / (2 * dx)

    return dv_dx - du_dy

def plot_vorticity(omega, X, Y):
    plt.figure(figsize=(10, 8))

    # Vorticity levels for Re=400 as specified
    levels = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

    # Create contour plot with specified line styles
    cs = plt.contour(X, Y, omega, levels=levels, colors='k', linewidths=1.0)

    # Customize line styles: dashed for negative, solid for positive, thick for zero
    for i, collection in enumerate(cs.collections):
        level = cs.levels[i]
        if level < 0:
            plt.setp(collection, linestyle='dashed')
        elif level == 0:
            plt.setp(collection, linewidth=2.0)

    # Label all levels
    plt.clabel(cs, levels=levels, inline=True, fontsize=10, fmt='%1.0f')

    # Formatting
    plt.gca().set_aspect('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.linspace(0, 1, 6))
    plt.yticks(np.linspace(0, 1, 6))
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.title(f'Vorticity Contours (Re={Re})', fontsize=16)
    plt.xlabel('X/D', fontsize=14)
    plt.ylabel('Y/D', fontsize=14)

    # Add secondary title with simulation details
    plt.figtext(0.5, 0.01, f'Grid: {N_POINTS}x{N_POINTS}, Time step: {TIME_STEP_LENGTH:.1e}',
                ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'vorticity_Re{Re}.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print(f"Starting Re={Re} simulation with {N_POINTS}x{N_POINTS} grid...")
    u, v, p, X, Y, dx = simulate_lid_driven_cavity()

    print("\nComputing vorticity...")
    omega = compute_vorticity(u, v, dx)
    print(f"Vorticity range: min={np.min(omega):.2f}, max={np.max(omega):.2f}")

    print("\nPlotting results...")
    plot_vorticity(omega, X, Y)
    print(f"Done! Results saved as vorticity_Re{Re}.png")
