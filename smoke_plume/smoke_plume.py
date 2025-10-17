"""
Se resuelve la ecuación de incompresibilidad de Navier-Stokes en 2D, en conjunto
con la ecuación de advección-difusión para simular una columna de humo.

Ecuación del momento (Navier-Stokes):
∂u/∂t + (u · ∇)u = -∇p + ν∇²u + f   (1)

Incompreibilidad:
∇ · u = 0                           (2)

Advección-difusión del humo:
∂c/∂t + (u · ∇)c = D∇²c + S           (3)

Variables:
u: campo de velocidad (u_x, u_y)
p: presión
c: concentración de humo
ν: viscosidad cinemática
D: coeficiente de difusión del humo
S: fuente de humo caliente (ubicada en la base)
f: fuerza externa (e.g., gravedad)
∇: operador nabla
∇²: operador laplaciano
t: tiempo
x, y: coordenadas espaciales

"""

from phi.jax import flow
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter # <--- ADD FFMpegWriter
from tqdm import tqdm
import numpy as np
import pickle

N_TIME_STEPS = 375
FPS = 30 
OUTPUT_DATA_FILE = "simulation_results.pkl"
OUTPUT_VIDEO_FILE = "smoke_plume_simulation.mp4"
OUTPUT_GIF_FILE = "smoke_plume_simulation.gif"

def main():
    # --- 1. Inicialización de Grids ---
    
    velocity = flow.StaggeredGrid(
        values=(0.0, 0.0),
        extrapolation=0.0,
        x=64,
        y=64,
        bounds=flow.Box(x=100, y=100),
    )
    smoke = flow.CenteredGrid(
        values=0.0,
        extrapolation=flow.extrapolation.BOUNDARY,
        x=200,
        y=200,
        bounds=flow.Box(x=100, y=100),
    )
    
    # Inflow field definition
    inflow = 0.2 * (flow.Sphere(x=40, y=9.5, radius=5) @ smoke)

    # Define la plantilla para el CenteredGrid (solo dimensiones espaciales)
    velocity_centered_template = flow.CenteredGrid(
        values=0.0,
        extrapolation=flow.extrapolation.ZERO,
        x=velocity.shape.get_size('x'),
        y=velocity.shape.get_size('y'),
        bounds=velocity.bounds,
    )

    # --- 2. Función de paso (JIT-compilada) ---
    @flow.math.jit_compile
    def step(velocity_prev, smoke_prev, inflow_field, dt=1.0, vel_struct=velocity):
        # Advección (u · ∇)u (término no lineal) y Fuente de Humo.
        smoke_next = flow.advect.mac_cormack(smoke_prev, velocity_prev, dt) + inflow_field
        
        # Difusión de humo (Término ν∇²u).
        smoke_next = flow.diffuse.explicit(smoke_next, D * dt) # Aplica la difusión

        # Fuerza de Flotación (f).
        buoyancy_force = smoke_next * (0.0, 0.1) @ vel_struct

        # Advección de Velocidad (No Lineal: (u · ∇)u) + Flotación.
        velocity_tent = flow.advect.semi_lagrangian(velocity_prev, velocity_prev, dt) + buoyancy_force * dt

        # Difusión/viscocidad (Término ν∇²u)
        velocity_tent = flow.diffuse.explicit(velocity_tent, nu * dt) # Aplica la viscosidad

        # Proyección Incompresible (Presión: -∇p y Condición: ∇ · u = 0)
        velocity_next, pressure = flow.fluid.make_incompressible(velocity_tent)
        return velocity_next, smoke_next

    # --- 3. Bucle de Simulación y Almacenamiento ---
    all_smoke_data = [] 
    all_velocity_data = [] 

    print("Iniciando simulación y recolección de datos...")
    
    current_velocity, current_smoke = velocity, smoke
    
    for _ in tqdm(range(N_TIME_STEPS)):
        current_velocity, current_smoke = step(current_velocity, current_smoke, inflow)
        
        all_smoke_data.append(current_smoke.values.numpy("y,x"))
        
        #uniform_velocity = current_velocity @ velocity_centered_template
        uniform_velocity = current_velocity @ smoke 
        all_velocity_data.append(uniform_velocity.numpy()) 

    print("\nSimulación finalizada. Guardando resultados...")

    # --- 4. Guardar Resultados de las Variables (.pkl) ---
    simulation_data = {
        "smoke_density": np.stack(all_smoke_data),
        "velocity_field": np.stack(all_velocity_data),
        "metadata": {
            "N_TIME_STEPS": N_TIME_STEPS,
            "smoke_resolution": current_smoke.shape.spatial.sizes,
            "velocity_resolution": current_velocity.shape.spatial.sizes,
        }
    }
    
    with open(OUTPUT_DATA_FILE, 'wb') as f:
        pickle.dump(simulation_data, f)
    print(f"✅ Datos de simulación guardados en: {OUTPUT_DATA_FILE}")

    # --- 5. Generar y Guardar Video (.mp4) ---
    print("Generando video...")

    # Change this path to the absolute location of your ffmpeg.exe
    FFMPEG_PATH = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'
    mpl.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH

    plt.style.use("dark_background")
    fig, ax = plt.subplots()

    im = ax.imshow(all_smoke_data[0], origin="lower", cmap='magma')
    ax.set_title("Simulación de Pluma de Humo")
    ax.axis('off')

    def update_frame(frame_index):
        im.set_array(all_smoke_data[frame_index])
        return [im]

    ani = FuncAnimation(
        fig, 
        update_frame, 
        frames=len(all_smoke_data), 
        blit=True, 
        interval=1000/FPS
    )

    writer = FFMpegWriter(fps=FPS, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(OUTPUT_VIDEO_FILE, writer=writer, dpi=100) 
    print(f"✅ Video guardado en: {OUTPUT_VIDEO_FILE}")

    print("Generando GIF...")
    #Generar un GIF alternativo usando Matplotlib (opcional)
    ani.save(OUTPUT_GIF_FILE, writer='pillow', fps=FPS, dpi=100)
    print(f"✅ Gif guardado en: {OUTPUT_GIF_FILE}")

    
    plt.close(fig)

if __name__ == "__main__":
    main()
