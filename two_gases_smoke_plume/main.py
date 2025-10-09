from phi.jax import flow
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter 
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
    
    # --- MODIFICACIÓN CLAVE: Definición de la Densidad de Fondo (Dos Gases) ---
    # Usamos un Grid para definir la densidad de cada gas
    
    # 1. Definir la densidad del Gas 1 (ej. densidad = 1.0)
    DENSIDAD_GAS_1 = 1.0
    # 2. Definir la densidad del Gas 2 (ej. densidad = 0.5, más ligero que el gas 1)
    DENSIDAD_GAS_2 = 0.5
    
    # Grid de la densidad de fondo, con la misma resolución que 'smoke'
    background_density = flow.CenteredGrid(
        values=DENSIDAD_GAS_1, # Inicialmente todo Gas 1
        extrapolation=flow.extrapolation.ZERO,
        x=200,
        y=200,
        bounds=flow.Box(x=100, y=100),
    )

    
    # Creamos una máscara para la mitad derecha (x > 50)
    # Asumimos que el dominio 'x' va de 0 a 100
    half_domain_mask = flow.math.where(background_density.center.vector['x'] > 50.0, 1.0, 0.0)

    # Asignamos la densidad del Gas 2 a la mitad derecha del dominio
    # La densidad en la izquierda es DENSIDAD_GAS_1 * (1 - mask) = DENSIDAD_GAS_1
    # La densidad en la derecha es DENSIDAD_GAS_2 * mask = DENSIDAD_GAS_2
    background_density = (background_density * (1.0 - half_domain_mask)) + (DENSIDAD_GAS_2 * half_domain_mask)

    # El humo inicial tiene una densidad (ej. 0.8) que es intermedia para una flotabilidad inicial
    DENSIDAD_INICIAL_HUMO = 0.8
    inflow_density = 0.2 * DENSIDAD_INICIAL_HUMO 
    
    # Inflow field definition
    inflow = inflow_density * (flow.Sphere(x=40, y=9.5, radius=5) @ smoke) # El inflow está en la región del Gas 1 (x=40)

    # ... (El resto del código de inicialización puede omitirse para simplificar la respuesta,
    #     ya que no afecta la lógica de simulación, solo la visualización)

    # --- 2. Función de paso (JIT-compilada) ---
    # Se añade background_density como argumento
    @flow.math.jit_compile
    def step(velocity_prev, smoke_prev, inflow_field, background_density_grid, dt=1.0, vel_struct=velocity):
        
        # Flujo de Humo
        smoke_next = flow.advect.mac_cormack(smoke_prev, velocity_prev, dt) + inflow_field
        
        # --- MODIFICACIÓN CLAVE: Flotabilidad Basada en Diferencia de Densidad ---
        # La fuerza es proporcional a la diferencia de densidad: (Densidad Fondo - Densidad Humo) * Gravedad
        # Para simplificar, la "densidad" del humo es su valor en el grid 'smoke_next'
        GRAVITY = 0.1 # Factor de gravedad (ajustado para la simulación)
        
        # Calcula la diferencia de densidad. El humo se define con una densidad de fondo de 0.8 
        # (Esto es una simplificación, en realidad, el valor del grid 'smoke' representa una
        # concentración que se convierte a densidad: rho_humo = rho_gas + smoke_value * (rho_inflow - rho_gas))
        # Asumiremos para la simplificación que 'smoke_next' es la densidad absoluta del fluido.
        # Es decir, la densidad del humo es (smoke_next) y la densidad de fondo es (background_density_grid)
        
        density_difference = background_density_grid - smoke_next 
        
        # La fuerza de flotabilidad solo actúa en el eje Y (vertical)
        # Multiplicamos por GRAVITY y la proyectamos al StaggeredGrid de velocidad
        buoyancy_force = density_difference * (0.0, GRAVITY) @ vel_struct
        
        velocity_tent = flow.advect.semi_lagrangian(velocity_prev, velocity_prev, dt) + buoyancy_force * dt
        velocity_next, pressure = flow.fluid.make_incompressible(velocity_tent)
        
        return velocity_next, smoke_next

    # --- 3. Bucle de Simulación y Almacenamiento ---
    all_smoke_data = [] 
    all_velocity_data = [] 

    print("Iniciando simulación y recolección de datos...")
    
    current_velocity, current_smoke = velocity, smoke
    
    # background_density es constante, no necesita cambiar
    
    for _ in tqdm(range(N_TIME_STEPS)):
        # Pasamos el background_density a la función step
        current_velocity, current_smoke = step(current_velocity, current_smoke, inflow, background_density)
        
        all_smoke_data.append(current_smoke.values.numpy("y,x"))
        
        uniform_velocity = current_velocity @ smoke 
        all_velocity_data.append(uniform_velocity.numpy()) 

    # ... (El resto del código de guardado y visualización sigue igual)
    print("\nSimulación finalizada. Guardando resultados...")

    # --- 4. Guardar Resultados de las Variables (.pkl) ---
    # Incluir la densidad de fondo en los datos guardados para referencia
    simulation_data = {
        "smoke_density": np.stack(all_smoke_data),
        "velocity_field": np.stack(all_velocity_data),
        "background_density": background_density.values.numpy('y,x'), # Añadido
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

    # Opcional: Visualizar la línea divisoria
    ax.axvline(x=100, color='lime', linestyle='--', linewidth=1, alpha=0.5) # x=100 es la mitad de 200 pixeles
    
    im = ax.imshow(all_smoke_data[0], origin="lower", cmap='magma')
    ax.set_title("Simulación de Pluma de Humo con 2 Gases")
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
