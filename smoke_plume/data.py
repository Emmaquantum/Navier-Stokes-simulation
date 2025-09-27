import pickle
import numpy as np
import pandas as pd

OUTPUT_DATA_FILE = "simulation_results.pkl"

def create_pinn_dataframe(pkl_file_path):
    # ... (Cargar el archivo .pkl) ...
    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ ERROR al cargar el archivo: {e}")
        return None

    smoke = data["smoke_density"]
    velocity = data["velocity_field"]
    metadata = data["metadata"]
    
    # Aseguramos que la resolución se toma del array más confiable (el humo 200x200)
    # y la longitud de tiempo se toma de ambos.
    num_timesteps, res_y, res_x = smoke.shape 

    # --- 2. Generar Coordenadas (Usando las Resoluciones del Humo) ---
    x_min, x_max = 0, 100
    y_min, y_max = 0, 100

    # Generar valores de coordenadas
    X_values = np.linspace(x_min, x_max, res_x, endpoint=False)
    X_values += (X_values[1] - X_values[0]) / 2 
    Y_values = np.linspace(y_min, y_max, res_y, endpoint=False) 
    Y_values += (Y_values[1] - Y_values[0]) / 2 
    T_values = np.arange(num_timesteps) * 1.0 
    
    # --- 3. Crear Malla (Meshgrid) y Aplanar ---
    
    T_grid, Y_grid, X_grid = np.meshgrid(T_values, Y_values, X_values, indexing='ij')

    T_flat = T_grid.flatten()
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()

    # CRITICAL: If the dimensions were different during saving, the flattening fails.
    # We must trust the final dimensions of the smoke data for the total array length.
    
    Smoke_flat = smoke.flatten()
    
    # Re-shape the U and V components to match the exact length of the T/X/Y grids (T*Y*X)
    # This requires assuming the velocity data was saved correctly on the same grid size as smoke.
    U_flat = velocity[:, :, :, 0].flatten() 
    V_flat = velocity[:, :, :, 1].flatten() 
    
    # Final check before DataFrame creation
    if T_flat.size != U_flat.size:
        print(f"🚨 ERROR FATAL: El número de puntos en las coordenadas ({T_flat.size}) no coincide con el número de puntos en los datos U ({U_flat.size}).")
        raise ValueError("El guardado de datos en el archivo .pkl es inconsistente en su resolución.")

    # --- 4. Crear el DataFrame ---
    data_frame = pd.DataFrame({
        't': T_flat,
        'x': X_flat,
        'y': Y_flat,
        'u': U_flat,
        'v': V_flat,
        'density': Smoke_flat
    })

    return data_frame

if __name__ == "__main__":
    df = create_pinn_dataframe(OUTPUT_DATA_FILE)

    if df is not None:
        print("\n--- DataFrame PINN Creado Exitosamente ---")
        print(f"Forma del DataFrame: {df.shape}")
        print(df.head())
        
        # Save to CSV for easy use in PINN
        df.to_csv("simulation_data_pinn.csv", index=False)
        print("✅ DataFrame guardado como simulation_data_pinn.csv")