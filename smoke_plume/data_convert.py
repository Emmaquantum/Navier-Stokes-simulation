import numpy as np
import pickle
import os

OUTPUT_DATA_FILE = "simulation_results.pkl"

def convert_pkl_to_npy(pkl_file):
    """Carga los datos del .pkl y los guarda como archivos .npy separados."""
    
    # 1. Cargar el archivo .pkl
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ Error al cargar {pkl_file}: {e}")
        return

    # Extraer los arrays de interés
    smoke_data = data["smoke_density"]
    velocity_data = data["velocity_field"]
    
    # 2. Definir los nombres de los archivos .npy
    smoke_npy_file = "smoke_density_data.npy"
    velocity_npy_file = "velocity_field_data.npy"
    
    # 3. Guardar los arrays
    np.save(smoke_npy_file, smoke_data)
    np.save(velocity_npy_file, velocity_data)
    
    print(f"✅ Datos de humo guardados en: {smoke_npy_file}")
    print(f"✅ Datos de velocidad guardados en: {velocity_npy_file}")
    print("\n¡Los archivos .npy están listos para ser cargados en TensorFlow/PyTorch!")

if __name__ == "__main__":
    convert_pkl_to_npy(OUTPUT_DATA_FILE)