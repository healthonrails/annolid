import os
from skimage import io
import numpy as np

def show_and_count_cells(output_path, img_name="img2_label.tiff", cmap="cividis"):
    pred_path = os.path.join(output_path, img_name)
    
    try:
        pred = io.imread(pred_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    if pred is None:
        print("Image is empty or could not be loaded.")
        return
    
    try:
        io.imshow(pred, cmap=cmap)
        io.show()
    except Exception as e:
        print(f"Error displaying image: {e}")
    
    cell_count = len(np.unique(pred)) - 1  # Exclude the background
    print(f"\n{cell_count} Cells detected!")

# Example usage:
output_path = "./results/mediar_base_prediction"
show_and_count_cells(output_path,'OpenTest_004_label.tiff')