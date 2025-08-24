import tkinter as tk
from tkinter import filedialog
import numpy as np
from pathlib import Path
try:
    import tkinterdnd2 as tkdnd
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False
    print("tkinterdnd2 not available. Drag and drop functionality disabled.")
    print("Install it with: pip install tkinterdnd2")

try:
    import trimesh
    from trimesh.visual import ColorVisuals
except ImportError:
    raise ImportError("trimesh is required for GLTF conversion. Install it with: pip install trimesh")


class VoxelViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Voxel Model Viewer")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Create the main frame
        self.main_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create drop zone
        self.drop_zone = tk.Frame(self.main_frame, relief=tk.RAISED, borderwidth=2, bg="#e0e0e0")
        self.drop_zone.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add label to drop zone
        self.drop_label = tk.Label(
            self.drop_zone, 
            text="Drag and drop .npy files here\nor click to select files",
            font=("Arial", 14),
            bg="#e0e0e0",
            fg="#333333"
        )
        self.drop_label.pack(expand=True)
        
        # Bind events for drag and drop
        self.drop_zone.bind("<Button-1>", self.select_files)
        self.drop_label.bind("<Button-1>", self.select_files)
        
        # Enable drag and drop events if available
        if DND_AVAILABLE:
            self.root.drop_target_register(tkdnd.DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def select_files(self, event=None):
        """Open file dialog to select numpy files"""
        files = filedialog.askopenfilenames(
            title="Select Numpy Files",
            filetypes=[("Numpy files", "*.npy"), ("All files", "*.*")]
        )
        
        if files:
            self.process_files(files)

    def on_drop(self, event):
        """Handle file drop event"""
        files = self.root.tk.splitlist(event.data)
        if files:
            self.process_files(files)

    def process_files(self, files):
        """Process the selected or dropped files"""
        self.status_var.set(f"Processing {len(files)} file(s)...")
        self.root.update()
        
        # Process all files
        processed_count = 0
        for file_path in files:
            try:
                self.convert_npy_to_gltf(file_path)
                processed_count += 1
            except Exception as e:
                self.status_var.set(f"Error processing {Path(file_path).name}: {str(e)}")
                self.root.update()
                continue
            
            # Update status after each file is processed
            self.status_var.set(f"Processed {processed_count}/{len(files)} files...")
            self.root.update()
        
        self.status_var.set(f"Completed! Processed {processed_count}/{len(files)} files.")

    def convert_npy_to_gltf(self, npy_path):
        """
        Load a numpy file and convert it to GLTF format
        """
        npy_path = Path(npy_path)
        
        # Check if file exists
        if not npy_path.exists():
            raise FileNotFoundError(f"Numpy file not found: {npy_path}")
        
        # Load the numpy data
        voxel_data = np.load(npy_path)
        
        # Check dimensions
        if voxel_data.shape != (32, 32, 32):
            raise ValueError(f"Expected 32x32x32 voxel data, got {voxel_data.shape}")
        
        # Convert to GLTF
        self.save_as_gltf(voxel_data, npy_path)

    def save_as_gltf(self, voxel_data, npy_path):
        """
        Save voxel data as GLTF file
        """
        # Create a list to store all cube meshes
        meshes = []
        
        # Iterate through the 32x32x32 array
        for x in range(32):
            for y in range(32):
                for z in range(32):
                    if voxel_data[x, y, z] == 1:
                        # Create a cube at the given position
                        cube = trimesh.creation.box(extents=(1, 1, 1))
                        cube.apply_translation((x + 0.5, y + 0.5, z + 0.5))
                        
                        # Set the cube color to black
                        cube.visual.face_colors = [0, 0, 0, 255]  # RGBA black
                        
                        # Add the cube to the list of meshes
                        meshes.append(cube)
        
        if not meshes:
            raise ValueError("No voxels with value 1 found in the data")
        
        # Combine all cubes into a single mesh
        combined_mesh = trimesh.util.concatenate(meshes)
        
        # Save as GLTF
        gltf_path = npy_path.with_suffix('.gltf')
        combined_mesh.export(gltf_path)


def main():
    """Main function to run the VoxelViewer application"""
    if DND_AVAILABLE:
        root = tkdnd.Tk()
    else:
        root = tk.Tk()
    
    app = VoxelViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
