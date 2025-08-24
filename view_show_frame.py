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
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not available. Image saving functionality disabled.")
    print("Install it with: pip install matplotlib")

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
        
        # 3D visualization
        self.voxel_data = None
        self.figure = None
        self.canvas = None
        self.ax = None
        
        # Rotation variables
        self.rotate_axis = 'y'
        self.rotation_angle = 0

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
        if not MATPLOTLIB_AVAILABLE:
            self.status_var.set("Error: matplotlib not installed")
            return
            
        self.status_var.set(f"Processing {len(files)} file(s)...")
        self.root.update()
        
        # Process all files
        processed_count = 0
        for file_path in files:
            try:
                self.load_and_save_voxel_image(file_path)
                processed_count += 1
                self.status_var.set(f"Processed {processed_count}/{len(files)} files...")
                self.root.update()
            except Exception as e:
                self.status_var.set(f"Error processing {Path(file_path).name}: {str(e)}")
                self.root.update()
                continue
        
        self.status_var.set(f"Completed! Processed {processed_count}/{len(files)} files.")

    def load_and_save_voxel_image(self, npy_path):
        """
        Load a numpy file and save its 3D voxel representation as an image file
        """
        npy_path = Path(npy_path)
        
        # Check if file exists
        if not npy_path.exists():
            raise FileNotFoundError(f"Numpy file not found: {npy_path}")
        
        # Load the numpy data
        self.voxel_data = np.load(npy_path)
        
        # Check dimensions
        if self.voxel_data.shape != (32, 32, 32):
            raise ValueError(f"Expected 32x32x32 voxel data, got {self.voxel_data.shape}")
        
        # Save the voxel model as an image
        self.save_voxel_image(npy_path)

    def save_voxel_image(self, npy_path):
        """
        Save the voxel model as an image file in the same directory
        """
        # Create figure
        self.figure = plt.figure(figsize=(8, 6))
        self.ax = self.figure.add_subplot(111, projection='3d')
        
        # Get voxel coordinates where value is 1
        voxels = self.voxel_data.astype(bool)
        
        # Create voxel visualization
        self.ax.voxels(voxels, facecolors='blue', edgecolors='black', alpha=0.7)
        
        # Set labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'3D Voxel Model: {npy_path.name}')
        
        # Save image to the same directory with .png extension
        image_path = npy_path.with_suffix('.png')
        self.figure.savefig(image_path, dpi=150, bbox_inches='tight')
        
        # Close the figure to free memory
        plt.close(self.figure)
        
        self.figure = None
        self.ax = None





def main():
    if DND_AVAILABLE:
        root = tkdnd.Tk()
    else:
        root = tk.Tk()
        
    if not MATPLOTLIB_AVAILABLE:
        tk.Label(root, text="Error: matplotlib is required for image saving.\nInstall it with: pip install matplotlib", 
                 fg="red", font=("Arial", 12)).pack(pady=50)
    else:
        app = VoxelViewer(root)
        
    root.mainloop()

if __name__ == "__main__":
    main()
