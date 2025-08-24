import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import h5py
from pathlib import Path
try:
    import tkinterdnd2 as tkdnd
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False
    print("tkinterdnd2 not available. Drag and drop functionality disabled.")
    print("Install it with: pip install tkinterdnd2")

class ResizeTo32Converter:
    def __init__(self, root):
        self.root = root
            
        self.root.title("Resize to 32x32x32 Converter")
        self.root.geometry("500x300")
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
            text="Drag and drop .hdf5 files here\nor click to select files",
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
        
        # List of processed files
        self.processed_files = []

    def select_files(self, event=None):
        """Open file dialog to select HDF5 files"""
        files = filedialog.askopenfilenames(
            title="Select HDF5 Files",
            filetypes=[("HDF5 files", "*.hdf5"), ("All files", "*.*")]
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
        
        success_count = 0
        for file_path in files:
            try:
                if self.resize_to_32(file_path):
                    success_count += 1
            except Exception as e:
                messagebox.showerror("Error", f"Failed to convert {file_path}:\n{str(e)}")
        
        self.status_var.set(f"Completed: {success_count}/{len(files)} files converted")
        if success_count > 0:
            messagebox.showinfo("Success", f"Successfully converted {success_count} file(s) to 32x32x32 format")

    def resize_to_32(self, hdf5_path):
        """
        Resize an HDF5 file to 32x32x32, centering the model
        Skip models that exceed 32 in any dimension
        """
        hdf5_path = Path(hdf5_path)
        
        # Check if file exists
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        
        # Load the HDF5 data
        try:
            with h5py.File(hdf5_path, 'r') as f:
                original_blocks = f['blocks'][:]
                # Get original dimensions
                orig_height, orig_length, orig_width = original_blocks.shape
        except Exception as e:
            raise Exception(f"Failed to load HDF5 data: {str(e)}")
        
        # Check if any dimension exceeds 32
        if orig_width > 32 or orig_height > 32 or orig_length > 32:
            raise Exception(f"Model dimensions ({orig_width}x{orig_length}x{orig_height}) exceed 32 in at least one dimension. Skipping.")
        
        # Create new 32x32x32 array filled with zeros (air)
        new_blocks = np.zeros((32, 32, 32), dtype=np.uint8)
        
        # Calculate centering offsets
        offset_x = (32 - orig_width) // 2
        offset_y = (32 - orig_height) // 2
        offset_z = (32 - orig_length) // 2
        
        # Place original model in the center of the new array
        new_blocks[
            offset_y:offset_y + orig_height,
            offset_z:offset_z + orig_length,
            offset_x:offset_x + orig_width
        ] = original_blocks
        
        # Create output file path
        output_path = hdf5_path.with_name(f"{hdf5_path.stem}_32{hdf5_path.suffix}")
        
        # Save to HDF5
        try:
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('blocks', data=new_blocks, compression='gzip')
                # Store metadata
                f.attrs['width'] = 32
                f.attrs['height'] = 32
                f.attrs['length'] = 32
                f.attrs['source_file'] = hdf5_path.name
                f.attrs['original_width'] = orig_width
                f.attrs['original_height'] = orig_height
                f.attrs['original_length'] = orig_length
        except Exception as e:
            raise Exception(f"Failed to save HDF5 file: {str(e)}")
        
        return True

def main():
    if DND_AVAILABLE:
        root = tkdnd.Tk()
    else:
        root = tk.Tk()
    app = ResizeTo32Converter(root)
    root.mainloop()

if __name__ == "__main__":
    main()