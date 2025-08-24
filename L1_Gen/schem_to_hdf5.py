import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import h5py
import nbtlib
from pathlib import Path
try:
    import tkinterdnd2 as tkdnd
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False
    print("tkinterdnd2 not available. Drag and drop functionality disabled.")
    print("Install it with: pip install tkinterdnd2")

class SchematicToHDF5Converter:
    def __init__(self, root):
        self.root = root
            
        self.root.title("Schematic to HDF5 Converter")
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
            text="Drag and drop .schematic files here\nor click to select files",
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
        """Open file dialog to select schematic files"""
        files = filedialog.askopenfilenames(
            title="Select Schematic Files",
            filetypes=[("Schematic files", "*.schematic"), ("All files", "*.*")]
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
                if self.convert_schematic_to_hdf5(file_path):
                    success_count += 1
            except Exception as e:
                messagebox.showerror("Error", f"Failed to convert {file_path}:\n{str(e)}")
        
        self.status_var.set(f"Completed: {success_count}/{len(files)} files converted")
        if success_count > 0:
            messagebox.showinfo("Success", f"Successfully converted {success_count} file(s) to HDF5 format")

    def convert_schematic_to_hdf5(self, schematic_path):
        """
        Convert a Minecraft schematic file to HDF5 binary format
        Air blocks -> 0, Non-air blocks -> 1
        """
        schematic_path = Path(schematic_path)
        
        # Check if file exists
        if not schematic_path.exists():
            raise FileNotFoundError(f"Schematic file not found: {schematic_path}")
        
        # Load the NBT data
        try:
            nbt_data = nbtlib.load(schematic_path)
            schematic = nbt_data
        except Exception as e:
            raise Exception(f"Failed to load NBT data: {str(e)}")
        
        # Extract dimensions
        # Schematics Format: <File '': {'size': List[Int]([Int(11), Int(11), Int(9)]), 'entities': List([]), 'blocks': List[Compound]([Compound({'pos': List[Int]([Int(0), Int(
        try:
            width = int(schematic['size'][0])
            height = int(schematic['size'][1])
            length = int(schematic['size'][2])
        except KeyError as e:
            raise Exception(f"Missing dimension data in schematic: {str(e)}")
        
        # Extract block data
        try:
            blocks = schematic["blocks"]
            # data = schematic["data"]
        except KeyError as e:
            raise Exception(f"Missing block data in schematic: {str(e)}")
        
        # Convert to numpy array
        # Blocks is a byte array representing the block IDs
        # blocks: Compound({'pos': List[Int]([Int(0), Int(0), Int(0)]), 'state': Int(0)})
        
        # Create empty 3D array filled with air (0)
        blocks_3d = np.zeros((height, length, width), dtype=np.uint8)
        
        # Process each block in the blocks list
        for block in blocks:
            pos = block['pos']
            state = int(block['state'])
            x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
            
            # Set block: air (state=0) -> 0, non-air -> 1
            blocks_3d[y, z, x] = 0 if state == 0 else 1
        
        # Create binary array: 0 for air (block ID 0), 1 for everything else
        binary_array = blocks_3d
        
        # Create output file path
        hdf5_path = schematic_path.with_suffix('.hdf5')
        
        # Save to HDF5
        try:
            with h5py.File(hdf5_path, 'w') as f:
                f.create_dataset('blocks', data=binary_array, compression='gzip')
                # Store metadata
                f.attrs['width'] = width
                f.attrs['height'] = height
                f.attrs['length'] = length
                f.attrs['source_file'] = schematic_path.name
        except Exception as e:
            raise Exception(f"Failed to save HDF5 file: {str(e)}")
        
        return True

def main():
    if DND_AVAILABLE:
        root = tkdnd.Tk()
    else:
        root = tk.Tk()
    app = SchematicToHDF5Converter(root)
    root.mainloop()

if __name__ == "__main__":
    main()