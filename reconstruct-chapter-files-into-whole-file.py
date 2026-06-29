from datetime import datetime
from pathlib import Path

# 1. Setup paths (./chapters points to the local subfolder)
input_folder = Path("./chapters")
output_folder = Path("./full-draft")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
reconstructed_filename = Path(f"reconstructed-whole-file_{timestamp}.txt")
reconstructed_fullpath = output_folder / reconstructed_filename

# 2. Gather all text files and sort them numerically by their filename integer
chunk_files = sorted(
    input_folder.glob("*.txt"), 
    key=lambda p: int(p.stem)
)

# 3. Combine them sequentially into the timestamped output file
with open(reconstructed_fullpath, "w", encoding="utf-8") as master_file:
    for chunk_path in chunk_files:
        with open(chunk_path, "r", encoding="utf-8") as chunk:
            for line in chunk:
                master_file.write(line)

print(f"Successfully combined {len(chunk_files)} chapters into: {reconstructed_fullpath}")

