from pathlib import Path

def split_large_file(input_filename):
    output_folder_path = Path("./chapters")
    input_folder_path = Path("./initial-source-files")    

    input_full_path = input_folder_path / input_filename
    current_output_file = None
    
    with open(input_full_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            stripped_line = line.strip()
            
            # Identify lines containing only whole numbers
            if stripped_line.isdigit():
                if current_output_file:
                    current_output_file.close()
                
                # Create the new file using the number as the name
                output_filename = f"{stripped_line}.txt"
                output_full_path = output_folder_path / output_filename
                current_output_file = open(output_full_path, 'w', encoding='utf-8')
                
                # Write the trigger number line into the new file
                current_output_file.write(line)
                continue
            
            # Write content only if a valid number file has already been opened
            if current_output_file:
                current_output_file.write(line)
                
        if current_output_file:
            current_output_file.close()

# Run the script
split_large_file('whole-file_first-edition.txt')
