import sys

def clean_file(input_path):
    print(f"Propcessing {input_path}...")
    try:
        # rockyou is typically latin-1
        with open(input_path, 'r', encoding='latin-1', errors='ignore') as f:
            lines = f.readlines()
            
        cleaned_lines = []
        for line in lines:
            # strip newline for processing
            s = line.strip()
            # check if all chars are ascii printable (or we can just filter non-ascii)
            # User said "eng and symbols", so basically ASCII.
            try:
                # Try encoding to ascii, if it fails, it has non-ascii
                s.encode('ascii')
                cleaned_lines.append(line)
            except UnicodeEncodeError:
                continue

        with open(input_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
            
        print(f"Cleaned {input_path}. Kept {len(cleaned_lines)} out of {len(lines)} lines.")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    clean_file("rockyou.txt")
