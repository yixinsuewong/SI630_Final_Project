import os

# The specific question file path
question_file = "/Users/yixin/Desktop/SI 630/Project_RAG/quality_estimation/quality_question.txt"

print(f"Checking file: {question_file}")

# Check if file exists
if not os.path.exists(question_file):
    print("ERROR: File does not exist!")
else:
    # Get file size
    file_size = os.path.getsize(question_file)
    print(f"File size: {file_size} bytes")
    
    if file_size == 0:
        print("File is completely empty (0 bytes)")
    else:
        # Read raw binary data
        with open(question_file, 'rb') as f:
            raw_data = f.read()
        
        # Print hexadecimal representation to see any hidden characters
        print("First 100 bytes in hex:")
        print(' '.join(f'{b:02x}' for b in raw_data[:100]))
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'ascii']
        for encoding in encodings:
            try:
                content = raw_data.decode(encoding)
                print(f"\nSuccessfully decoded with {encoding}")
                
                # Print content representation
                if len(content) > 100:
                    print(f"Content (first 100 chars): '{content[:100]}...'")
                else:
                    print(f"Content: '{content}'")
                
                # Count newlines
                newlines = content.count('\n')
                print(f"Number of newlines: {newlines}")
                
                # Split into lines and check for non-empty lines
                lines = content.splitlines()
                non_empty_lines = [line for line in lines if line.strip()]
                print(f"Total lines: {len(lines)}")
                print(f"Non-empty lines: {len(non_empty_lines)}")
                
                if non_empty_lines:
                    print("\nFirst few non-empty lines:")
                    for i, line in enumerate(non_empty_lines[:3]):
                        print(f"  Line {i+1}: '{line[:50]}{'...' if len(line) > 50 else ''}'")
                else:
                    print("\nNo non-empty lines found in the file")
                
                break
            except UnicodeDecodeError:
                print(f"Failed to decode with {encoding}")
                continue