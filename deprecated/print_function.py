#!/usr/bin/env python
"""Script to print the list_futures_contracts function from view_futures_contracts.py"""

def main():
    file_path = 'src/scripts/market_data/view_futures_contracts.py'
    functions_to_find = ['def list_futures_contracts', 'def get_futures_contracts']
    
    with open(file_path, 'r') as file:
        content = file.read()
        lines = content.splitlines()
    
    print(f"Total lines in file: {len(lines)}")
    
    for target_function in functions_to_find:
        print(f"\nSearching for function: {target_function}")
        
        # Find the function
        start_line = -1
        for i, line in enumerate(lines):
            if target_function in line:
                start_line = i
                print(f"Found function at line {i+1}")
                break
        
        if start_line == -1:
            print(f"Function '{target_function}' not found in {file_path}")
            continue
            
        # Look at the full function
        function_name = target_function.split()[1].split('(')[0]
        print(f"Analyzing function: {function_name}")
        
        # Check for any code related to table creation or printing
        table_creation = False
        missing_return = True
        for i in range(start_line, len(lines)):
            if 'table = Table(' in lines[i]:
                table_creation = True
                print(f"Table creation found at line {i+1}: {lines[i].strip()}")
            if 'console.print(' in lines[i] and 'table' in lines[i]:
                print(f"Table printing found at line {i+1}: {lines[i].strip()}")
            if 'return ' in lines[i] and function_name != 'get_futures_contracts':
                missing_return = False
                print(f"Return statement found at line {i+1}: {lines[i].strip()}")
    
    # Now let's specifically look for list_futures_contracts and find where it ends
    function_to_analyze = 'list_futures_contracts'
    print(f"\n\nAnalyzing function: {function_to_analyze}")
    
    # Find the function
    start_line = -1
    for i, line in enumerate(lines):
        if f"def {function_to_analyze}" in line:
            start_line = i
            break
    
    if start_line == -1:
        print(f"Function '{function_to_analyze}' not found in {file_path}")
        return
    
    # Print what comes after the comment about table creation
    in_function = True
    for i in range(start_line, len(lines)):
        if lines[i].strip() == '# ... (table creation and printing) ...':
            print(f"Found comment about table creation at line {i+1}")
            print("\nCode after this comment:")
            
            # Print the next few lines to see what's actually there
            for j in range(i+1, min(i+15, len(lines))):
                print(f"{j+1}: {lines[j]}")
                
            # Check if there's any implementation after this comment
            implementation_found = False
            for j in range(i+1, min(i+15, len(lines))):
                if lines[j].strip() and not lines[j].strip().startswith('#'):
                    implementation_found = True
                    break
            
            if not implementation_found:
                print("\nWARNING: No implementation found after this comment!")
            
            break
    
    # Print the entire file content to a text file for inspection
    with open('view_futures_contracts_full.txt', 'w') as f:
        f.write(content)
    print("\nWrote full file content to view_futures_contracts_full.txt for inspection")

if __name__ == "__main__":
    main() 