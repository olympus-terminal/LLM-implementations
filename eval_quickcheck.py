import sys

def count_strings(file1, file2):
    # Count strings in file1
    with open(file1, 'r') as f:
        file1_content = f.read()
        file1_at_count = file1_content.count('@@@')
        file1_excl_count = file1_content.count('!!!')

    # Count strings in file2
    with open(file2, 'r') as f:
        file2_content = f.read()
        file2_at_count = file2_content.count('@@@')
        file2_excl_count = file2_content.count('!!!')

    # Print the results
    print(f"sys.argv[1](algal)")
    print(f"algal tag  '@@@' count: {file1_at_count}")
    print(f"bact tag  '!!!' count: {file1_excl_count}")

    print(f"sys.argv[2](bacterial)")
    print(f"algal tag  '@@@' count: {file2_at_count}")
    print(f"bact tag  '!!!' count: {file2_excl_count}")

# Example usage
count_strings(sys.argv[1], sys.argv[2])
