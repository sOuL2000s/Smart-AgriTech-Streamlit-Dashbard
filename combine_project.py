import os

def combine_files_to_single_file(root_dir, output_filename="combined_project_files.txt"):
    """
    Combines the content of specific files in a directory into a single file.

    Args:
        root_dir (str): The root directory of the project.
        output_filename (str): The name of the output file.
    """
    combined_content = []
    # Define the exact files to be included, relative to the root_dir
    # Note: This assumes index.html is in the root_dir and app.py is in the root_dir
    # If app.py can be in a subdirectory, you would need more sophisticated path matching
    # or specify the full relative path, e.g., ['src/app.py']
    included_files_only = ['index.html', 'app.py']

    print(f"Starting to combine files from: {root_dir}")
    print(f"Only including files: {included_files_only}")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # We don't need to modify dirnames in-place because we are explicitly including files,
        # not excluding directories. However, if there were very large subdirectories
        # that definitely wouldn't contain our target files (e.g., 'node_modules'),
        # we could still add an exclusion for efficiency. For this specific request,
        # it's not strictly necessary.
        pass # No modification to dirnames needed for this specific inclusion logic

        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # Make path relative to root_dir for comparison and the header
            relative_file_path = os.path.relpath(file_path, root_dir)

            # Check if the current file's name (or relative path if needed) is in our allowed list
            # For simplicity, we'll check just the filename here.
            # If 'app.py' could exist in multiple subdirectories and you only want a specific one,
            # you'd need to check `relative_file_path` more carefully (e.g., `relative_file_path == 'src/app.py'`).
            # Given the request, checking just the filename is sufficient for 'index.html' and 'app.py'
            # typically found at the root or directly by name.
            if filename in included_files_only:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    combined_content.append(f"\n--- START FILE: {relative_file_path} ---\n\n")
                    combined_content.append(content)
                    combined_content.append(f"\n\n--- END FILE: {relative_file_path} ---\n")
                    print(f"Included file: {relative_file_path}")
                except UnicodeDecodeError:
                    print(f"Skipping binary or undecodable file (UnicodeDecodeError): {relative_file_path}")
                except Exception as e:
                    print(f"Error reading file {relative_file_path}: {e}")
            else:
                print(f"Skipping file (not in included list): {relative_file_path}")


    output_path = os.path.join(root_dir, output_filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write("".join(combined_content))
        print(f"\nSuccessfully combined specified files into: {output_path}")
    except Exception as e:
        print(f"Error writing output file {output_path}: {e}")

if __name__ == "__main__":
    # Get the current working directory where the script is run
    # This assumes you run the script from your project's root directory
    project_root = os.getcwd()
    combine_files_to_single_file(project_root)
