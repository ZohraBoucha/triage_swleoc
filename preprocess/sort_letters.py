import os

def find_and_remove_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        earliest_c = {'date': None, 'filepath': None}
        
        # First pass: find the earliest C file
        for filename in filenames:
            if ' C' in filename or '-C' in filename:
                # Extract the date and type
                date_str = filename.split()[0]
                date_tuple = tuple(map(int, date_str.split('-')))

                # Update the earliest C file information
                if earliest_c['date'] is None or date_tuple < earliest_c['date']:
                    earliest_c['date'] = date_tuple
                    earliest_c['filepath'] = os.path.join(dirpath, filename)

        # Second pass: remove unwanted files
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            date_str = filename.split()[0]
            date_tuple = tuple(map(int, date_str.split('-')))

            if ' C' in filename or '-C' in filename:
                # Keep only the earliest C file
                if filepath != earliest_c['filepath']:
                    os.remove(filepath)
                    print(f"Removed C file: {filepath}")
            elif ' R' in filename or '-R' in filename:
                # Remove R files that are dated after the earliest C file
                if earliest_c['date'] and date_tuple > earliest_c['date']:
                    os.remove(filepath)
                    print(f"Removed R file: {filepath}")
        
        if earliest_c['filepath'] is None:
            print(f"Warning: No 'C' file found in directory: {dirpath}")

# Example usage
root_directory = '/home/swleocresearch/Desktop/triage-ai/datasets/cleaned_all'
find_and_remove_files(root_directory)
