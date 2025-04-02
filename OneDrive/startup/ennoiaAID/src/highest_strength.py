import csv

# Version for multiple files
def find_max_signal_strength_to_csv(file_list, output_filename="max_signal_strengths.csv", min_strength=-80):
    max_strengths = {}
    
    for filename in file_list:
        try:
            with open(filename, 'r') as file:
                for line in file:
                    if not line.strip():
                        continue
                    
                    try:
                        freq, strength = map(float, line.split(','))
                        freq = int(freq)
                        # Only consider strengths >= min_strength
                        if strength >= min_strength:
                            if freq not in max_strengths or strength > max_strengths[freq]:
                                max_strengths[freq] = strength
                            
                    except ValueError:
                        print(f"Skipping invalid line in {filename}: {line.strip()}")
                        continue
                    
        except FileNotFoundError:
            print(f"Could not find file: {filename}")
            continue
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    # Sort and save to CSV
    sorted_results = sorted(max_strengths.items())
    
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frequency', 'Signal_Strength'])  # Header
        writer.writerows(sorted_results)
    
    return dict(sorted_results)

# Version for single data string
def process_single_data_to_csv(data_string, output_filename="max_signal_strengths.csv", min_strength=-80):
    max_strengths = {}
    
    lines = data_string.strip().split('\n')
    lines = lines[1:]  # Skip header
    for line in lines:
        if not line.strip():
            continue
            
        try:
            freq, strength = map(float, line.split(','))
            freq = int(freq)
            # Only consider strengths >= min_strength
            if strength >= min_strength:
                if freq not in max_strengths or strength > max_strengths[freq]:
                    max_strengths[freq] = strength
                
        except ValueError:
            print(f"Skipping invalid line: {line.strip()}")
            continue
    
    # Sort and save to CSV
    sorted_results = sorted(max_strengths.items())
    
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frequency', 'Signal_Strength'])  # Header
        writer.writerows(sorted_results)
    
    return dict(sorted_results)

# Your existing code with the file list
file_list = []
for i in range(10):
    filename = f"output_{i + 1}.csv"
    file_list.append(filename)

result = find_max_signal_strength_to_csv(file_list)
print(result)