import csv
import matplotlib.pyplot as plt

def aggregate_signal_strengths(file_list, output_filename="aggregated_signal_strengths.csv", plot_title="Aggregated Signal Strengths vs Frequency"):
    # Dictionary to store all signal strengths for each frequency
    # Key: frequency, Value: list of strengths from each file
    all_strengths = {}
    num_files = len(file_list)
    
    # Process each file
    for file_idx, filename in enumerate(file_list):
        try:
            with open(filename, 'r') as file:
                for line in file:
                    if not line.strip():
                        continue
                    
                    try:
                        freq, strength = map(float, line.split(','))
                        freq = int(freq)
                        
                        # Initialize list for this frequency if not exists
                        if freq not in all_strengths:
                            all_strengths[freq] = [None] * num_files
                        
                        # Store the strength at the appropriate file index
                        all_strengths[freq][file_idx] = strength
                            
                    except ValueError:
                        print(f"Skipping invalid line in {filename}: {line.strip()}")
                        continue
                    
        except FileNotFoundError:
            print(f"Could not find file: {filename}")
            continue
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    # Sort by frequency
    sorted_frequencies = sorted(all_strengths.keys())
    
    # Prepare CSV header
    header = ['Frequency'] + [f'File_{i+1}' for i in range(num_files)]
    
    # Write to CSV
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for freq in sorted_frequencies:
            row = [freq] + all_strengths[freq]
            writer.writerow(row)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot each file's data as a separate line
    for file_idx in range(num_files):
        strengths = [all_strengths[freq][file_idx] if all_strengths[freq][file_idx] is not None else float('nan') 
                    for freq in sorted_frequencies]
        plt.plot(sorted_frequencies, strengths, '-', label=f'File {file_idx+1}', linewidth=1)
    
    plt.title(plot_title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Signal Strength (dB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('aggregated_signal_strength_plot.png')
    plt.close()
    
    return all_strengths





file_list = []

for i in range(10):
    filename = f"output_{i + 1}.csv"
    file_list.append(filename)

file_list.append('max_signal_strengths.csv')

result = aggregate_signal_strengths(file_list)



