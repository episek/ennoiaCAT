import matplotlib.pyplot as plt
import csv

def plot_signal_strength(filename, output_plot="signal_strength_plot.png", title="Signal Strength vs Frequency"):
    # Lists to store the data
    frequencies = []
    strengths = []
    
    # Read the CSV file
    try:
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                freq = int(float(row['Frequency']))  # Convert to int for cleaner numbers
                strength = float(row['Signal_Strength'])
                frequencies.append(freq)
                strengths.append(strength)
                
    except FileNotFoundError:
        print(f"Could not find file: {filename}")
        return
    except Exception as e:
        print(f"Error reading {filename}: {str(e)}")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, strengths, 'b.', linewidth=1, label='Signal Strength')
    
    # Customize the plot
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Signal Strength (dB)')
    plt.grid(True)
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()
    
    print(f"Plot saved as {output_plot}")




plot_signal_strength('max_signal_strengths.csv')