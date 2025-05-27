import os

def read_numbers_from_file(file_path):
    with open(file_path, 'r') as file:
        return [float(num.strip()) for num in file.read().split(',')]

def calculate_mean(numbers):
    return round(sum(numbers) / len(numbers), 2) if numbers else 0

def subtract_values_and_calculate_mean(file1, file2, output_file):
    if os.path.exists(file1) and os.path.exists(file2):
        numbers1 = read_numbers_from_file(file1)
        numbers2 = read_numbers_from_file(file2)

        if len(numbers1) != len(numbers2):
            print("Error: Files contain different numbers of values.")
            return

        differences = [round(num1 - num2, 2) for num1, num2 in zip(numbers1, numbers2)]

        with open(output_file, 'w') as out_file:
            out_file.write(', '.join(map(str, differences)))

        mean_value = calculate_mean(differences)
        print(f"Results saved to {output_file}")
        print(f"Mean of results: {mean_value:.2f}")
    else:
        print("Error: One or both input files not found.")

# Example usage
file1 = "pi_before_noise.txt"  # Replace with actual file path
file2 = "pi_after_noise.txt"  # Replace with actual file path
output_file = "before-after_noise.txt"

subtract_values_and_calculate_mean(file1, file2, output_file)
