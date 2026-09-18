import joblib
import os

input_file = r"C:\Users\KGCAS\Downloads\UrbanPulseAI\models\traffic_model.pkl"
output_file = r"C:\Users\KGCAS\Downloads\UrbanPulseAI\models\traffic_model_compressed.pkl"

print("Loading original model...")

model_package = joblib.load(input_file)

print("Original model loaded successfully.")

print("Saving compressed model...")

joblib.dump(
    model_package,
    output_file,
    compress=("gzip", 9)
)

original_size = os.path.getsize(input_file)
compressed_size = os.path.getsize(output_file)

print()
print("Compression completed successfully!")
print(f"Original size   : {original_size / (1024 * 1024):.2f} MB")
print(f"Compressed size : {compressed_size / (1024 * 1024):.2f} MB")
print(f"Output file     : {output_file}")