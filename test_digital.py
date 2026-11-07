import os
import sys
import numpy as np
import preprocess

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("Error: Could not import tflite_runtime or tensorflow.")
        sys.exit(1)

MODEL_TFLITE_PATH = "baby_cry_model.tflite"
SCALER_PATH = "scaler_stats.npz"
CLASSES = ["discomfort", "hunger", "other"]

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_digital.py <path_to_wav_file>")
        sys.exit(1)
        
    wav_path = sys.argv[1]
    if not os.path.exists(wav_path):
        print(f"Error: WAV file not found at '{wav_path}'")
        sys.exit(1)
        
    if not os.path.exists(MODEL_TFLITE_PATH) or not os.path.exists(SCALER_PATH):
        print(f"Error: Missing '{MODEL_TFLITE_PATH}' or '{SCALER_PATH}' in current directory.")
        sys.exit(1)

    print(f"Loading '{wav_path}'...")
    import librosa
    y, sr = librosa.load(wav_path, sr=None)
    print(f"Original Sample Rate: {sr} Hz, Duration: {len(y)/sr:.2f}s")

    # Extract features
    features = preprocess.extract_features_from_audio(y, sr=sr)
    
    # Scale features using base-MFCC CMS
    features_scaled = preprocess.normalize_sample(features)
    
    # Shape for TFLite (1, 120, 200, 1)
    input_tensor = np.expand_dims(features_scaled, axis=(0, -1)).astype(np.float32)
    
    # Run interpreter
    interpreter = tflite.Interpreter(model_path=MODEL_TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]['index'])[0]
    
    print("\n--- Model Prediction Probabilities ---")
    for idx, cls in enumerate(CLASSES):
        print(f"{cls.capitalize():<12}: {probs[idx]*100:6.2f}%")
    
    max_idx = np.argmax(probs)
    print(f"\nResult: Predicted as '{CLASSES[max_idx]}' (Confidence: {probs[max_idx]*100:.2f}%)")

if __name__ == "__main__":
    main()
