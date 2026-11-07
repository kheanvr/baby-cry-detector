import librosa
import numpy as np

def extract_features_from_audio(y, sr, target_sr=8000, duration=3.0, n_mels=40, target_frames=200):
    """
    Extracts Log-Mel Spectrogram + Delta + Delta-Delta features from an in-memory audio array.
    Outputs a standard 3-channel shape of (40, target_frames, 3) for CNN spatial learning.
    """
    # Resample if needed using fast algorithm
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, resample_type='kaiser_fast')
        sr = target_sr
        
    # Ensure mono
    if len(y.shape) > 1:
        y = librosa.to_mono(y)
        
    # Remove DC offset (zero-center)
    y = y - np.mean(y)
        
    # Pad or truncate raw audio to target duration
    target_samples = int(target_sr * duration)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')
    else:
        y = y[:target_samples]
        
    # Normalize audio volume (amplitude normalization)
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
        
    # Window settings (25ms window, 15ms hop)
    win_length = int(0.025 * target_sr) # 200 samples
    hop_length = int(0.015 * target_sr) # 120 samples
    n_fft = 512
    
    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=target_sr, n_mels=n_mels, n_fft=n_fft, 
                                         win_length=win_length, hop_length=hop_length)
    # Log-Mel
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # Delta and Delta-Delta
    delta = librosa.feature.delta(log_mel)
    delta2 = librosa.feature.delta(log_mel, order=2)
    
    # Cepstral Mean Subtraction (CMS) applied to Log-Mel to remove stationary room/mic response
    mean = np.mean(log_mel, axis=1, keepdims=True)
    log_mel_norm = log_mel - mean
    
    # Force exact frame dimension
    if log_mel_norm.shape[1] > target_frames:
        log_mel_norm = log_mel_norm[:, :target_frames]
        delta = delta[:, :target_frames]
        delta2 = delta2[:, :target_frames]
    elif log_mel_norm.shape[1] < target_frames:
        pad_width = target_frames - log_mel_norm.shape[1]
        log_mel_norm = np.pad(log_mel_norm, ((0, 0), (0, pad_width)), mode='constant')
        delta = np.pad(delta, ((0, 0), (0, pad_width)), mode='constant')
        delta2 = np.pad(delta2, ((0, 0), (0, pad_width)), mode='constant')
        
    # Stack along channels: shape (40, 200, 3)
    features_3ch = np.stack([log_mel_norm, delta, delta2], axis=-1)
    return features_3ch

def normalize_sample(features):
    """
    CMS is already applied inside feature extraction. Returns features as-is.
    """
    return features

def extract_features_from_file(file_path, target_sr=8000, duration=3.0, n_mfcc=40, target_frames=200):
    """
    Loads a WAV file and extracts MFCC + Delta + Delta-Delta features.
    """
    try:
        y, sr = librosa.load(file_path, sr=target_sr)
        return extract_features_from_audio(y, sr, target_sr, duration, n_mfcc, target_frames)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
