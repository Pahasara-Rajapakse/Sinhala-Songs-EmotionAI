import numpy as np
import librosa

SR = 44100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 1024
TARGET_FRAMES = 431

def extract_logmel(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel.astype(np.float32)

def prepare_input(mel):
    if mel.shape[1] < TARGET_FRAMES:
        mel = np.pad(mel, ((0,0),(0,TARGET_FRAMES - mel.shape[1])), 'constant')
    else:
        mel = mel[:, :TARGET_FRAMES]

    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    x = np.expand_dims(mel, axis=-1)
    x = np.repeat(x, 3, axis=-1)
    return np.expand_dims(x, axis=0)
