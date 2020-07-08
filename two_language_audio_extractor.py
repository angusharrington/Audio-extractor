#%%

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import librosa

import librosa.display



def two_language_audio_extractor(sound_path_1, sound_path_2, extracted_1, extracted_2):
    y1, sr = librosa.load(sound_path_1)
    y2, sr = librosa.load(sound_path_2)


    S_full1, phase1 = librosa.magphase(librosa.stft(y1))

    S_full2, phase2 = librosa.magphase(librosa.stft(y2))



    idx = slice(*librosa.time_to_frames([0, 5], sr=sr))
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(librosa.amplitude_to_db(S_full1[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
    plt.colorbar()
    plt.tight_layout()

    idx = slice(*librosa.time_to_frames([0, 5], sr=sr))
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(librosa.amplitude_to_db(S_full2[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
    plt.colorbar()
    plt.tight_layout()
    S1 = S_full1 - S_full2
    S2 = S_full2 - S_full1

    for i in range (S1.shape[0]):
        for j in range (S1.shape[1]):
            if S1[i][j] < 0:
                S1[i][j] = 0
            if S2[i][j] < 0:
                S2[i][j] = 0


    idx = slice(*librosa.time_to_frames([0, 5], sr=sr))
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(librosa.amplitude_to_db(S1[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
    plt.colorbar()
    plt.tight_layout()

    idx = slice(*librosa.time_to_frames([0, 5], sr=sr))
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(librosa.amplitude_to_db(S2[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
    plt.colorbar()
    plt.tight_layout()

    def write_wav(path, y, sr=sr, norm=False):
    
        librosa.util.valid_audio(y, mono=False)

        if norm and np.issubdtype(y.dtype, np.floating):
            wav = librosa.util.normalize(y, norm=np.inf, axis=None)
        else:
            wav = y

        if wav.ndim > 1 and wav.shape[0] == 2:
            wav = wav.T

    
        scipy.io.wavfile.write(path, sr, wav)
    
    h1 = librosa.core.istft(S1*phase1)
    h2 = librosa.core.istft(S2*phase2)

    return write_wav(extracted_1, h1), write_wav(extracted_2, h2)


