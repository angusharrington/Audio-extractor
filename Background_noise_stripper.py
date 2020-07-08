
# %%

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import librosa

import librosa.display


def background_noise_stripper (audio_path, new_path):   
    '''
    This function takes the audio file in either mp3 or wav format and creates a new wav file
    (need to specify .wav in 'new_path') of the vocals only. Also returns spectograms of snippets of 
    the audio file and of the background and foreground audio.
    '''
    y, sr = librosa.load(audio_path)


    S_full, phase = librosa.magphase(librosa.stft(y))

    idx = slice(*librosa.time_to_frames([30, 35], sr=sr))
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
    plt.colorbar()
    plt.tight_layout()


    S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))

    S_filter = np.minimum(S_full, S_filter)

    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)


    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
    plt.title('Full spectrum')
    plt.colorbar()

    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
    plt.title('Background')
    plt.colorbar()
    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
    plt.title('Foreground')
    plt.colorbar()
    plt.tight_layout()
    plt.show()



    def write_wav(path, y, sr, norm=False):

        librosa.util.valid_audio(y, mono=False)

        if norm and np.issubdtype(y.dtype, np.floating):
            wav = librosa.util.normalize(y, norm=np.inf, axis=None)
        else:
            wav = y

        if wav.ndim > 1 and wav.shape[0] == 2:
            wav = wav.T

    
        scipy.io.wavfile.write(path, sr, wav)
    
    h = librosa.core.istft(S_foreground*phase)

    return write_wav(new_path, h, sr)


# %%
