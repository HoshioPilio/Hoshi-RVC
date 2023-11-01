import torchaudio
import numpy as np
import torchaudio.transforms as T
from df import enhance, init_df

df_sr = 48000
model, df_state, _ = init_df()

def audio_enchance(input_audio):
    extension = input_audio.split('.')[-1]
    if extension not in ['wav', 'mpeg', 'ogg']:
        return "El formato del audio no es valido, usa wav, mpeg o ogg", None
    else:
        noisy_audio, sr = torchaudio.load(input_audio)
        print("np.shape(noisy_audio)", np.shape(noisy_audio))
        
        if sr != df_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=df_sr)
            noisy_audio = resampler(noisy_audio)
        
        output_audio = enhance(model, df_state, noisy_audio)
        return np.shape(noisy_audio), noisy_audio
    
    