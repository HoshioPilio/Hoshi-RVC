import os
import uuid
import numpy as np
import torch
import soundfile as sf
from gtts import gTTS
import edge_tts
from inference import Inference
import asyncio
#git+https://github.com/suno-ai/bark.git
# from transformers import AutoProcessor, BarkModel
# import nltk
# from nltk.tokenize import sent_tokenize
# from bark import SAMPLE_RATE

# now_dir = os.getcwd()

def cast_to_device(tensor, device):
    try:
        return tensor.to(device)
    except Exception as e:
        print(e)
        return tensor

# Buscar la forma de evitar descargar el archivo de 4gb cada vez que crea una instancia
# def _bark_conversion_(text, voice_preset):
#     os.makedirs(os.path.join(now_dir, "tts"), exist_ok=True)

#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     dtype = torch.float32 if "cpu" in device else torch.float16
#     bark_processor = AutoProcessor.from_pretrained(
#         "suno/bark",
#         cache_dir=os.path.join(now_dir, "tts", "suno/bark"),
#         torch_dtype=dtype,
#     )
#     bark_model = BarkModel.from_pretrained(
#         "suno/bark",
#         cache_dir=os.path.join(now_dir, "tts", "suno/bark"),
#         torch_dtype=dtype,
#     ).to(device)
#     # bark_model.enable_cpu_offload()
#     inputs = bark_processor(text=[text], return_tensors="pt", voice_preset=voice_preset)
#     tensor_dict = {
#         k: cast_to_device(v, device) if hasattr(v, "to") else v
#         for k, v in inputs.items()
#     }
#     speech_values = bark_model.generate(**tensor_dict, do_sample=True)
#     sampling_rate = bark_model.generation_config.sample_rate
#     speech = speech_values.cpu().numpy().squeeze()
#     return speech, sampling_rate


def tts_infer(tts_text, model_url, tts_method, tts_model):
    print("*****************")
    print(tts_text)
    print(model_url)
    if not tts_text:
        return 'Primero escribe el texto que quieres convertir.', None
    if not tts_model:
        return 'Selecciona un modelo TTS antes de convertir.', None
    if not model_url:
        return 'Escribe la url de modelo que quieres usar antes de convertir.', None
        
    f0_method = "harvest" 
    output_folder = "audios"
    os.makedirs(output_folder, exist_ok=True)
    converted_tts_filename = os.path.join(output_folder, f"tts_out_{uuid.uuid4()}.wav")
    success = False

    if len(tts_text) > 60:
        tts_text = tts_text[:60]
        print("DEMO; limit to 60 characters")

    language = tts_model[:2]
    if tts_method == "Edge-tts":
        try:
            asyncio.run(
                edge_tts.Communicate(
                    tts_text, "-".join(tts_model.split("-")[:-1])
                ).save(converted_tts_filename)
            )
            success = True
        except Exception as e:
            print("ERROR", e)
            try:
                tts = gTTS(tts_text, lang=language)
                tts.save(converted_tts_filename)
                print(
                    f"No audio was received. Please change the tts voice for {tts_model}. USING gTTS."
                )
                success = True
            except:
                tts = gTTS("a", lang=language)
                tts.save(converted_tts_filename)
                print("Error: Audio will be replaced.")
                success = False
                
    # elif tts_method == "Bark-tts":
    #     try:
    #         script = tts_text.replace("\n", " ").strip()
    #         sentences = sent_tokenize(script)
    #         silence = np.zeros(int(0.25 * SAMPLE_RATE))
    #         pieces = []
    #         for sentence in sentences:
    #             audio_array, _ = _bark_conversion_(sentence, tts_model.split("-")[0])
    #             pieces += [audio_array, silence.copy()]

    #         sf.write(
    #             file=converted_tts_filename, samplerate=SAMPLE_RATE, data=np.concatenate(pieces)
    #         )
            
    #     except Exception as e:
    #         print(f"{e}")
    #         return None, None
    
    if success:
        inference = Inference(
            model_name=model_url,
            f0_method=f0_method,
            source_audio_path=converted_tts_filename,
            output_file_name=os.path.join("./audio-outputs", os.path.basename(converted_tts_filename)),
        )
        output = inference.run()
        if os.path.exists(converted_tts_filename):
            os.remove(converted_tts_filename)
        
        if os.path.exists(os.path.join("weights", inference.model_name)):
            os.remove(os.path.join("weights", inference.model_name))
            
        if 'success' in output and output['success']:
            return output, output['file']
        else:
            return output, None
    else:
        return "Ocurrió un error durante la conversión", None
    
    
    