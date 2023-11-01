import os
import uuid
import numpy as np
import torch
import soundfile as sf
from gtts import gTTS
import edge_tts
from inference import Inference
import asyncio
from elevenlabs import voices, generate, save
from elevenlabs.api.error import UnauthenticatedRateLimitError
# Not working in windows
import platform

COQUI_LANGUAGES = []
if platform.system() != 'Windows':
    from neon_tts_plugin_coqui import CoquiTTS
    
    # CoquiTTS
    COQUI_LANGUAGES = list(CoquiTTS.langs.keys())
    coquiTTS = CoquiTTS()


# Elevenlabs
ELEVENLABS_VOICES_RAW = voices()

def get_elevenlabs_voice_names():
    elevenlabs_voice_names = []
    for voice in ELEVENLABS_VOICES_RAW:
        elevenlabs_voice_names.append(voice.name)
    return elevenlabs_voice_names

ELEVENLABS_VOICES_NAMES = get_elevenlabs_voice_names()

def tts_infer(tts_text, model_url, tts_method, tts_model, tts_api_key, language):
    if not tts_text:
        return 'Primero escribe el texto que quieres convertir.', None
    if not tts_model and tts_method != 'CoquiTTS':
        return 'Selecciona un modelo TTS antes de convertir.', None
        
    f0_method = "harvest" 
    output_folder = "audios"
    os.makedirs(output_folder, exist_ok=True)
    converted_tts_filename = os.path.join(output_folder, f"tts_out_{uuid.uuid4()}.wav")
    success = False

    if tts_method == "Edge-tts":
        language = tts_model[:2]
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
                
    # if tts_method == "Tortoise":
    #     api.TextToSpeech()
        
    if tts_method == "CoquiTTS":
        if platform.system() == 'Windows':
            return "Funcionalidad no disponible en windows", None
            
        print(tts_text, language)
        # return output
        coquiTTS.get_tts(tts_text, converted_tts_filename, speaker = {"language" : language})
        success = True
        
    if tts_method == 'ElevenLabs':
        if len(tts_text) > 2499:
            return "El límite de cuentas no logeadas es de 2500 caracteres.", None
        try:
            audio = generate(
                text=tts_text,
                voice=tts_model,
                model="eleven_multilingual_v2",
                api_key=tts_api_key
            )
            save(audio=audio, filename=converted_tts_filename)
            success = True
        except UnauthenticatedRateLimitError:
            return "Necesitas configurar tu API Key para usar elevenlabs", None
        
    if not model_url:
        return 'Pon la url del modelo si quieres aplicarle otro tono.', converted_tts_filename

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
    
    
    