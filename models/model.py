import zipfile
import hashlib
from utils.model import model_downloader, get_model
import requests
import json
import torch
import os
from inference import Inference
import gradio as gr
from constants import VOICE_METHODS, BARK_VOICES, EDGE_VOICES, zips_folder, unzips_folder
from tts.conversion import tts_infer, ELEVENLABS_VOICES_RAW, ELEVENLABS_VOICES_NAMES

api_url = "https://rvc-models-api.onrender.com/uploadfile/"

if not os.path.exists(zips_folder):
      os.mkdir(zips_folder)
if not os.path.exists(unzips_folder):
  os.mkdir(unzips_folder)
           
def get_info(path):
    path = os.path.join(unzips_folder, path)
    try:
        a = torch.load(path, map_location="cpu")
        return a
    except Exception as e:
        print("*****************eeeeeeeeeeeeeeeeeeeerrrrrrrrrrrrrrrrrr*****")
        print(e)
        return {

        }
def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def compress(modelname, files):
    file_path = os.path.join(zips_folder, f"{modelname}.zip")
    # Select the compression mode ZIP_DEFLATED for compression
    # or zipfile.ZIP_STORED to just store the file
    compression = zipfile.ZIP_DEFLATED

    # Comprueba si el archivo ZIP ya existe
    if not os.path.exists(file_path):
        # Si no existe, crea el archivo ZIP
        with zipfile.ZipFile(file_path, mode="w") as zf:
            try:
                for file in files:
                    if file:
                        # Agrega el archivo al archivo ZIP
                        zf.write(unzips_folder if ".index" in file else os.path.join(unzips_folder, file), compress_type=compression)
            except FileNotFoundError as fnf:
                print("An error occurred", fnf)
    else:
        # Si el archivo ZIP ya existe, agrega los archivos a un archivo ZIP existente
        with zipfile.ZipFile(file_path, mode="a") as zf:
            try:
                for file in files:
                    if file:
                        # Agrega el archivo al archivo ZIP
                         zf.write(unzips_folder if ".index" in file else os.path.join(unzips_folder, file), compress_type=compression)
            except FileNotFoundError as fnf:
                print("An error occurred", fnf)

    return file_path

def infer(model, f0_method, audio_file, index_rate, vc_transform0, protect0, resample_sr1, filter_radius1):
    
    if not model:
        return "No model url specified, please specify a model url.", None
    
    if not audio_file:
        return "No audio file specified, please load an audio file.", None
    
    
    inference = Inference(
        model_name=model,
        f0_method=f0_method,
        source_audio_path=audio_file,
        feature_ratio=index_rate,
        transposition=vc_transform0,
        protection_amnt=protect0,
        resample=resample_sr1,
        harvest_median_filter=filter_radius1,
        output_file_name=os.path.join("./audio-outputs", os.path.basename(audio_file))
    )
    output = inference.run()
    if 'success' in output and output['success']:
        print("Inferencia realizada exitosamente...")
        return output, output['file']
    else:
        print("Fallo en la inferencia...", output)
        return "Failed", None
    
def post_model(name, model_url, version, creator):
    modelname = model_downloader(model_url, zips_folder, unzips_folder)
    
    if not modelname:
        return "No se ha podido descargar el modelo, intenta con otro enlace o intentalo más tarde."
    
    model_files = get_model(unzips_folder, modelname)
    
    if not model_files:
        return "No se encontrado un modelo valido, verifica el contenido del enlace e intentalo más tarde."

    if not model_files.get('pth'):
        return "No se encontrado un modelo valido, verifica el contenido del enlace e intentalo más tarde."
    
    md5_hash = calculate_md5(os.path.join(unzips_folder,model_files['pth']))
    zipfile = compress(modelname, list(model_files.values()))
    
    a = get_info(model_files.get('pth'))
    file_to_upload = open(zipfile, "rb")
    info = a.get("info", "None"),
    sr = a.get("sr", "None"),
    f0 = a.get("f0", "None"),
    
    data = {
        "name": name,
        "version": version,
        "creator": creator,
        "hash": md5_hash,
        "info": info,
        "sr": sr,
        "f0": f0
    }
    print("Subiendo archivo...")
    # Realizar la solicitud POST
    response = requests.post(api_url, files={"file": file_to_upload}, data=data)
    result = response.json()
    
    # Comprobar la respuesta
    if response.status_code == 200:
        result = response.json()
        return json.dumps(result, indent=4)
    else:
        print("Error al cargar el archivo:", response.status_code)
        return result
        

def search_model(name):
    web_service_url = "https://script.google.com/macros/s/AKfycbyRaNxtcuN8CxUrcA_nHW6Sq9G2QJor8Z2-BJUGnQ2F_CB8klF4kQL--U2r2MhLFZ5J/exec"
    response = requests.post(web_service_url, json={
        'type': 'search_by_filename',
        'name': name
    })
    result = []
    response.raise_for_status()  # Lanza una excepción en caso de error
    json_response = response.json()
    cont = 0
    result.append("""| Nombre del modelo | Url | Epoch | Sample Rate |
                  | ---------------- | -------------- |:------:|:-----------:|
                  """)
    yield "<br />".join(result)
    if json_response.get('ok', None):
        for model in json_response['ocurrences']:
            if cont < 20:
                model_name = str(model.get('name', 'N/A')).strip()
                model_url = model.get('url', 'N/A')
                epoch = model.get('epoch', 'N/A')
                sr = model.get('sr', 'N/A')
                line = f"""|{model_name}|<a>{model_url}</a>|{epoch}|{sr}|
                """
                result.append(line)
                yield "".join(result)
            cont += 1
            
def update_tts_methods_voice(select_value):
    if select_value == "Edge-tts":
        return gr.Dropdown.update(choices=EDGE_VOICES, visible=True, value="es-CO-GonzaloNeural-Male"), gr.Markdown.update(visible=False), gr.Textbox.update(visible=False),gr.Radio.update(visible=False)
    elif select_value == "Bark-tts":
        return gr.Dropdown.update(choices=BARK_VOICES, visible=True), gr.Markdown.update(visible=False), gr.Textbox.update(visible=False),gr.Radio.update(visible=False)
    elif select_value == 'ElevenLabs':
        return gr.Dropdown.update(choices=ELEVENLABS_VOICES_NAMES, visible=True, value="Bella"), gr.Markdown.update(visible=True), gr.Textbox.update(visible=True), gr.Radio.update(visible=False)
    elif select_value == 'CoquiTTS':
        return gr.Dropdown.update(visible=False), gr.Markdown.update(visible=False), gr.Textbox.update(visible=False), gr.Radio.update(visible=True)
