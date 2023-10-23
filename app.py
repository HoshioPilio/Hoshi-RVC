import gradio as gr
from inference import Inference
import os
import zipfile
import hashlib
from utils.model import model_downloader, get_model
import requests
import json

api_url = "https://rvc-models-api.onrender.com/uploadfile/"

zips_folder = "./zips"
unzips_folder = "./unzips"
if not os.path.exists(zips_folder):
      os.mkdir(zips_folder)
if not os.path.exists(unzips_folder):
  os.mkdir(unzips_folder)
           
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

def infer(model, f0_method, audio_file):
    print("****", audio_file)
    inference = Inference(
        model_name=model,
        f0_method=f0_method,
        source_audio_path=audio_file,
        output_file_name=os.path.join("./audio-outputs", os.path.basename(audio_file))
    )
    output = inference.run()
    if 'success' in output and output['success']:
        return output, output['file']
    else:
        return
    
def post_model(name, model_url, version, creator):
    modelname = model_downloader(model_url, zips_folder, unzips_folder)
    model_files = get_model(unzips_folder, modelname)
    md5_hash = calculate_md5(os.path.join(unzips_folder,model_files['pth']))
    zipfile = compress(modelname, list(model_files.values()))
    file_to_upload = open(zipfile, "rb")
    data = {
        "name": name,
        "version": version,
        "creator": creator,
        "hash": md5_hash
    }
    print("Subiendo archivo...")
    # Realizar la solicitud POST
    response = requests.post(api_url, files={"file": file_to_upload}, data=data)
    
    # Comprobar la respuesta
    if response.status_code == 200:
        result = response.json()
        return json.dumps(result, indent=4)
    else:
        print("Error al cargar el archivo:", response.status_code)
        return result

def search_model(name):
    web_service_url = "https://script.google.com/macros/s/AKfycbzfIOiwmPj-q8-hEyvjRQfgLtO7ESolmtsQmnNheCujwnitDApBSjgTecdfXb8f2twT/exec"
    response = requests.post(web_service_url, json={
        'type': 'search_by_filename',
        'name': name
    })
    result = []
    response.raise_for_status()  # Lanza una excepción en caso de error
    json_response = response.json()
    cont = 0
    if json_response.get('ok', None):
        for model in json_response['ocurrences']:
            if cont < 20:
                model_name = model.get('name', 'N/A')
                model_url = model.get('url', 'N/A')
                result.append(f"**Nombre del modelo: {model_name}**</br>{model_url}</br>")
                yield "</br>".join(result)
            cont += 1

with gr.Blocks() as app:
    gr.HTML("<h1> Simple RVC Inference - by Juuxn 💻 </h1>")
    
    with gr.Tab("Inferencia"):
        model_url = gr.Textbox(placeholder="https://huggingface.co/AIVER-SE/BillieEilish/resolve/main/BillieEilish.zip", label="Url del modelo", show_label=True)
        audio_path = gr.Audio(label="Archivo de audio", show_label=True, type="filepath", )
        f0_method = gr.Dropdown(choices=["harvest", "pm", "crepe", "crepe-tiny", "mangio-crepe", "mangio-crepe-tiny", "rmvpe"], 
                                value="harvest", 
                                label="Algoritmo", show_label=True)
        # Salida
        with gr.Row():
            vc_output1 = gr.Textbox(label="Salida")
            vc_output2 = gr.Audio(label="Audio de salida")
                            
        btn = gr.Button(value="Convertir")
        btn.click(infer, inputs=[model_url, f0_method, audio_path], outputs=[vc_output1, vc_output2])
        
    with gr.Tab("Recursos"):
        gr.HTML("<h4>Buscar modelos</h4>")
        search_name = gr.Textbox(placeholder="Billie Eillish (RVC v2 - 100 epoch)", label="Nombre", show_label=True)
         # Salida
        with gr.Row():
            sarch_output = gr.Markdown(label="Salida")
            
        btn_search_model = gr.Button(value="Buscar")
        btn_search_model.click(fn=search_model, inputs=[search_name], outputs=[sarch_output])
        
        gr.HTML("<h4>Publica tu modelo</h4>")
        post_name = gr.Textbox(placeholder="Billie Eillish (RVC v2 - 100 epoch)", label="Nombre", show_label=True)
        post_model_url = gr.Textbox(placeholder="https://huggingface.co/AIVER-SE/BillieEilish/resolve/main/BillieEilish.zip", label="Url del modelo", show_label=True)
        post_creator = gr.Textbox(placeholder="ID de discord o enlace al perfil del creador", label="Creador", show_label=True)
        post_version = gr.Dropdown(choices=["RVC v1", "RVC v2"], value="RVC v1", label="Versión", show_label=True)
        
         # Salida
        with gr.Row():
            post_output = gr.Markdown(label="Salida")
            
        btn_post_model = gr.Button(value="Publicar")
        btn_post_model.click(fn=post_model, inputs=[post_name, post_model_url, post_version, post_creator], outputs=[post_output])
    
    app.queue(concurrency_count=511, max_size=1022).launch(share=True)