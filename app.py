import gradio as gr
from inference import Inference
import os
import zipfile
import hashlib
from utils.model import model_downloader, get_model
import requests
import json
from tts.constants import VOICE_METHODS, BARK_VOICES, EDGE_VOICES
from tts.conversion import tts_infer

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
    
    if not model_files:
        return "No se encontrado un modelo valido, verifica el contenido del enlace e intentalo más tarde."

    if not model_files.get('pth'):
        return "No se encontrado un modelo valido, verifica el contenido del enlace e intentalo más tarde."
    
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
        return gr.update(choices=EDGE_VOICES)
    elif select_value == "Bark-tts":
        return gr.update(choices=BARK_VOICES)

with gr.Blocks() as app:
    gr.HTML("<h1> Simple RVC Inference - by Juuxn 💻 </h1>")
    
    with gr.Tab("Inferencia"):
        model_url = gr.Textbox(placeholder="https://huggingface.co/AIVER-SE/BillieEilish/resolve/main/BillieEilish.zip", label="Url del modelo", show_label=True)
        audio_path = gr.Audio(label="Archivo de audio", show_label=True, type="filepath", )
        f0_method = gr.Dropdown(choices=["harvest", "pm", "crepe", "crepe-tiny", "mangio-crepe", "mangio-crepe-tiny", "rmvpe"], 
                                value="rmvpe", 
                                label="Algoritmo", show_label=True)
        # Salida
        with gr.Row():
            vc_output1 = gr.Textbox(label="Salida")
            vc_output2 = gr.Audio(label="Audio de salida")
                            
        btn = gr.Button(value="Convertir")
        btn.click(infer, inputs=[model_url, f0_method, audio_path], outputs=[vc_output1, vc_output2])
        
    with gr.TabItem("TTS"):
        with gr.Row():
            tts_text = gr.Textbox(
                label="Texto:",
                placeholder="Texto que deseas convertir a voz...",
                lines=6,
            )

        with gr.Column():
            with gr.Row():
                tts_model_url = gr.Textbox(placeholder="https://huggingface.co/AIVER-SE/BillieEilish/resolve/main/BillieEilish.zip", label="Url del modelo RVC", show_label=True)
                
            with gr.Column():
                tts_method = gr.Dropdown(choices=VOICE_METHODS, value="Edge-tts", label="Método TTS:", visible=False)
                tts_model = gr.Dropdown(choices=EDGE_VOICES, label="Modelo TTS:", visible=True, interactive=True)
                tts_method.change(fn=update_tts_methods_voice, inputs=[tts_method], outputs=[tts_model])
                    
            with gr.Row():
                tts_vc_output1 = gr.Textbox(label="Salida")
                tts_vc_output2 = gr.Audio(label="Audio de salida")   
            
        tts_btn = gr.Button(value="Convertir")
        tts_btn.click(fn=tts_infer, inputs=[tts_text, tts_model_url, tts_method, tts_model], outputs=[tts_vc_output1, tts_vc_output2])
        
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
                

        #     with gr.Column():
        #         model_voice_path07 = gr.Dropdown(
        #             label=i18n("RVC Model:"),
        #             choices=sorted(names),
        #             value=default_weight,
        #         )
        #         best_match_index_path1, _ = match_index(
        #             model_voice_path07.value
        #         )

        #         file_index2_07 = gr.Dropdown(
        #             label=i18n("Select the .index file:"),
        #             choices=get_indexes(),
        #             value=best_match_index_path1,
        #             interactive=True,
        #             allow_custom_value=True,
        #         )
        # with gr.Row():
        #     refresh_button_ = gr.Button(i18n("Refresh"), variant="primary")
        #     refresh_button_.click(
        #         fn=change_choices2,
        #         inputs=[],
        #         outputs=[model_voice_path07, file_index2_07],
        #     )
        # with gr.Row():
        #     original_ttsvoice = gr.Audio(label=i18n("Audio TTS:"))
        #     ttsvoice = gr.Audio(label=i18n("Audio RVC:"))

        # with gr.Row():
        #     button_test = gr.Button(i18n("Convert"), variant="primary")

        # button_test.click(
        #     tts.use_tts,
        #     inputs=[
        #         text_test,
        #         tts_test,
        #         model_voice_path07,
        #         file_index2_07,
        #         # transpose_test,
        #         vc_transform0,
        #         f0method8,
        #         index_rate1,
        #         crepe_hop_length,
        #         f0_autotune,
        #         ttsmethod_test,
        #     ],
        #     outputs=[ttsvoice, original_ttsvoice],
        # )

    
    
    app.queue(concurrency_count=511, max_size=1022).launch()
    #share=True