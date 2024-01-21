import gradio as gr
import os
from constants import VOICE_METHODS, BARK_VOICES, EDGE_VOICES
import platform
from models.model import *
from tts.conversion import COQUI_LANGUAGES
import pytube
import os
import traceback
from pydub import AudioSegment
# from audio_enhance.functions import audio_enhance

def convert_yt_to_wav(url):
    if not url:
        return "First enter the video link", None
    
    try:
        print(f"Converting {url} video...")
        # Descargar el video utilizando pytube
        video = pytube.YouTube(url)
        stream = video.streams.filter(only_audio=True).first()
        video_output_folder = os.path.join(f"yt_videos")  # Ruta de destino de la carpeta
        audio_output_folder = 'audios'

        print("Downloading video")
        video_file_path = stream.download(output_path=video_output_folder)
        print(video_file_path)

        file_name = os.path.basename(video_file_path)
        
        audio_file_path = os.path.join(audio_output_folder, file_name.replace('.mp4','.wav'))
        # convert mp4 to wav
        print("Converting to wav")
        sound = AudioSegment.from_file(video_file_path,format="mp4")
        sound.export(audio_file_path, format="wav")
        
        if os.path.exists(video_file_path):
            os.remove(video_file_path)
            
        return "Success", audio_file_path
    except ConnectionResetError as cre:
        return "Connection has been lost, recharge or try again later.", None
    except Exception as e:
        return str(e), None
    
with gr.Blocks() as app:
    gr.HTML("<h1> Simple RVC Inference - by Juuxn 💻 </h1>")
    

    with gr.Tab("Inference"):
        model_url = gr.Textbox(placeholder="https://huggingface.co/AIVER-SE/BillieEilish/resolve/main/BillieEilish.zip", label="Url voice models", show_label=True)
        with gr.Row():
            with gr.Column():
                audio_path = gr.Audio(label="Audio file", show_label=True, type="filepath",)
                index_rate = gr.Slider(minimum=0, maximum=1, label="Search feature ratio:", value=0.75, interactive=True,)
                filter_radius1 = gr.Slider(minimum=0, maximum=7, label="Filter (reduction of breathing harshness)", value=3, step=1, interactive=True,)
            with gr.Column():
                f0_method = gr.Dropdown(choices=["harvest", "pm", "crepe", "crepe-tiny", "mangio-crepe", "mangio-crepe-tiny", "rmvpe"], 
                                    value="rmvpe", 
                                    label="pitch_algorit", show_label=True)
                vc_transform0 = gr.Slider(minimum=-12, label="Número de semitonos, subir una octava: 12, bajar una octava: -12", value=0, maximum=12, step=1)
                protect0 = gr.Slider(
                    minimum=0, maximum=0.5, label="Protejer las consonantes sordas y los sonidos respiratorios. 0.5 para desactivarlo.", value=0.33,
                    step=0.01,
                interactive=True,
                )
                resample_sr1 = gr.Slider(
                    minimum=0,
                    maximum=48000,
                    label="Re-muestreo sobre el audio de salida hasta la frecuencia de muestreo final. 0 para no re-muestrear.",
                    value=0,
                    step=1,
                    interactive=True,
                )
                 
        # Salida
        with gr.Row():
            vc_output1 = gr.Textbox(label="Salida")
            vc_output2 = gr.Audio(label="Audio de salida")
                            
        btn = gr.Button(value="Convertir")
        btn.click(infer, inputs=[model_url, f0_method, audio_path, index_rate, vc_transform0, protect0, resample_sr1, filter_radius1], outputs=[vc_output1, vc_output2])
        
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
                
            with gr.Row():
                tts_method = gr.Dropdown(choices=VOICE_METHODS, value="Edge-tts", label="Método TTS:", visible=True)
                tts_model = gr.Dropdown(choices=EDGE_VOICES, label="Modelo TTS:", visible=True, interactive=True)
                tts_api_key = gr.Textbox(label="ElevenLabs Api key", show_label=True, placeholder="4a4afce72349680c8e8b6fdcfaf2b65a",interactive=True, visible=False)
            
            tts_coqui_languages = gr.Radio(
                label="Language",
                choices=COQUI_LANGUAGES,
                value="en",
                visible=False
            )
            
            tts_btn = gr.Button(value="Convertir")
                
            with gr.Row():
                tts_vc_output1 = gr.Textbox(label="Salida")
                tts_vc_output2 = gr.Audio(label="Audio de salida")   
            
        tts_btn.click(fn=tts_infer, inputs=[tts_text, tts_model_url, tts_method, tts_model, tts_api_key, tts_coqui_languages], outputs=[tts_vc_output1, tts_vc_output2])
        
        tts_msg = gr.Markdown("""**Recomiendo que te crees una cuenta de eleven labs y pongas tu clave de api, es gratis y tienes 10k caracteres de limite al mes.** <br/>
                ![Imgur](https://imgur.com/HH6YTu0.png)
                """, visible=False)
        
        tts_method.change(fn=update_tts_methods_voice, inputs=[tts_method], outputs=[tts_model, tts_msg, tts_api_key, tts_coqui_languages])
    
    with gr.TabItem("Youtube"):
        gr.Markdown("## Convertir video de Youtube a audio")
        with gr.Row():
            yt_url = gr.Textbox(
                label="Url del video:",
                placeholder="https://www.youtube.com/watch?v=3vEiqil5d3Q"
            )
        yt_btn = gr.Button(value="Convertir")
                
        with gr.Row():
            yt_output1 = gr.Textbox(label="Salida")
            yt_output2 = gr.Audio(label="Audio de salida")   
            
        yt_btn.click(fn=convert_yt_to_wav, inputs=[yt_url], outputs=[yt_output1, yt_output2])
         
    # with gr.TabItem("Mejora de audio"):
    #     enhance_input_audio = gr.Audio(label="Audio de entrada")
    #     enhance_output_audio = gr.Audio(label="Audio de salida")

    #     btn_enhance_audio = gr.Button()
    #     # btn_enhance_audio.click(fn=audio_enhance, inputs=[enhance_input_audio], outputs=[enhance_output_audio])
        
        
    with gr.Tab("Modelos"):
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

    
    
    app.queue(concurrency_count=200, max_size=1022).launch()
    #share=True
