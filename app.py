import gradio as gr
from inference import Inference
import os

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

with gr.Blocks() as app:
    gr.HTML("<h1> Simple RVC Inference - by Juuxn 💻 </h1>")
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
    
    app.queue(concurrency_count=511, max_size=1022).launch(share=True)