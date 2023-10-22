[![Licence](https://img.shields.io/github/license/liujing04/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/blob/main/%E4%BD%BF%E7%94%A8%E9%9C%80%E9%81%B5%E5%AE%88%E7%9A%84%E5%8D%8F%E8%AE%AE-LICENSE.txt)

[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)]()

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/drive/1iWOLYE9znqT6XE5Rw2iETE19ZlqpziLx?usp=sharing)

# Instalación de dependencias 🖥️
Usando pip (python3.9.8 es recomendado)
```bash
python -m venv env
pip install -r requirements.txt
```

## Uso local

Aquí esta el listado de los archivos necesarios para correr el programa:
Puedes descargar los dos primeros desde [Huggingface space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/).

```bash
hubert_base.pt

rmvpe.pt
#Si estás usando windows, necesitas este archivo, omitelo si ffmpeg ffpbobe están instalados; los usuarios de ubuntu/debian pueden instalar estas dos librerías a través de apt install ffmpeg

./ffmpeg

./ffprobe
```

## Créditos
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
+ [Mangio FORK](https://github.com/Mangio621/Mangio-RVC-Fork)

