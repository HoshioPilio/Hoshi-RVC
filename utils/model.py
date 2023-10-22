import os
import shutil
from mega import Mega
import gdown
import re
import wget
import sys
import uuid
import zipfile


class InvalidDriveId(Exception):
    def __init__(self, message="Error de la url"):
        self.message = message
        super().__init__(self.message)


def model_downloader(url, zip_path, dest_path):
    """Download and unzip a file from Google Drive or Mega."""
    
    def drive_download(url, dest_folder):
        print(f"Descargando desde drive...")
        try:
            filename = gdown.download(url, os.path.join(dest_folder, f"{uuid.uuid4()}.zip"), fuzzy=True)
            return filename
        except:
            print("El intento de descargar con drive no funcionó")
            return None

    def mega_download(url, dest_folder):
        try:
            file_id = None
            if "#!" in url:
                file_id = url.split("#!")[1].split("!")[0]
            elif "file/" in url:
                file_id = url.split("file/")[1].split("/")[0]
            else:
                file_id = None

            print(f"Descargando desde mega...")
            if file_id:
                mega = Mega()
                m = mega.login()
                filename = m.download_url(url, dest_path=dest_folder, dest_filename=f"{uuid.uuid4()}.zip")

                return os.path.basename(filename)
            else:
                return None

        except Exception as e:
            print("Ocurrio un error**")
            print(e)
            return None

    def download(url, dest_folder):
        try:
            print(f"Descargando desde url generica...")
            dest_path = wget.download(url=url, out=os.path.join(dest_folder, f"{uuid.uuid4()}.zip"))

            return os.path.basename(dest_path)
        except Exception as e:
            print(f"Error al descargar el archivo: {str(e)}")

    filename = ""

    if not os.path.exists(zip_path):
        os.mkdir(zip_path)

    if url and 'drive.google.com' in url:
        # Descargar el elemento si la URL es de Google Drive
        filename = drive_download(url, zip_path)
    elif url and 'mega.nz' in url:
        filename = mega_download(url, zip_path)
    elif url and 'pixeldrain' in url:
        print("No se puede descargar de pixeldrain")
        sys.exit()
    else:
        filename = download(url, zip_path)

    if filename:
        print(f"Descomprimiendo {filename}...")
        modelname = str(filename).replace(".zip", "")
        zip_file_path = os.path.join(zip_path, filename)
        
        try:
            shutil.unpack_archive(zip_file_path, os.path.join(dest_path, modelname))
        except Exception as e:
            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(dest_path)
            except zipfile.BadZipFile as e:
                print(f"Error: El archivo ZIP no es válido - {e}")
            except Exception as e:
                print(f"Error inesperado: {e}")
        
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)

        return modelname
    else:
        return None


def get_models(weight_path):
    # Obtener todos los elementos en la ruta
    files = os.listdir(weight_path)
    # Filtrar solo los directorios
    return [file for file in files if os.path.isdir(os.path.join(weight_path, file))]


def get_model(weight_path, modelname):
    resources = {}
    for root, dirs, files in os.walk(os.path.join(weight_path, modelname)):
        for file in files:
            if file.endswith('.index'):
                resources['index'] =  os.path.relpath(os.path.join(root, file))
            if file.endswith('.pth'):
                resources['pth'] =  os.path.relpath(os.path.join(root, file), start=weight_path)
    return resources


def get_audios(audios_path):
    # Obtener todos los elementos en la ruta
    files = os.listdir(audios_path)
    # Filtrar solo los directorios
    return [file for file in files if not os.path.isdir(os.path.join(audios_path, file)) and os.path.join(audios_path, file).endswith(('.mp3', '.wav'))]