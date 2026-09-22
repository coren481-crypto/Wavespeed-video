import json
import os
import time
import uuid
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from http.server import HTTPServer, SimpleHTTPRequestHandler


# =========================================
# CONFIGURACIÓN DEL SERVIDOR
# =========================================

HOST = "0.0.0.0"
PORT = 8000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")


# =========================================
# WAVE SPEED
# =========================================

WAVESPEED_URL = (
    "https://api.wavespeed.ai/api/v3/alibaba/wan-3.0/reference-to-video"
)

WAVESPEED_RESULT_URL = (
    "https://api.wavespeed.ai/api/v3/predictions/{}/result"
)


# =========================================
# IMGBB
# =========================================

IMGBB_URL = "https://api.imgbb.com/1/upload"


# =========================================
# CONFIG.JSON
# =========================================

def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {}

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


# =========================================
# RESPUESTAS JSON
# =========================================

def send_json(handler, data, status=200):
    body = json.dumps(
        data,
        ensure_ascii=False
    ).encode("utf-8")

    handler.send_response(status)

    handler.send_header(
        "Content-Type",
        "application/json; charset=utf-8"
    )

    handler.send_header(
        "Content-Length",
        str(len(body))
    )

    handler.send_header(
        "Cache-Control",
        "no-store"
    )

    handler.end_headers()

    handler.wfile.write(body)


# =========================================
# LEER BODY
# =========================================

def read_body(handler):

    length = int(
        handler.headers.get(
            "Content-Length",
            "0"
        )
    )

    if length > 40 * 1024 * 1024:
        raise ValueError(
            "Archivo demasiado grande."
        )

    return handler.rfile.read(length)


# =========================================
# PARSER MULTIPART
# =========================================

def parse_multipart(body, content_type):

    """
    Parser sencillo de multipart/form-data.
    No utiliza cgi, que ya no está disponible
    en Python 3.13.
    """

    if "boundary=" not in content_type:
        raise ValueError(
            "No se encontró boundary multipart."
        )

    boundary = content_type.split(
        "boundary=",
        1
    )[1].strip()

    if (
        boundary.startswith('"')
        and boundary.endswith('"')
    ):
        boundary = boundary[1:-1]

    boundary_bytes = (
        "--" + boundary
    ).encode("utf-8")

    parts = body.split(boundary_bytes)

    fields = {}
    files = {}

    for part in parts:

        if not part:
            continue

        if part.startswith(b"--"):
            continue

        if part.startswith(b"\r\n"):
            part = part[2:]

        if part.endswith(b"\r\n"):
            part = part[:-2]

        separator = b"\r\n\r\n"

        if separator not in part:
            continue

        raw_headers, content = part.split(
            separator,
            1
        )

        headers_text = raw_headers.decode(
            "utf-8",
            errors="replace"
        )

        disposition = None

        for line in headers_text.split("\r\n"):

            if line.lower().startswith(
                "content-disposition:"
            ):
                disposition = line
                break

        if not disposition:
            continue

        name = None
        filename = None

        pieces = disposition.split(";")

        for piece in pieces:

            piece = piece.strip()

            if piece.startswith("name="):
                name = piece[5:].strip('"')

            elif piece.startswith("filename="):
                filename = piece[9:].strip('"')

        if not name:
            continue

        if filename:

            files[name] = {
                "filename": filename,
                "data": content
            }

        else:

            fields[name] = content.decode(
                "utf-8",
                errors="replace"
            )

    return fields, files


# =========================================
# SUBIR IMAGEN A IMGBB
# =========================================

def upload_to_imgbb(
    image_bytes,
    filename,
    api_key
):

    boundary = (
        "----PythonImgBB"
        + uuid.uuid4().hex
    )

    body = bytearray()

    body.extend(
        f"--{boundary}\r\n".encode()
    )

    body.extend(
        b'Content-Disposition: form-data; '
        b'name="image"; '
        + f'filename="{filename}"'.encode()
        + b"\r\n"
    )

    body.extend(
        b"Content-Type: application/octet-stream\r\n\r\n"
    )

    body.extend(image_bytes)

    body.extend(b"\r\n")

    body.extend(
        f"--{boundary}--\r\n".encode()
    )

    url = (
        IMGBB_URL
        + "?"
        + urlencode({
            "key": api_key
        })
    )

    request = Request(
        url,
        data=bytes(body),
        headers={
            "Content-Type":
                f"multipart/form-data; boundary={boundary}",

            "Content-Length":
                str(len(body))
        },
        method="POST"
    )

    try:

        with urlopen(
            request,
            timeout=120
        ) as response:

            result = json.load(response)

    except HTTPError as e:

        error_body = e.read().decode(
            "utf-8",
            errors="replace"
        )

        raise RuntimeError(
            f"ImgBB HTTP {e.code}: {error_body}"
        )

    except URLError as e:

        raise RuntimeError(
            f"No se pudo conectar con ImgBB: {e}"
        )

    if not result.get("success"):

        raise RuntimeError(
            "ImgBB rechazó la imagen: "
            + json.dumps(
                result,
                ensure_ascii=False
            )
        )

    data = result.get(
        "data",
        {}
    )

    image_url = data.get("url")

    if not image_url:

        raise RuntimeError(
            "ImgBB no devolvió una URL de imagen."
        )

    return image_url


# =========================================
# REQUEST A WAVESPEED
# =========================================

def wavespeed_request(
    url,
    api_key,
    data=None
):

    headers = {
        "Authorization":
            f"Bearer {api_key}",

        "Content-Type":
            "application/json"
    }

    encoded = None

    if data is not None:

        encoded = json.dumps(
            data
        ).encode("utf-8")

    request = Request(
        url,
        data=encoded,
        headers=headers,
        method=(
            "POST"
            if data is not None
            else "GET"
        )
    )

    try:

        with urlopen(
            request,
            timeout=120
        ) as response:

            return json.load(response)

    except HTTPError as e:

        error_body = e.read().decode(
            "utf-8",
            errors="replace"
        )

        raise RuntimeError(
            f"WaveSpeed HTTP {e.code}: {error_body}"
        )

    except URLError as e:

        raise RuntimeError(
            f"No se pudo conectar con WaveSpeed: {e}"
        )


# =========================================
# GENERAR VIDEO
# =========================================

def generate_video(
    prompt,
    image_url,
    resolution,
    aspect_ratio,
    duration,
    enable_audio,
    enable_prompt_expansion,
    enable_safety_checker,
    api_key
):

    payload = {

        "prompt": prompt,

        "resolution": resolution,

        "aspect_ratio": aspect_ratio,

        "duration": duration,

        "enable_prompt_expansion":
            enable_prompt_expansion,

        "enable_audio":
            enable_audio,

        "enable_safety_checker":
            enable_safety_checker,

        "reference_images":
            [image_url]
    }

    submit_body = wavespeed_request(
        WAVESPEED_URL,
        api_key,
        payload
    )

    task = submit_body.get(
        "data",
        submit_body
    )

    prediction_id = task.get("id")

    if not prediction_id:

        raise RuntimeError(
            "WaveSpeed no devolvió un prediction ID:\n"
            + json.dumps(
                submit_body,
                ensure_ascii=False
            )
        )

    result_url = (
        WAVESPEED_RESULT_URL.format(
            prediction_id
        )
    )

    print()
    print(
        "Tarea WaveSpeed creada:"
    )
    print(prediction_id)

    while True:

        result_body = wavespeed_request(
            result_url,
            api_key
        )

        result = result_body.get(
            "data",
            result_body
        )

        status = result.get(
            "status"
        )

        print(
            "Estado:",
            status
        )

        if status == "completed":

            outputs = result.get(
                "outputs",
                []
            )

            if not outputs:

                raise RuntimeError(
                    "WaveSpeed terminó pero "
                    "no devolvió outputs."
                )

            return {
                "prediction_id":
                    prediction_id,

                "outputs":
                    outputs,

                "result":
                    result
            }

        if status in {
            "failed",
            "cancelled",
            "timeout",
            "deleted"
        }:

            raise RuntimeError(
                "La generación terminó con "
                "estado "
                + str(status)
                + ":\n"
                + json.dumps(
                    result,
                    ensure_ascii=False
                )
            )

        time.sleep(2)


# =========================================
# HANDLER WEB
# =========================================

class WebHandler(
    SimpleHTTPRequestHandler
):

    def __init__(
        self,
        *args,
        **kwargs
    ):

        super().__init__(
            *args,
            directory=WEB_DIR,
            **kwargs
        )


    # =====================================
    # GET
    # =====================================

    def do_GET(self):

        if self.path == "/api/status":

            config = load_config()

            send_json(
                self,
                {
                    "wavespeed_configured":
                        bool(
                            config.get(
                                "wavespeed_api_key"
                            )
                        )
                }
            )

            return

        super().do_GET()


    # =====================================
    # POST
    # =====================================

    def do_POST(self):

        # ---------------------------------
        # GUARDAR WAVESPEED API KEY
        # ---------------------------------

        if self.path == "/api/save-key":

            try:

                body = read_body(
                    self
                )

                data = json.loads(
                    body.decode("utf-8")
                )

                key = data.get(
                    "api_key",
                    ""
                ).strip()

                if not key:

                    send_json(
                        self,
                        {
                            "success": False,
                            "error":
                                "La API key está vacía."
                        },
                        400
                    )

                    return

                config = load_config()

                config[
                    "wavespeed_api_key"
                ] = key

                # Por seguridad, si existiera
                # una antigua clave de ImgBB
                # en config.json, la eliminamos.

                config.pop(
                    "imgbb_api_key",
                    None
                )

                save_config(
                    config
                )

                send_json(
                    self,
                    {
                        "success": True
                    }
                )

            except Exception as e:

                send_json(
                    self,
                    {
                        "success": False,
                        "error": str(e)
                    },
                    500
                )

            return


        # ---------------------------------
        # ELIMINAR WAVESPEED API KEY
        # ---------------------------------

        if self.path == "/api/delete-key":

            try:

                config = load_config()

                config.pop(
                    "wavespeed_api_key",
                    None
                )

                # También limpiamos cualquier
                # antigua clave de ImgBB.

                config.pop(
                    "imgbb_api_key",
                    None
                )

                save_config(
                    config
                )

                send_json(
                    self,
                    {
                        "success": True
                    }
                )

            except Exception as e:

                send_json(
                    self,
                    {
                        "success": False,
                        "error": str(e)
                    },
                    500
                )

            return


        # ---------------------------------
        # GENERAR VIDEO
        # ---------------------------------

        if self.path == "/api/generate":

            try:

                content_type = (
                    self.headers.get(
                        "Content-Type",
                        ""
                    )
                )

                body = read_body(
                    self
                )

                fields, files = (
                    parse_multipart(
                        body,
                        content_type
                    )
                )

                prompt = fields.get(
                    "prompt",
                    ""
                ).strip()

                if not prompt:

                    raise ValueError(
                        "El prompt está vacío."
                    )

                if "image" not in files:

                    raise ValueError(
                        "No se recibió ninguna imagen."
                    )

                resolution = fields.get(
                    "resolution",
                    "480p"
                )

                aspect_ratio = fields.get(
                    "aspect_ratio",
                    "9:16"
                )

                try:

                    duration = int(
                        fields.get(
                            "duration",
                            "21"
                        )
                    )

                except ValueError:

                    raise ValueError(
                        "La duración no es válida."
                    )

                enable_audio = (
                    fields.get(
                        "enable_audio",
                        "true"
                    ).lower()
                    == "true"
                )

                enable_prompt_expansion = (
                    fields.get(
                        "enable_prompt_expansion",
                        "false"
                    ).lower()
                    == "true"
                )

                enable_safety_checker = (
                    fields.get(
                        "enable_safety_checker",
                        "false"
                    ).lower()
                    == "true"
                )

                allowed_resolutions = {
                    "480p",
                    "720p",
                    "1080p"
                }

                allowed_ratios = {
                    "16:9",
                    "9:16",
                    "1:1",
                    "4:3",
                    "3:4"
                }

                if resolution not in allowed_resolutions:

                    raise ValueError(
                        "Resolución no válida."
                    )

                if aspect_ratio not in allowed_ratios:

                    raise ValueError(
                        "Relación de aspecto no válida."
                    )

                if not 2 <= duration <= 30:

                    raise ValueError(
                        "La duración debe estar "
                        "entre 2 y 30 segundos."
                    )

                # ---------------------------------
                # CARGAR CLAVES
                # ---------------------------------

                config = load_config()

                wavespeed_key = config.get(
                    "wavespeed_api_key"
                )

                # ImgBB NO está en config.json.
                # Se obtiene exclusivamente desde
                # la variable de entorno de Codespaces.

                imgbb_key = os.environ.get(
                    "IMGBB_API_KEY"
                )

                if not wavespeed_key:

                    raise ValueError(
                        "No hay una API key de "
                        "WaveSpeed configurada."
                    )

                if not imgbb_key:

                    raise ValueError(
                        "No está configurada "
                        "IMGBB_API_KEY en el entorno."
                    )

                # ---------------------------------
                # IMAGEN
                # ---------------------------------

                image = files["image"]

                image_bytes = image["data"]

                filename = image["filename"]

                if not image_bytes:

                    raise ValueError(
                        "La imagen está vacía."
                    )

                # ---------------------------------
                # SUBIR A IMGBB
                # ---------------------------------

                print()
                print(
                    "Subiendo imagen a ImgBB..."
                )

                image_url = upload_to_imgbb(
                    image_bytes,
                    filename,
                    imgbb_key
                )

                print(
                    "Imagen ImgBB:"
                )

                print(
                    image_url
                )

                print()

                # ---------------------------------
                # WAVESPEED
                # ---------------------------------

                print(
                    "Enviando tarea a WaveSpeed..."
                )

                result = generate_video(
                    prompt=prompt,

                    image_url=image_url,

                    resolution=resolution,

                    aspect_ratio=aspect_ratio,

                    duration=duration,

                    enable_audio=enable_audio,

                    enable_prompt_expansion=
                        enable_prompt_expansion,

                    enable_safety_checker=
                        enable_safety_checker,

                    api_key=wavespeed_key
                )

                # ---------------------------------
                # RESPUESTA
                # ---------------------------------

                send_json(
                    self,
                    {
                        "success": True,

                        "prediction_id":
                            result[
                                "prediction_id"
                            ],

                        "image_url":
                            image_url,

                        "outputs":
                            result[
                                "outputs"
                            ]
                    }
                )

            except Exception as e:

                print()
                print(
                    "ERROR:"
                )

                print(
                    str(e)
                )

                print()

                send_json(
                    self,
                    {
                        "success": False,
                        "error": str(e)
                    },
                    500
                )

            return


        # ---------------------------------
        # ENDPOINT DESCONOCIDO
        # ---------------------------------

        send_json(
            self,
            {
                "success": False,
                "error":
                    "Endpoint no encontrado."
            },
            404
        )


# =========================================
# INICIAR SERVIDOR
# =========================================

server = HTTPServer(
    (HOST, PORT),
    WebHandler
)

print()
print(
    "==================================="
)

print(
    " WaveSpeed Video Generator"
)

print(
    "==================================="
)

print()

print(
    f"http://{HOST}:{PORT}"
)

print()

print(
    "Pulsa CTRL+C para detenerlo."
)

print()

server.serve_forever()