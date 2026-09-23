import json
import os
import time
import mimetypes
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

HOST = "0.0.0.0"
PORT = 8000

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
CONFIG_FILE = BASE_DIR / "config.json"

# ----------------------------
# WaveSpeed
# ----------------------------

WAVESPEED_VIDEO_URL = (
    "https://api.wavespeed.ai/api/v3/"
    "alibaba/wan-3.0/reference-to-video"
)

WAVESPEED_IMAGE_URL = (
    "https://api.wavespeed.ai/api/v3/"
    "bytedance/seedream-v5.0-pro"
)

WAVESPEED_EDIT_URL = (
    "https://api.wavespeed.ai/api/v3/"
    "bytedance/seedream-v5.0-pro/edit"
)

WAVESPEED_RESULT_URL = (
    "https://api.wavespeed.ai/api/v3/"
    "predictions/{}/result"
)

# ----------------------------
# ImgBB
# ----------------------------

IMGBB_URL = "https://api.imgbb.com/1/upload"


# ============================================================
# CONFIG FILE
# ============================================================

def load_config():
    if not CONFIG_FILE.exists():
        return {}

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ============================================================
# HTTP HELPERS
# ============================================================

def send_json(handler, data, status=200):
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")

    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_body(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    return handler.rfile.read(length)


def request_json(url, payload, api_key):
    data = json.dumps(payload).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    request = Request(url, data=data, headers=headers, method="POST")

    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    except HTTPError as e:
        try:
            error_body = e.read().decode("utf-8")
        except Exception:
            error_body = str(e)

        raise RuntimeError(f"WaveSpeed HTTP {e.code}: {error_body}")

    except URLError as e:
        raise RuntimeError(f"Network error: {e}")


def wavespeed_request(url, payload, api_key):
    return request_json(url, payload, api_key)


# ============================================================
# PREDICTION HELPERS
# ============================================================

def get_prediction_id(response):
    data = response.get("data", response)

    if isinstance(data, dict):
        prediction_id = data.get("id")

        if prediction_id:
            return prediction_id

    raise RuntimeError(
        "WaveSpeed no devolvió un prediction ID.\n"
        + json.dumps(response, ensure_ascii=False)
    )


def poll_prediction(prediction_id, api_key):
    url = WAVESPEED_RESULT_URL.format(prediction_id)
    headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        request = Request(url, headers=headers, method="GET")

        try:
            with urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))

        except HTTPError as e:
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                error_body = str(e)

            raise RuntimeError(
                f"Error consultando WaveSpeed HTTP {e.code}: {error_body}"
            )

        result = body.get("data", body)

        if not isinstance(result, dict):
            raise RuntimeError("Respuesta inválida de WaveSpeed.")

        status = result.get("status")

        if status == "completed":
            return result

        if status in {"failed", "cancelled", "timeout", "deleted"}:
            raise RuntimeError(json.dumps(result, ensure_ascii=False))

        time.sleep(2)


def get_first_output(result):
    outputs = result.get("outputs", [])

    if isinstance(outputs, str):
        return outputs

    if isinstance(outputs, list) and outputs:
        first = outputs[0]

        if isinstance(first, str):
            return first

        if isinstance(first, dict):
            for key in ("url", "video_url", "image_url", "output"):
                if first.get(key):
                    return first[key]

    return None


# ============================================================
# MULTIPART PARSER
# ============================================================

def parse_multipart(handler, body):
    content_type = handler.headers.get("Content-Type", "")

    if "boundary=" not in content_type:
        raise RuntimeError("No se encontró boundary multipart.")

    boundary = content_type.split("boundary=", 1)[1].strip()

    if boundary.startswith('"'):
        boundary = boundary[1:-1]

    boundary_bytes = b"--" + boundary.encode()

    fields = {}
    files = []

    parts = body.split(boundary_bytes)

    for part in parts:
        if not part:
            continue

        part = part.strip(b"\r\n-")

        if not part:
            continue

        if b"\r\n\r\n" not in part:
            continue

        header_data, content = part.split(b"\r\n\r\n", 1)
        headers = header_data.decode("utf-8", errors="ignore")

        disposition = None

        for line in headers.split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                disposition = line

        if not disposition:
            continue

        name = None
        filename = None

        for item in disposition.split(";"):
            item = item.strip()

            if item.startswith("name="):
                name = item.split("=", 1)[1].strip('"')
            elif item.startswith("filename="):
                filename = item.split("=", 1)[1].strip('"')

        if not name:
            continue

        if filename:
            content_type_value = "application/octet-stream"

            for line in headers.split("\r\n"):
                if line.lower().startswith("content-type:"):
                    content_type_value = line.split(":", 1)[1].strip()

            files.append({
                "field": name,
                "filename": filename,
                "content_type": content_type_value,
                "data": content
            })

        else:
            fields[name] = content.decode("utf-8", errors="ignore")

    return fields, files


# ============================================================
# IMGBB
# ============================================================

def upload_to_imgbb(image_bytes, filename, api_key):
    if not api_key:
        raise RuntimeError(
            "No existe IMGBB_API_KEY en los secretos de Codespaces."
        )

    boundary = "----WaveSpeedBoundary" + str(int(time.time() * 1000))
    body = bytearray()

    def add_field(name, value):
        body.extend(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            ).encode()
        )
        body.extend(value.encode())
        body.extend(b"\r\n")

    add_field("key", api_key)

    body.extend(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; '
            f'filename="{filename}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode()
    )

    body.extend(image_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body))
    }

    request = Request(IMGBB_URL, data=bytes(body), headers=headers, method="POST")

    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))

    except HTTPError as e:
        try:
            error_body = e.read().decode("utf-8")
        except Exception:
            error_body = str(e)

        raise RuntimeError(f"ImgBB HTTP {e.code}: {error_body}")

    if not result.get("success"):
        raise RuntimeError(
            "ImgBB rechazó la imagen:\n" + json.dumps(result, ensure_ascii=False)
        )

    return result["data"]["url"]


# ============================================================
# WAVESPEED IMAGE GENERATION
# ============================================================

def generate_image(prompt, aspect_ratio, resolution, output_format,
                    prompt_optimization_mode, api_key):
    payload = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "output_format": output_format,
        "prompt_optimization_mode": prompt_optimization_mode
    }

    response = wavespeed_request(WAVESPEED_IMAGE_URL, payload, api_key)
    prediction_id = get_prediction_id(response)
    result = poll_prediction(prediction_id, api_key)
    output = get_first_output(result)

    if not output:
        raise RuntimeError(
            "Seedream terminó pero no devolvió una URL de imagen.\n"
            + json.dumps(result, ensure_ascii=False)
        )

    return prediction_id, output, result


# ============================================================
# WAVESPEED IMAGE EDIT
# ============================================================

def edit_image(prompt, image_urls, aspect_ratio, resolution, output_format,
               prompt_optimization_mode, api_key):
    payload = {
        "prompt": prompt,
        "images": image_urls,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "output_format": output_format,
        "prompt_optimization_mode": prompt_optimization_mode
    }

    response = wavespeed_request(WAVESPEED_EDIT_URL, payload, api_key)
    prediction_id = get_prediction_id(response)
    result = poll_prediction(prediction_id, api_key)
    output = get_first_output(result)

    if not output:
        raise RuntimeError(
            "Seedream Edit terminó pero no devolvió una URL de imagen.\n"
            + json.dumps(result, ensure_ascii=False)
        )

    return prediction_id, output, result


# ============================================================
# WAVESPEED VIDEO
# ============================================================

def generate_video(prompt, image_url, resolution, aspect_ratio, duration,
                    enable_audio, enable_prompt_expansion, api_key):
    payload = {
        "prompt": prompt,
        "reference_images": [image_url],
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
        "enable_audio": enable_audio,
        "enable_prompt_expansion": enable_prompt_expansion
    }

    response = wavespeed_request(WAVESPEED_VIDEO_URL, payload, api_key)
    prediction_id = get_prediction_id(response)
    result = poll_prediction(prediction_id, api_key)
    outputs = result.get("outputs", [])

    return prediction_id, outputs, result


# ============================================================
# HTTP HANDLER
# ============================================================

class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print("[HTTP]", format % args)

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    def do_GET(self):

        if self.path == "/api/status":
            config = load_config()

            send_json(self, {
                "success": True,
                "wavespeed_configured": bool(config.get("wavespeed_api_key")),
                "imgbb_configured": bool(os.environ.get("IMGBB_API_KEY"))
            })

            return

        if self.path == "/":
            path = WEB_DIR / "index.html"
        else:
            requested = self.path.split("?", 1)[0].lstrip("/")
            path = WEB_DIR / requested

        if not path.exists() or not path.is_file():
            self.send_error(404, "Archivo no encontrado")
            return

        content_type, _ = mimetypes.guess_type(str(path))

        if not content_type:
            content_type = "application/octet-stream"

        data = path.read_bytes()

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    def do_POST(self):

        try:

            # =================================================
            # SAVE WAVESPEED KEY
            # =================================================

            if self.path == "/api/save-key":
                body = read_body(self)
                data = json.loads(body.decode("utf-8"))
                api_key = data.get("api_key", "").strip()

                if not api_key:
                    send_json(self, {
                        "success": False,
                        "error": "API key vacía."
                    }, 400)
                    return

                config = load_config()
                config["wavespeed_api_key"] = api_key
                config.pop("imgbb_api_key", None)
                save_config(config)

                send_json(self, {"success": True})
                return

            # =================================================
            # DELETE WAVESPEED KEY
            # =================================================

            if self.path == "/api/delete-key":
                config = load_config()
                config.pop("wavespeed_api_key", None)
                config.pop("imgbb_api_key", None)
                save_config(config)

                send_json(self, {"success": True})
                return

            # =================================================
            # GENERATE IMAGE
            # =================================================

            if self.path == "/api/generate-image":
                body = read_body(self)
                data = json.loads(body.decode("utf-8"))

                prompt = data.get("prompt", "").strip()
                aspect_ratio = data.get("aspect_ratio", "9:16")
                resolution = data.get("resolution", "1k")
                output_format = data.get("output_format", "jpeg")
                prompt_optimization_mode = data.get(
                    "prompt_optimization_mode", "standard"
                )

                config = load_config()
                api_key = config.get("wavespeed_api_key")

                if not api_key:
                    raise RuntimeError("Primero configura tu WaveSpeed API Key.")

                if not prompt:
                    raise RuntimeError("El prompt de imagen está vacío.")

                allowed_ratios = {
                    "1:1", "1:2", "2:1", "1:3", "3:1", "2:3", "3:2",
                    "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "9:21", "21:9"
                }

                if aspect_ratio not in allowed_ratios:
                    raise RuntimeError("Relación de aspecto no válida.")

                if resolution not in {"1k", "1.5k", "2k"}:
                    raise RuntimeError("Resolución no válida.")

                if output_format not in {"jpeg", "png"}:
                    raise RuntimeError("Formato de imagen no válido.")

                if prompt_optimization_mode not in {"standard", "fast"}:
                    raise RuntimeError("Modo de optimización no válido.")

                prediction_id, image_url, result = generate_image(
                    prompt, aspect_ratio, resolution, output_format,
                    prompt_optimization_mode, api_key
                )

                send_json(self, {
                    "success": True,
                    "prediction_id": prediction_id,
                    "image_url": image_url,
                    "outputs": result.get("outputs", [])
                })

                return

            # =================================================
            # EDIT IMAGE
            # =================================================

            if self.path == "/api/edit-image":
                body = read_body(self)
                fields, files = parse_multipart(self, body)

                prompt = fields.get("prompt", "").strip()
                aspect_ratio = fields.get("aspect_ratio", "9:16")
                resolution = fields.get("resolution", "1k")
                output_format = fields.get("output_format", "jpeg")
                prompt_optimization_mode = fields.get(
                    "prompt_optimization_mode", "standard"
                )

                if not prompt:
                    raise RuntimeError("El prompt de edición está vacío.")

                if not files:
                    raise RuntimeError("Debes seleccionar al menos una imagen.")

                if len(files) > 10:
                    raise RuntimeError(
                        "Seedream permite hasta 10 imágenes de referencia."
                    )

                config = load_config()
                api_key = config.get("wavespeed_api_key")
                imgbb_key = os.environ.get("IMGBB_API_KEY")

                if not api_key:
                    raise RuntimeError("Primero configura tu WaveSpeed API Key.")

                if not imgbb_key:
                    raise RuntimeError(
                        "Falta el secreto IMGBB_API_KEY en Codespaces."
                    )

                image_urls = []

                for file_info in files:
                    image_url = upload_to_imgbb(
                        file_info["data"], file_info["filename"], imgbb_key
                    )
                    image_urls.append(image_url)

                prediction_id, image_url, result = edit_image(
                    prompt, image_urls, aspect_ratio, resolution,
                    output_format, prompt_optimization_mode, api_key
                )

                send_json(self, {
                    "success": True,
                    "prediction_id": prediction_id,
                    "image_url": image_url,
                    "reference_images": image_urls,
                    "outputs": result.get("outputs", [])
                })

                return

            # =================================================
            # GENERATE VIDEO
            # =================================================

            if self.path == "/api/generate":
                body = read_body(self)
                fields, files = parse_multipart(self, body)

                prompt = fields.get("prompt", "").strip()
                resolution = fields.get("resolution", "720p")
                aspect_ratio = fields.get("aspect_ratio", "9:16")
                duration = int(fields.get("duration", "21"))
                enable_audio = fields.get("enable_audio", "true").lower() == "true"
                enable_prompt_expansion = (
                    fields.get("enable_prompt_expansion", "false").lower() == "true"
                )
                generated_image_url = fields.get("image_url", "").strip()

                config = load_config()
                api_key = config.get("wavespeed_api_key")
                imgbb_key = os.environ.get("IMGBB_API_KEY")

                if not api_key:
                    raise RuntimeError("Primero configura tu WaveSpeed API Key.")

                if not prompt:
                    raise RuntimeError("El prompt está vacío.")

                if resolution not in {"480p", "720p", "1080p"}:
                    raise RuntimeError("Resolución no válida.")

                if aspect_ratio not in {"16:9", "9:16", "1:1", "4:3", "3:4"}:
                    raise RuntimeError("Relación de aspecto no válida.")

                if duration < 2 or duration > 30:
                    raise RuntimeError(
                        "La duración debe estar entre 2 y 30 segundos."
                    )

                # ---------------------------------------------
                # Caso A: imagen generada/editada previamente
                # ---------------------------------------------

                image_url = generated_image_url

                # ---------------------------------------------
                # Caso B: imagen subida manualmente
                # ---------------------------------------------

                if not image_url:
                    if not files:
                        raise RuntimeError(
                            "Debes seleccionar una imagen de referencia."
                        )

                    if not imgbb_key:
                        raise RuntimeError(
                            "Falta IMGBB_API_KEY en Codespaces."
                        )

                    first_file = files[0]
                    image_url = upload_to_imgbb(
                        first_file["data"], first_file["filename"], imgbb_key
                    )

                # ---------------------------------------------
                # Generar video
                # ---------------------------------------------

                prediction_id, outputs, result = generate_video(
                    prompt, image_url, resolution, aspect_ratio, duration,
                    enable_audio, enable_prompt_expansion, api_key
                )

                send_json(self, {
                    "success": True,
                    "prediction_id": prediction_id,
                    "image_url": image_url,
                    "outputs": outputs
                })

                return

            # =================================================
            # UNKNOWN ENDPOINT
            # =================================================

            send_json(self, {
                "success": False,
                "error": "Endpoint no encontrado."
            }, 404)

        except Exception as e:
            print("\n[ERROR]", repr(e))

            send_json(self, {
                "success": False,
                "error": str(e)
            }, 500)


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(" WaveSpeed AI Frontend")
    print("=" * 60)
    print(f" Server: http://{HOST}:{PORT}")
    print()
    print(" Modelos:")
    print(" - Seedream V5.0 Pro")
    print(" - Seedream V5.0 Pro Edit")
    print(" - Wan 3.0 Reference-to-Video")
    print()
    print(" ImgBB:")
    print(" - IMGBB_API_KEY desde Codespaces Secrets")
    print("=" * 60)
    print()

    server = HTTPServer((HOST, PORT), Handler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        server.server_close()
