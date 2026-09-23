import json
import os
import re
import time
import threading
import uuid
import mimetypes
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import unquote
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

HOST = "0.0.0.0"
PORT = 8000

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

# Tiempo máximo (segundos) esperando el resultado de una predicción
POLL_TIMEOUT = 900

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
# WAVESPEED API KEY (por petición)
# ============================================================
# El servidor NO guarda la key. Cada usuario la guarda en el
# localStorage de su navegador y la envía en la cabecera
# X-WaveSpeed-Key con cada petición.

WAVESPEED_KEY_HEADER = "X-WaveSpeed-Key"


def get_wavespeed_key(handler):
    api_key = handler.headers.get(WAVESPEED_KEY_HEADER, "").strip()

    if not api_key:
        raise RuntimeError("Primero configura tu WaveSpeed API Key.")

    return api_key


# ============================================================
# HTTP HELPERS
# ============================================================

def send_json(handler, data, status=200):
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")

    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        # El cliente (o el proxy) ya cerró la conexión; no hay a quién responder.
        pass


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
    started = time.time()

    while True:
        if time.time() - started > POLL_TIMEOUT:
            raise RuntimeError(
                f"Tiempo de espera agotado ({POLL_TIMEOUT}s) esperando "
                f"a WaveSpeed. Prediction ID: {prediction_id}"
            )

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

        except URLError as e:
            raise RuntimeError(f"Network error: {e}")

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

_NAME_RE = re.compile(r'(?:^|;)\s*name="([^"]*)"')
_FILENAME_RE = re.compile(r'(?:^|;)\s*filename="([^"]*)"')


def parse_multipart(handler, body):
    content_type = handler.headers.get("Content-Type", "")

    if "boundary=" not in content_type:
        raise RuntimeError("No se encontró boundary multipart.")

    boundary = content_type.split("boundary=", 1)[1].split(";", 1)[0].strip()

    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]

    boundary_bytes = b"--" + boundary.encode()

    fields = {}
    files = []

    parts = body.split(boundary_bytes)

    for part in parts:
        if not part:
            continue

        # Cierre del multipart ("--\r\n"): no hay más partes
        if part.startswith(b"--"):
            continue

        # Quitar SOLO el salto de línea que rodea a cada parte,
        # sin tocar el contenido (antes se recortaban guiones y
        # saltos de línea reales del contenido).
        if part.startswith(b"\r\n"):
            part = part[2:]

        if part.endswith(b"\r\n"):
            part = part[:-2]

        if not part:
            continue

        if b"\r\n\r\n" not in part:
            continue

        header_data, content = part.split(b"\r\n\r\n", 1)
        headers = header_data.decode("utf-8", errors="ignore")

        disposition = None

        for line in headers.split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                disposition = line.split(":", 1)[1].strip()

        if not disposition:
            continue

        name_match = _NAME_RE.search(disposition)
        filename_match = _FILENAME_RE.search(disposition)

        if not name_match:
            continue

        name = name_match.group(1)
        filename = filename_match.group(1) if filename_match else None

        if filename is not None and filename != "":
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

        elif filename is None:
            fields[name] = content.decode("utf-8", errors="ignore")

    return fields, files


# ============================================================
# IMGBB
# ============================================================

def upload_to_imgbb(image_bytes, filename, api_key,
                    content_type="image/jpeg"):
    if not api_key:
        raise RuntimeError(
            "No existe IMGBB_API_KEY en los secretos de Codespaces."
        )

    if not content_type.startswith("image/"):
        content_type = "image/jpeg"

    safe_filename = filename.replace('"', "").replace("\r", "").replace("\n", "")

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
            f'filename="{safe_filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
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

    except URLError as e:
        raise RuntimeError(f"Network error (ImgBB): {e}")

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
# BACKGROUND JOBS
# ============================================================
# Las generaciones pueden tardar minutos y el proxy de Codespaces
# (o el móvil) puede cortar la conexión. Por eso el POST devuelve un
# job_id al instante y el navegador consulta /api/job/<id> hasta que
# termina. La API key solo existe en memoria durante el trabajo.

JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_TTL = 3600


def start_job(work):
    now = time.time()
    job_id = uuid.uuid4().hex

    with JOBS_LOCK:
        for old_id in [j for j, v in JOBS.items() if now - v["created"] > JOB_TTL]:
            del JOBS[old_id]

        JOBS[job_id] = {"status": "running", "created": now}

    def runner():
        try:
            update = {"status": "done", "result": work()}
        except Exception as e:
            print("\n[ERROR]", repr(e))
            update = {"status": "error", "error": str(e)}

        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id].update(update)

    threading.Thread(target=runner, daemon=True).start()
    return job_id


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

        if self.path.split("?", 1)[0] == "/api/status":
            send_json(self, {
                "success": True,
                "imgbb_configured": bool(os.environ.get("IMGBB_API_KEY"))
            })

            return

        job_prefix = "/api/job/"
        job_path = self.path.split("?", 1)[0]

        if job_path.startswith(job_prefix):
            with JOBS_LOCK:
                job = dict(JOBS.get(job_path[len(job_prefix):]) or {})

            if not job:
                send_json(self, {
                    "success": False,
                    "error": "Trabajo no encontrado (¿se reinició el servidor?)."
                }, 404)
                return

            send_json(self, {
                "success": True,
                "status": job["status"],
                "result": job.get("result"),
                "error": job.get("error")
            })

            return

        # ----------------------------------------------------
        # Archivos estáticos (protegido contra path traversal)
        # ----------------------------------------------------

        web_root = WEB_DIR.resolve()
        raw_path = self.path.split("?", 1)[0]

        if raw_path == "/":
            requested = "index.html"
        else:
            requested = unquote(raw_path).lstrip("/")

        try:
            path = (web_root / requested).resolve()
        except (OSError, ValueError):
            self.send_error(404, "Archivo no encontrado")
            return

        if not path.is_relative_to(web_root) or not path.is_file():
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

                api_key = get_wavespeed_key(self)

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

                def work():
                    prediction_id, image_url, result = generate_image(
                        prompt, aspect_ratio, resolution, output_format,
                        prompt_optimization_mode, api_key
                    )

                    return {
                        "success": True,
                        "prediction_id": prediction_id,
                        "image_url": image_url,
                        "outputs": result.get("outputs", [])
                    }

                send_json(self, {"success": True, "job_id": start_job(work)})
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

                api_key = get_wavespeed_key(self)
                imgbb_key = os.environ.get("IMGBB_API_KEY")

                if not imgbb_key:
                    raise RuntimeError(
                        "Falta el secreto IMGBB_API_KEY en Codespaces."
                    )

                def work():
                    image_urls = []

                    for file_info in files:
                        image_urls.append(upload_to_imgbb(
                            file_info["data"],
                            file_info["filename"],
                            imgbb_key,
                            file_info["content_type"]
                        ))

                    prediction_id, image_url, result = edit_image(
                        prompt, image_urls, aspect_ratio, resolution,
                        output_format, prompt_optimization_mode, api_key
                    )

                    return {
                        "success": True,
                        "prediction_id": prediction_id,
                        "image_url": image_url,
                        "reference_images": image_urls,
                        "outputs": result.get("outputs", [])
                    }

                send_json(self, {"success": True, "job_id": start_job(work)})
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
                enable_audio = fields.get("enable_audio", "true").lower() == "true"
                enable_prompt_expansion = (
                    fields.get("enable_prompt_expansion", "false").lower() == "true"
                )
                generated_image_url = fields.get("image_url", "").strip()

                try:
                    duration = int(fields.get("duration", "21"))
                except ValueError:
                    raise RuntimeError("Duración no válida.")

                api_key = get_wavespeed_key(self)
                imgbb_key = os.environ.get("IMGBB_API_KEY")

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

                # ---------------------------------------------
                # Subir (si hace falta) y generar video
                # ---------------------------------------------

                def work():
                    final_url = image_url

                    if not final_url:
                        final_url = upload_to_imgbb(
                            first_file["data"],
                            first_file["filename"],
                            imgbb_key,
                            first_file["content_type"]
                        )

                    prediction_id, outputs, result = generate_video(
                        prompt, final_url, resolution, aspect_ratio, duration,
                        enable_audio, enable_prompt_expansion, api_key
                    )

                    return {
                        "success": True,
                        "prediction_id": prediction_id,
                        "image_url": final_url,
                        "outputs": outputs
                    }

                send_json(self, {"success": True, "job_id": start_job(work)})
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
    print(" WaveSpeed:")
    print(" - API key en el navegador de cada usuario (no se guarda aquí)")
    print()
    print(" ImgBB:")
    print(" - IMGBB_API_KEY desde Codespaces Secrets")
    print("=" * 60)
    print()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        server.server_close()
