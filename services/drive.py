"""
Google Drive integration for the client onboarding pipeline.

Authentication uses a Google **service account** — the cleanest path for an
unattended backend that creates folders & uploads files on its own.

Setup (one-time):
  1. Create a service account in Google Cloud → IAM → Service Accounts.
  2. Enable the Drive API on the project.
  3. Download the JSON key, point GDRIVE_SERVICE_ACCOUNT_FILE at it.
  4. (Optional) Share a parent folder in your Drive with the service account's
     email so created folders show up there — otherwise they live in the
     service account's own My Drive.

Environment variables:
  GDRIVE_SERVICE_ACCOUNT_FILE   absolute path to the JSON key
  GDRIVE_SERVICE_ACCOUNT_JSON   alternative: raw JSON contents
  GDRIVE_PARENT_FOLDER_ID       (optional) parent folder for new client folders

If the libs aren't installed or no credentials are configured the helper
falls back to **simulation mode**: it returns deterministic fake folder
links and stores uploaded bytes under ``./onboarding_uploads/<token>/``.
This keeps the whole onboarding pipeline demoable end-to-end without any
cloud setup, while the production code path is preserved verbatim.
"""

from __future__ import annotations

import io
import json
import os
import threading
from pathlib import Path
from typing import Optional

# Optional Google client. Import lazily inside _client() so missing libs
# don't break the rest of the app.
_GDRIVE_OK = False
_GDRIVE_ERROR: Optional[str] = None
try:
    from google.oauth2 import service_account as _sa
    from googleapiclient.discovery import build as _g_build
    from googleapiclient.http import MediaIoBaseUpload as _MediaIoBaseUpload
    _GDRIVE_OK = True
except Exception as _exc:  # pragma: no cover
    _GDRIVE_ERROR = f"Google Drive libs not installed ({_exc!s})"


SCOPES = ["https://www.googleapis.com/auth/drive"]

SERVICE_ACCOUNT_FILE = os.getenv("GDRIVE_SERVICE_ACCOUNT_FILE", "").strip()
SERVICE_ACCOUNT_JSON = os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON", "").strip()
PARENT_FOLDER_ID     = os.getenv("GDRIVE_PARENT_FOLDER_ID", "").strip() or None

LOCAL_FALLBACK_DIR = Path(os.getenv(
    "ONBOARDING_LOCAL_DIR",
    str(Path(__file__).resolve().parent.parent / "onboarding_uploads"),
))
LOCAL_FALLBACK_DIR.mkdir(parents=True, exist_ok=True)

_service = None
_service_lock = threading.Lock()
_init_error: Optional[str] = _GDRIVE_ERROR


def _client():
    """Return a cached Drive v3 client, or None if not configured."""
    global _service, _init_error
    if _service is not None:
        return _service
    if not _GDRIVE_OK:
        return None
    creds = None
    try:
        with _service_lock:
            if SERVICE_ACCOUNT_JSON:
                info = json.loads(SERVICE_ACCOUNT_JSON)
                creds = _sa.Credentials.from_service_account_info(info, scopes=SCOPES)
            elif SERVICE_ACCOUNT_FILE and Path(SERVICE_ACCOUNT_FILE).is_file():
                creds = _sa.Credentials.from_service_account_file(
                    SERVICE_ACCOUNT_FILE, scopes=SCOPES,
                )
            else:
                _init_error = (
                    "GDRIVE_SERVICE_ACCOUNT_FILE / GDRIVE_SERVICE_ACCOUNT_JSON "
                    "not set — running in simulation mode."
                )
                return None
            _service = _g_build("drive", "v3", credentials=creds, cache_discovery=False)
            return _service
    except Exception as exc:
        _init_error = f"Google Drive auth failed: {exc!s}"
        return None


def status() -> dict:
    """Health snapshot for the UI."""
    cli = _client()
    return {
        "configured": cli is not None,
        "libs_installed": _GDRIVE_OK,
        "parent_folder_id": PARENT_FOLDER_ID,
        "fallback_dir": str(LOCAL_FALLBACK_DIR),
        "error": _init_error,
    }


# ───────────────────────────── public API ───────────────────────────────────

def create_client_folder(client_name: str, token: str) -> dict:
    """
    Create a Drive folder named ``{client_name}_{token}``.

    Returns a dict with: ``id``, ``url``, ``name``, ``simulated`` (bool).
    """
    safe_name = "".join(c for c in (client_name or "Client") if c.isalnum() or c in " _-")
    safe_name = safe_name.strip().replace(" ", "_") or "Client"
    folder_name = f"{safe_name}_{token}"

    cli = _client()
    if cli is None:
        (LOCAL_FALLBACK_DIR / token).mkdir(parents=True, exist_ok=True)
        return {
            "id": f"sim-{token}",
            "name": folder_name,
            "url": f"https://drive.google.com/drive/folders/sim-{token}",
            "simulated": True,
        }

    metadata = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder"}
    if PARENT_FOLDER_ID:
        metadata["parents"] = [PARENT_FOLDER_ID]
    folder = cli.files().create(
        body=metadata,
        fields="id, name, webViewLink",
        supportsAllDrives=True,
    ).execute()
    fid = folder["id"]

    cli.permissions().create(
        fileId=fid,
        body={"role": "reader", "type": "anyone"},
        fields="id",
        supportsAllDrives=True,
    ).execute()

    return {
        "id": fid,
        "name": folder.get("name", folder_name),
        "url": folder.get("webViewLink") or f"https://drive.google.com/drive/folders/{fid}",
        "simulated": False,
    }


def upload_bytes_to_folder(
    *, folder_id: str, token: str, filename: str,
    data: bytes, mime_type: str = "application/octet-stream",
) -> dict:
    """
    Upload ``data`` as ``filename`` into ``folder_id``.

    Returns: ``id``, ``url``, ``filename``, ``size``, ``simulated``.
    """
    safe_name = Path(filename).name or "upload.bin"

    if folder_id.startswith("sim-") or _client() is None:
        dest_dir = LOCAL_FALLBACK_DIR / token
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / safe_name
        dest.write_bytes(data)
        return {
            "id": f"sim-file-{safe_name}",
            "filename": safe_name,
            "url": f"file://{dest}",
            "size": len(data),
            "simulated": True,
            "local_path": str(dest),
        }

    cli = _client()
    media = _MediaIoBaseUpload(
        io.BytesIO(data),
        mimetype=mime_type or "application/octet-stream",
        resumable=False,
    )
    file = cli.files().create(
        body={"name": safe_name, "parents": [folder_id]},
        media_body=media,
        fields="id, name, webViewLink, size",
        supportsAllDrives=True,
    ).execute()
    fid = file["id"]
    return {
        "id": fid,
        "filename": file.get("name", safe_name),
        "url": file.get("webViewLink") or f"https://drive.google.com/file/d/{fid}/view",
        "size": int(file.get("size") or len(data)),
        "simulated": False,
        "local_path": None,
    }
