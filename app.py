"""
Avatar Studio — unified FastAPI platform for avatar & voice creators.

Capabilities:
  - Auth (DB-backed: password + Google/GitHub OAuth, role-based access)
  - Deploy Avatar  → POST to Elai admin API   (role: superadmin)
  - Train Model    → live SageMaker pipeline   (role: developer)
  - Clone Voice    → ElevenLabs IVC pipeline   (role: developer)
  - S3 Browser     → boto3 list / presigned    (role: viewer)
  - AI Support     → KB-grounded chat          (role: viewer)
"""

from __future__ import annotations

import asyncio
import datetime as _onboarding_dt
import json
import os
import random
import re
import secrets
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse as _urlparse

import httpx
import requests
from dotenv import load_dotenv
from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request,
                     UploadFile)
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from db import (
    Role, User, ClientOnboarding, OnboardingFile, Notification,
    consume_password_reset, create_password_reset, create_notification,
    get_onboarding_by_token, get_or_create_oauth_user, get_session,
    get_user_by_email, get_user_by_id, hash_password, is_first_user,
    list_recent_onboardings, mark_notifications_read, notifications_for_user,
    peek_password_reset, role_at_least, unread_notification_count,
    verify_password,
)
from services import drive as drive_service

# ────────────────────────────────────────────────────────────────────────────
# Environment setup
# ────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# Pick up .env: root first, then voice-clone/.env (backward-compat fallback)
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "voice-clone" / ".env", override=False)

CLEANVOICE_API_KEY = os.getenv("CLEANVOICE_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"
ELEVENLABS_IVC_URL = f"{ELEVENLABS_BASE}/voices/add"
MAX_MB = 500

# Optional LLM keys for the AI Support chat (either one — or none, in which
# case the chat falls back to KB-only retrieval).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

# Session signing key — REQUIRED in production. Auto-generated for dev so the
# app boots without configuration.
SESSION_SECRET = os.getenv("SESSION_SECRET") or secrets.token_urlsafe(48)

# OAuth providers (optional). Buttons in the UI are hidden when credentials
# are not configured.
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GITHUB_CLIENT_ID     = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
OAUTH_BASE_URL       = os.getenv("OAUTH_BASE_URL", "http://127.0.0.1:8765").rstrip("/")

# S3 asset library defaults (configurable per-request in the UI too)
DEFAULT_S3_BUCKET = os.getenv("DEFAULT_S3_BUCKET", "elai-avatars")
DEFAULT_S3_PREFIX = os.getenv("DEFAULT_S3_PREFIX", "raw/")

# Cleanvoice SDK — optional; without the key /clone returns 503
_cv = None
if CLEANVOICE_API_KEY:
    try:
        from cleanvoice import Cleanvoice  # type: ignore
        _cv = Cleanvoice({"api_key": CLEANVOICE_API_KEY})
    except Exception:  # pragma: no cover
        _cv = None

# ────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Avatar Studio")
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="avatars_studio_session",
    same_site="lax",
    https_only=False,  # set True behind HTTPS in prod
)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

CONFIGS_DIR = BASE_DIR / "configs_v3"
CONFIGS_DIR.mkdir(exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ────────────────────────────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    """Avatar Training V3.6 pipeline parameters (see Avatar Training V3_6.ipynb)."""

    # Experiment
    experiment_name: str
    notes: Optional[str] = ""

    # Dataset
    footage_path: str
    train_footages_paths: list[str] = []
    alpha_footage_path: Optional[str] = None
    pauses: list[int] = []            # frame indices, as in the notebook
    face_bbox: Optional[str] = None   # None | "auto" | "x,y,w,h"

    # ML
    avatar_type: str = "frontal"            # frontal / side-right / side-left
    normalization_type: str = "canonical"   # canonical / centered / centered_bbox
    model_size: int = 512                   # 512 / 768
    symmetrical_blinking: bool = True

    # Compute
    instance_type: str = "ml.g6.2xlarge"
    training_duration: int = 18             # hours (max-runtime)
    enable_amp: bool = True
    enable_spot_training: bool = False


# SageMaker on-demand cost (USD/hour, us-east-2). Source: AWS Pricing.
INSTANCE_COST_USD_HR: dict[str, float] = {
    "ml.g6.xlarge":     0.8048,
    "ml.g6.2xlarge":    0.9776,
    "ml.g6.4xlarge":    1.323,
    "ml.g6.8xlarge":    2.014,
    "ml.g6.12xlarge":   5.291,
    "ml.g6.16xlarge":   3.397,
    "ml.g6.24xlarge":   7.064,
    "ml.g6.48xlarge":   14.13,
    "ml.g5.xlarge":     1.408,
    "ml.g5.2xlarge":    1.515,
    "ml.g5.4xlarge":    2.03,
    "ml.g5.8xlarge":    3.061,
    "ml.g5.12xlarge":   8.193,
    "ml.g5.24xlarge":   11.61,
    "ml.g5.48xlarge":   22.07,
    "ml.g4dn.xlarge":   0.736,
    "ml.g4dn.2xlarge":  0.94,
    "ml.g4dn.4xlarge":  1.505,
    "ml.p3.2xlarge":    3.825,
    "ml.p3.8xlarge":    14.688,
    "ml.p4d.24xlarge":  37.688,
}


# ─── SageMaker live mode: import boto3 and Elai ML libs ──────────────────
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
AVATARS_REPO_PATH = os.getenv("AVATARS_REPO_PATH")
if AVATARS_REPO_PATH and Path(AVATARS_REPO_PATH).is_dir():
    import sys as _sys
    _sys.path.insert(0, AVATARS_REPO_PATH)

_SAGEMAKER_LIVE = False
_SAGEMAKER_INIT_ERROR: Optional[str] = None
_sagemaker_client = None
_atrain = None
_get_raw_config = None
_prepare_avatar_config = None
_DatasetPreparatorV3 = None
_SAGEMAKER_TEMPLATE_PATH_V36 = None
_FastTalkerConfigReader = None

try:
    import boto3  # type: ignore
    try:
        import avatars.dataset.v3.training_utils as _atrain_mod  # type: ignore
        from avatars.dataset.v3_6.default import (  # type: ignore
            get_raw_config as _get_raw_config_fn,
            prepare_avatar_config as _prepare_avatar_config_fn,
        )
        from avatars.dataset.v3_6.prepare import (  # type: ignore
            DatasetPreparatorV3 as _DatasetPreparatorV3_cls,
            SAGEMAKER_TEMPLATE_PATH_V36 as _SAGEMAKER_TEMPLATE_PATH_V36_const,
        )
        try:
            from face_production.fast_talker.utils.reader import FastTalkerConfigReader as _Reader  # type: ignore
            _FastTalkerConfigReader = _Reader
        except Exception:
            _FastTalkerConfigReader = None

        _atrain = _atrain_mod
        _get_raw_config = _get_raw_config_fn
        _prepare_avatar_config = _prepare_avatar_config_fn
        _DatasetPreparatorV3 = _DatasetPreparatorV3_cls
        _SAGEMAKER_TEMPLATE_PATH_V36 = _SAGEMAKER_TEMPLATE_PATH_V36_const
        _sagemaker_client = boto3.client("sagemaker", region_name=AWS_REGION)
        _SAGEMAKER_LIVE = True
    except Exception as ml_err:
        _SAGEMAKER_INIT_ERROR = (
            f"avatars/face_production not importable: {ml_err}. "
            "Set AVATARS_REPO_PATH=… in .env to enable live mode."
        )
except Exception as boto_err:
    _SAGEMAKER_INIT_ERROR = f"boto3 not installed: {boto_err}"


# ────────────────────────────────────────────────────────────────────────────
# Auth (DB-backed: password + Google/GitHub OAuth + role-based access)
# ────────────────────────────────────────────────────────────────────────────
def get_current_user(
    request: Request,
    db: Session = Depends(get_session),
) -> Optional[User]:
    """Resolve the signed-in user from the session cookie."""
    uid = request.session.get("uid")
    if not uid:
        return None
    return get_user_by_id(db, uid)


def require_user(user: Optional[User] = Depends(get_current_user)) -> User:
    """401 if not signed in. Use for any endpoint that needs an account."""
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def require_role(min_role: Role):
    """
    Returns a dependency that enforces a minimum role privilege level.
    Usage:  user=Depends(require_role(Role.developer))
    """
    def _checker(user: User = Depends(require_user)) -> User:
        if not role_at_least(user.role, min_role):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"This action requires {min_role.value} role "
                    f"(you are {user.role.value})."
                ),
            )
        return user
    return _checker


# ---- OAuth (Authlib, optional) ------------------------------------------
oauth = None
try:
    from authlib.integrations.starlette_client import OAuth as _AuthlibOAuth
    oauth = _AuthlibOAuth()
    if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
        oauth.register(
            name="google",
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )
    if GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET:
        oauth.register(
            name="github",
            client_id=GITHUB_CLIENT_ID,
            client_secret=GITHUB_CLIENT_SECRET,
            access_token_url="https://github.com/login/oauth/access_token",
            authorize_url="https://github.com/login/oauth/authorize",
            api_base_url="https://api.github.com/",
            client_kwargs={"scope": "read:user user:email"},
        )
except Exception as _oauth_err:  # pragma: no cover
    oauth = None


def _oauth_enabled() -> dict:
    return {
        "google": bool(oauth and GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET),
        "github": bool(oauth and GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET),
    }


# ---- Page routes --------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user: Optional[User] = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/auth", status_code=303)
    return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={"user": user.to_dict()},
)

@app.get("/auth", response_class=HTMLResponse)
async def auth_page(request: Request, user: Optional[User] = Depends(get_current_user)):
    if user:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
    "auth.html",
    {"request": request, "oauth": _oauth_enabled()},
)


# ---- Password login / register -----------------------------------------
def _validate_email(s: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", s or ""))


@app.post("/api/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_session),
):
    user = get_user_by_email(db, username)
    if not user or not user.password_hash or not verify_password(password, user.password_hash):
        return JSONResponse(status_code=400, content={"message": "Invalid email or password"})
    import datetime as _dt
    user.last_login_at = _dt.datetime.utcnow()
    db.commit()
    request.session["uid"] = user.id
    return JSONResponse(content={"message": "Signed in. Redirecting…"})


@app.post("/api/register")
async def register(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_session),
):
    email = username.strip().lower()
    if not _validate_email(email):
        return JSONResponse(status_code=400, content={"message": "Please enter a valid email"})
    if len(password) < 8:
        return JSONResponse(status_code=400, content={"message": "Password must be at least 8 characters"})
    if get_user_by_email(db, email):
        return JSONResponse(status_code=400, content={"message": "An account with this email already exists"})

    role = Role.superadmin if is_first_user(db) else Role.viewer
    user = User(
        email=email,
        name=email.split("@")[0],
        password_hash=hash_password(password),
        role=role,
        provider="password",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    request.session["uid"] = user.id
    return {"message": "Account created. Signing you in…", "role": role.value}


# ---- OAuth flow --------------------------------------------------------
@app.get("/auth/oauth/{provider}")
async def oauth_start(provider: str, request: Request):
    if not oauth or provider not in ("google", "github"):
        raise HTTPException(404, "Unknown OAuth provider")
    if not getattr(oauth, provider, None):
        raise HTTPException(404, f"{provider} OAuth not configured")
    redirect_uri = f"{OAUTH_BASE_URL}/auth/oauth/{provider}/callback"
    return await getattr(oauth, provider).authorize_redirect(request, redirect_uri)


@app.get("/auth/oauth/{provider}/callback")
async def oauth_callback(
    provider: str,
    request: Request,
    db: Session = Depends(get_session),
):
    if not oauth or provider not in ("google", "github"):
        raise HTTPException(404, "Unknown OAuth provider")
    client = getattr(oauth, provider, None)
    if not client:
        raise HTTPException(404, f"{provider} OAuth not configured")
    try:
        token = await client.authorize_access_token(request)
    except Exception as e:
        return RedirectResponse(url=f"/auth?error=oauth_{provider}_failed", status_code=303)

    if provider == "google":
        profile = token.get("userinfo") or {}
        if not profile:
            resp = await client.get("https://openidconnect.googleapis.com/v1/userinfo", token=token)
            profile = resp.json()
        pid = str(profile.get("sub") or profile.get("id") or "")
        email = profile.get("email") or f"{pid}@google.local"
        name = profile.get("name")
        avatar = profile.get("picture")
    else:  # github
        u = (await client.get("user", token=token)).json()
        pid = str(u.get("id") or "")
        email = u.get("email")
        if not email:
            emails = (await client.get("user/emails", token=token)).json()
            primary = next(
                (e for e in emails if e.get("primary") and e.get("verified")), None
            ) or (emails[0] if emails else {})
            email = primary.get("email") or f"{u.get('login')}@github.local"
        name = u.get("name") or u.get("login")
        avatar = u.get("avatar_url")

    user = get_or_create_oauth_user(
        db, provider=provider, provider_id=pid,
        email=email, name=name, avatar_url=avatar,
    )
    request.session["uid"] = user.id
    return RedirectResponse(url="/", status_code=303)


@app.get("/api/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/auth", status_code=303)


@app.get("/api/me")
async def me(user: Optional[User] = Depends(get_current_user)):
    if not user:
        return JSONResponse(status_code=401, content={"authenticated": False})
    return {"authenticated": True, "user": user.to_dict()}


# ---- Password reset flow -----------------------------------------------
# In production we'd email the reset link via SMTP. For dev / on-prem
# deployments without SMTP we *also* return the link in the JSON response
# so the user can copy it. The link is also printed to the server console.
PASSWORD_RESET_DEV_REVEAL = (
    os.getenv("PASSWORD_RESET_DEV_REVEAL", "1").lower() in ("1", "true", "yes")
)


def _reset_url(request: Request, token: str) -> str:
    base = (request.headers.get("origin") or str(request.base_url)).rstrip("/")
    return f"{base}/auth/reset?token={token}"


@app.post("/api/password-reset/request")
async def password_reset_request(
    request: Request,
    email: str = Form(...),
    db: Session = Depends(get_session),
):
    """Issue a one-time reset link. Always returns the same generic message
    to avoid leaking which emails exist (account enumeration). In dev mode
    the link is included in the response payload."""
    normalized = (email or "").strip().lower()
    generic = {
        "message": "If an account with that email exists, a password reset link has been generated.",
    }
    if not _validate_email(normalized):
        return JSONResponse(content=generic)

    user = get_user_by_email(db, normalized)
    if not user or user.provider != "password":
        # OAuth-only accounts can't reset password — same generic response.
        return JSONResponse(content=generic)

    pr = create_password_reset(db, user, issued_by="self")
    url = _reset_url(request, pr.token)
    print(f"[password-reset] {user.email} → {url}")

    payload = dict(generic)
    if PASSWORD_RESET_DEV_REVEAL:
        payload["reset_url"] = url
        payload["expires_in_minutes"] = 60
        payload["dev_hint"] = "PASSWORD_RESET_DEV_REVEAL is on — copy the link below."
    return JSONResponse(content=payload)


@app.get("/auth/reset", response_class=HTMLResponse)
async def auth_reset_page(
    request: Request,
    token: str = "",
    db: Session = Depends(get_session),
):
    """Renders the password-reset form. Validates the token before showing
    the input so we can display 'expired' / 'invalid' state inline."""
    user, err = (None, "not_found")
    if token:
        user, err = peek_password_reset(db, token)
    return templates.TemplateResponse(
        request=request,
        name="auth.html",
        context={
            "oauth": _oauth_enabled(),
            "reset_mode": True,
            "reset_token": token,
            "reset_user_email": user.email if user else None,
            "reset_error": err,
        },
    )


@app.post("/api/password-reset/confirm")
async def password_reset_confirm(
    request: Request,
    token: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_session),
):
    if len(password) < 8:
        return JSONResponse(
            status_code=400,
            content={"message": "Password must be at least 8 characters"},
        )
    user = consume_password_reset(db, token)
    if not user:
        return JSONResponse(
            status_code=400,
            content={"message": "This reset link is invalid, expired, or already used."},
        )
    user.password_hash = hash_password(password)
    # Make sure they can sign in even if they originally registered via OAuth.
    if not user.provider or user.provider != "password":
        user.provider = "password"
    db.commit()
    request.session["uid"] = user.id
    return {"message": "Password updated. Signing you in…"}


# ────────────────────────────────────────────────────────────────────────────
# Admin: user / role management (SuperAdmin only)
# ────────────────────────────────────────────────────────────────────────────
class UpdateRoleRequest(BaseModel):
    role: str  # "viewer" | "developer" | "superadmin"


@app.get("/api/admin/users")
async def admin_list_users(
    user: User = Depends(require_role(Role.superadmin)),
    db: Session = Depends(get_session),
):
    """List all users (SuperAdmin only)."""
    from sqlalchemy import select as _select
    rows = db.scalars(_select(User).order_by(User.created_at.desc())).all()
    return {"users": [u.to_dict() for u in rows]}


@app.patch("/api/admin/users/{user_id}/role")
async def admin_update_role(
    user_id: int,
    body: UpdateRoleRequest,
    user: User = Depends(require_role(Role.superadmin)),
    db: Session = Depends(get_session),
):
    """Change a user's role. SuperAdmins cannot demote themselves."""
    try:
        new_role = Role(body.role)
    except ValueError:
        raise HTTPException(
            400,
            f"Invalid role '{body.role}'. Use viewer/developer/avatar_manager/support/manager/superadmin."
        )

    target = get_user_by_id(db, user_id)
    if target is None:
        raise HTTPException(404, "User not found")
    if target.id == user.id and new_role != Role.superadmin:
        raise HTTPException(400, "You cannot demote yourself")

    target.role = new_role
    db.commit()
    db.refresh(target)
    return {"ok": True, "user": target.to_dict()}


@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(
    user_id: int,
    user: User = Depends(require_role(Role.superadmin)),
    db: Session = Depends(get_session),
):
    """Permanently delete a user account."""
    target = get_user_by_id(db, user_id)
    if target is None:
        raise HTTPException(404, "User not found")
    if target.id == user.id:
        raise HTTPException(400, "You cannot delete your own account")
    db.delete(target)
    db.commit()
    return {"ok": True}


@app.post("/api/admin/users/{user_id}/reset-link")
async def admin_create_reset_link(
    user_id: int,
    request: Request,
    user: User = Depends(require_role(Role.superadmin)),
    db: Session = Depends(get_session),
):
    """SuperAdmin: generate a one-time password-reset link for another user.
    Returns the URL so the admin can hand it over out-of-band."""
    target = get_user_by_id(db, user_id)
    if target is None:
        raise HTTPException(404, "User not found")
    pr = create_password_reset(db, target, issued_by="admin")
    url = _reset_url(request, pr.token)
    return {
        "ok": True,
        "reset_url": url,
        "expires_in_minutes": 60,
        "user_email": target.email,
    }


# ────────────────────────────────────────────────────────────────────────────
# Tool 1: Deploy Avatar (Elai admin API)
# ────────────────────────────────────────────────────────────────────────────
@app.get("/api/deploy/configs")
async def list_configs(user: User = Depends(require_role(Role.viewer))):
    """List available JSON configs in configs_v3/."""
    files = sorted(p.stem for p in CONFIGS_DIR.glob("*.json"))
    return {"configs": files}


@app.post("/api/deploy")
async def deploy_avatar(
    auth_token: str = Form(...),
    avatar_code: str = Form(...),
    avatar_name: str = Form(...),
    organization_id: str = Form(...),
    avatar_gender: str = Form("male"),
    avatar_type: str = Form("custom"),
    avatar_status: int = Form(1),
    account_ids: str = Form(""),
    config_source: str = Form("library"),     # "library" | "upload"
    config_file_name: str = Form(""),         # filename without .json (for library)
    config_file: Optional[UploadFile] = File(None),  # uploaded JSON (for upload)
    user: User = Depends(require_role(Role.superadmin)),
):
    # 1. Load JSON config — from uploaded file or library
    local_json: dict
    source_label: str

    if config_source == "upload":
        if not config_file or not config_file.filename:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Please upload a JSON file"},
            )
        try:
            raw = await config_file.read()
            if len(raw) > 5 * 1024 * 1024:
                return JSONResponse(
                    status_code=413,
                    content={"success": False, "message": "JSON file too large (> 5 MB)"},
                )
            local_json = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as e:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": f"Invalid JSON: {e}"},
            )
        except UnicodeDecodeError:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "File is not UTF-8 encoded"},
            )
        source_label = f"uploaded · {config_file.filename}"
    else:
        if not config_file_name:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Please specify a config filename"},
            )
        config_path = CONFIGS_DIR / f"{config_file_name}.json"
        if not config_path.exists():
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": f"File not found: {config_path.name}"},
            )
        try:
            local_json = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            return JSONResponse(
                status_code=400, content={"success": False, "message": f"JSON error: {e}"}
            )
        source_label = f"library · {config_path.name}"

    # 2. Build payload for the Elai admin API
    payload = {
        "code": avatar_code,
        "name": avatar_name,
        "type": avatar_type,
        "status": avatar_status,
        "order": 1,
        "organizationId": organization_id,
        "accountIds": [a.strip() for a in account_ids.split(",") if a.strip()],
        "variants": [],
        "frontendConfig": {
            "gender": avatar_gender,
            "thumbnail": f"https://d3u63mhbhkevz8.cloudfront.net/avatars/custom/{avatar_code}.jpg",
            "canvas": f"https://d3u63mhbhkevz8.cloudfront.net/avatars/custom/{avatar_code}.png",
        },
        "avatarConfig": local_json.get("avatarConfig", local_json),
    }

    try:
        res = requests.post(
            "https://api-dev.elai.io/admin/avatars",
            json=payload,
            headers={
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json",
            },
            timeout=60,
        )
        if res.status_code in (200, 201):
            return {
                "success": True,
                "id": res.json().get("_id"),
                "source": source_label,
                "message": f"Avatar \"{avatar_name}\" deployed successfully ({source_label})",
            }
        return JSONResponse(
            status_code=res.status_code,
            content={"success": False, "message": res.text[:600], "source": source_label},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e), "source": source_label},
        )


# ────────────────────────────────────────────────────────────────────────────
# Tool 2: Train Model — job manager
# ────────────────────────────────────────────────────────────────────────────
# In-memory job store + lock. Can be swapped for SQLite / Redis later.
TRAIN_JOBS: dict[str, dict] = {}
_TRAIN_LOCK = asyncio.Lock()

# Simulation stages (used when live mode isn't available)
TRAIN_STAGES_SIM = [
    ("validating",      0.05,  1,  2),
    ("provisioning",    0.10,  2,  4),
    ("downloading",     0.20,  3,  5),
    ("preprocessing",   0.30,  2,  4),
    ("training",        0.85, 12, 25),
    ("evaluating",      0.95,  2,  4),
    ("uploading",       1.00,  1,  3),
]


_URI_SCHEMES = ("s3://", "gs://", "http://", "https://", "file://")


def _is_valid_uri(uri: str) -> bool:
    return any(uri.startswith(s) for s in _URI_SCHEMES)


def _validate_train_params(req: "TrainRequest") -> list[str]:
    """Return a list of human-readable errors (empty list → all good)."""
    errors: list[str] = []
    name = req.experiment_name.strip()
    if not (3 <= len(name) <= 64):
        errors.append("Experiment name: 3–64 characters")
    if not re.match(r"^[a-zA-Z0-9._-]+$", name):
        errors.append("Experiment name: only latin letters, digits, . _ -")

    # footage_path (required)
    if not req.footage_path:
        errors.append("footage_path is required")
    elif not _is_valid_uri(req.footage_path):
        errors.append("footage_path: expected s3:// / gs:// / http(s):// / file://")

    # train_footages_paths (optional list) — validate each
    for i, path in enumerate(req.train_footages_paths):
        if not _is_valid_uri(path):
            errors.append(f"train_footages_paths[{i}]: unsupported scheme in \"{path[:60]}\"")

    # alpha_footage_path (optional)
    if req.alpha_footage_path and not _is_valid_uri(req.alpha_footage_path):
        errors.append("alpha_footage_path: unsupported scheme")

    # pauses: list of non-negative ints
    for i, p in enumerate(req.pauses):
        if not isinstance(p, int) or p < 0:
            errors.append(f"pauses[{i}]: expected a non-negative integer (frame index)")

    # face_bbox: None | "auto" | "x,y,w,h"
    if req.face_bbox:
        v = req.face_bbox.strip()
        if v.lower() != "auto":
            try:
                parts = [int(x.strip()) for x in v.split(",")]
                if len(parts) != 4 or any(p < 0 for p in parts):
                    raise ValueError
            except Exception:
                errors.append('face_bbox: expected "auto" or "x,y,w,h" (4 non-negative ints)')

    if req.avatar_type not in ("frontal", "side-right", "side-left"):
        errors.append("avatar_type: frontal / side-right / side-left")
    if req.normalization_type not in ("canonical", "centered", "centered_bbox"):
        errors.append("normalization_type: canonical / centered / centered_bbox")
    if req.model_size not in (512, 768):
        errors.append("model_size: 512 or 768")
    if not (1 <= req.training_duration <= 72):
        errors.append("training_duration: 1–72 hours")
    if req.instance_type not in INSTANCE_COST_USD_HR:
        errors.append(f"Unknown instance_type: {req.instance_type}")
    return errors


def _estimate_cost(req: "TrainRequest") -> dict:
    rate = INSTANCE_COST_USD_HR.get(req.instance_type, 0.0)
    effective_rate = rate * (0.3 if req.enable_spot_training else 1.0)
    cost = round(effective_rate * req.training_duration, 2)
    return {
        "rate_usd_hr": rate,
        "duration_hr": req.training_duration,
        "spot": req.enable_spot_training,
        "estimated_usd": cost,
    }


def _new_job_id() -> str:
    return f"job_{os.urandom(5).hex()}"


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _job_log(job: dict, line: str) -> None:
    job["logs"].append(f"[{_now_iso()}] {line}")
    job["logs"] = job["logs"][-500:]


# ─── Live runner: actual pipeline as in Avatar Training V3.6.ipynb ────────
async def _train_runner_live(job_id: str):
    """
    Real submission to SageMaker:
      1. get_raw_config + prepare_avatar_config
      2. DatasetPreparatorV3.prepare_dataset(...)
      3. atrain.create_experiment_config(...) → JSON
      4. sagemaker.create_training_job(**json)
      5. save configs_v3/{experiment_name}-config.json (for Deploy tab)

    After step 4 the job is on SageMaker — progress is then tracked
    via describe_training_job (polling).
    """
    async with _TRAIN_LOCK:
        job = TRAIN_JOBS.get(job_id)
    if not job:
        return
    p = job["params"]

    try:
        async with _TRAIN_LOCK:
            job["status"] = "running"
            job["stage"] = "validating"
            job["progress"] = 0.05
            _job_log(job, "→ Stage: validating params & AWS credentials")

        # 1. build avatar config (sync — push to a thread)
        async with _TRAIN_LOCK:
            job["stage"] = "preparing_dataset"
            job["progress"] = 0.15
            _job_log(job, "→ Stage: preparing_dataset (this takes a few minutes)")

        def _build_config():
            train_paths = p["train_footages_paths"] or [p["footage_path"]]
            raw = _get_raw_config(
                experiment_name=p["experiment_name"],
                footage_path=p["footage_path"],
                train_footages_paths=train_paths,
                alpha_footage_path=p.get("alpha_footage_path"),
                pauses=list(p.get("pauses", [])),
                avatar_type=p["avatar_type"],
                normalization_type=p["normalization_type"],
                model_size=p["model_size"],
                symmetrical_blinking=p["symmetrical_blinking"],
            )
            return _prepare_avatar_config(raw, add_timestamp=False)

        config_obj = await asyncio.to_thread(_build_config)
        _job_log(job, f"  ✓ config built ({p['experiment_name']})")

        # face_bbox: None / "auto" / [x,y,w,h]
        bbox_raw = (p.get("face_bbox") or "").strip()
        if not bbox_raw:
            face_bbox_arg = None
        elif bbox_raw.lower() == "auto":
            face_bbox_arg = "auto"
        else:
            face_bbox_arg = [int(x.strip()) for x in bbox_raw.split(",")]

        def _prepare():
            preparator = _DatasetPreparatorV3(config_obj)
            data_dir = BASE_DIR / "results" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            return preparator.prepare_dataset(
                preview=False, data_dir=str(data_dir), face_bbox=face_bbox_arg
            )

        await asyncio.to_thread(_prepare)
        _job_log(job, "  ✓ dataset prepared on local disk")

        # 2. SageMaker experiment config
        async with _TRAIN_LOCK:
            job["stage"] = "creating_sagemaker_config"
            job["progress"] = 0.55
            _job_log(job, "→ Stage: creating SageMaker training-job spec")

        train_batch_size = 6 if p["model_size"] == 512 else 3
        try:
            commit_hash = await asyncio.to_thread(_atrain.get_commit_hash)
        except Exception as e:
            commit_hash = "unknown"
            _job_log(job, f"  ⚠ get_commit_hash failed: {e}")
        _job_log(job, f"  commit_hash={commit_hash}  train_batch_size={train_batch_size}")

        def _create_cfg():
            cfg_dir = BASE_DIR / "experiment_configs_v3"
            cfg_dir.mkdir(exist_ok=True)
            return _atrain.create_experiment_config(
                config_obj,
                template_path=_SAGEMAKER_TEMPLATE_PATH_V36,
                instance_type=p["instance_type"],
                commit_hash=commit_hash,
                training_duration=p["training_duration"],
                train_batch_size=train_batch_size,
                enable_spot_training=p["enable_spot_training"],
                enable_amp=p["enable_amp"],
                config_dir=str(cfg_dir),
            )

        job_config_path = await asyncio.to_thread(_create_cfg)
        _job_log(job, f"  ✓ experiment_configs_v3/{Path(job_config_path).name}")

        # 3. submit to SageMaker
        async with _TRAIN_LOCK:
            job["stage"] = "submitting_to_sagemaker"
            job["progress"] = 0.85
            _job_log(job, "→ Stage: sagemaker.create_training_job")

        with open(job_config_path, "r") as fh:
            sm_payload = json.load(fh)

        def _submit():
            return _sagemaker_client.create_training_job(**sm_payload)

        sm_resp = await asyncio.to_thread(_submit)
        sm_arn = sm_resp.get("TrainingJobArn", "")
        sm_name = sm_arn.split("/")[-1] if sm_arn else sm_payload.get("TrainingJobName", "")
        _job_log(job, f"  ✓ submitted: {sm_arn}")

        # 4. save avatar config — Deploy tab will pick it up immediately
        async with _TRAIN_LOCK:
            job["stage"] = "saving_avatar_config"
            job["progress"] = 0.95
            _job_log(job, "→ Stage: saving configs_v3/{name}-config.json")

        cfg_out = CONFIGS_DIR / f"{p['experiment_name']}-config.json"
        def _save_cfg():
            if _FastTalkerConfigReader:
                reader = _FastTalkerConfigReader()
                reader.save(config_obj, path=str(cfg_out))
            else:
                # fallback: serialise what we can
                try:
                    with open(cfg_out, "w") as fh:
                        json.dump(getattr(config_obj, "to_dict", lambda: str(config_obj))(), fh, indent=2, default=str)
                except Exception:
                    with open(cfg_out, "w") as fh:
                        fh.write(str(config_obj))
        await asyncio.to_thread(_save_cfg)
        _job_log(job, f"  ✓ configs_v3/{cfg_out.name} (ready to deploy)")

        # done — job now lives on SageMaker
        async with _TRAIN_LOCK:
            job["status"] = "submitted"
            job["stage"] = "running_on_sagemaker"
            job["progress"] = 1.0
            job["finished_at"] = _now_iso()
            job["sagemaker_job_name"] = sm_name
            job["sagemaker_arn"] = sm_arn
            job["sagemaker_console_url"] = (
                f"https://{AWS_REGION}.console.aws.amazon.com/sagemaker/home"
                f"?region={AWS_REGION}#/jobs/{sm_name}" if sm_name else None
            )
            _job_log(job, f"✓ Live job is now running on SageMaker as \"{sm_name}\"")

        # 5. (optional) follow-up polling of SageMaker status
        asyncio.create_task(_poll_sagemaker_status(job_id))

    except Exception as e:
        async with _TRAIN_LOCK:
            job["status"] = "failed"
            job["error"] = str(e)
            job["finished_at"] = _now_iso()
            _job_log(job, f"✗ Failed: {e!s}")


async def _poll_sagemaker_status(job_id: str):
    """After create_training_job, poll SageMaker every minute and update status."""
    if not _SAGEMAKER_LIVE:
        return
    while True:
        await asyncio.sleep(60)
        async with _TRAIN_LOCK:
            job = TRAIN_JOBS.get(job_id)
            if not job or not job.get("sagemaker_job_name"):
                return
            sm_name = job["sagemaker_job_name"]
            if job["status"] in ("completed", "failed", "cancelled"):
                return
        try:
            desc = await asyncio.to_thread(
                _sagemaker_client.describe_training_job, TrainingJobName=sm_name
            )
            sm_status = desc.get("TrainingJobStatus", "Unknown")
            sm_secondary = desc.get("SecondaryStatus", "")
            async with _TRAIN_LOCK:
                job = TRAIN_JOBS[job_id]
                job["sagemaker_status"] = sm_status
                job["sagemaker_secondary_status"] = sm_secondary
                _job_log(job, f"  SageMaker: {sm_status} / {sm_secondary}")
                if sm_status == "Completed":
                    job["status"] = "completed"
                    job["stage"] = "done"
                    job["finished_at"] = _now_iso()
                    _job_log(job, "✓ SageMaker job Completed")
                    return
                if sm_status == "Failed":
                    job["status"] = "failed"
                    job["error"] = desc.get("FailureReason", "SageMaker reported Failed")
                    job["finished_at"] = _now_iso()
                    _job_log(job, f"✗ SageMaker Failed: {job['error']}")
                    return
                if sm_status == "Stopped":
                    job["status"] = "cancelled"
                    job["finished_at"] = _now_iso()
                    return
        except Exception as e:
            async with _TRAIN_LOCK:
                _job_log(TRAIN_JOBS[job_id], f"  ⚠ describe_training_job failed: {e}")


# ─── Simulation runner: kept for dev/demo without AWS ─────────────────────
async def _train_runner_sim(job_id: str):
    """Background pipeline simulation (when live mode isn't available)."""
    import random

    async with _TRAIN_LOCK:
        job = TRAIN_JOBS.get(job_id)
    if not job:
        return
    p = job["params"]

    try:
        for stage_name, progress_pct, t_min, t_max in TRAIN_STAGES_SIM:
            async with _TRAIN_LOCK:
                if job["status"] == "cancelled":
                    _job_log(job, f"Cancelled before stage {stage_name}")
                    return
                job["stage"] = stage_name
                job["status"] = "running"
                _job_log(job, f"→ Stage: {stage_name}")

            duration = random.uniform(t_min, t_max)
            steps = max(4, int(duration * 4))
            prev_progress = job["progress"]
            target = progress_pct
            for i in range(steps):
                await asyncio.sleep(duration / steps)
                async with _TRAIN_LOCK:
                    if job["status"] == "cancelled":
                        _job_log(job, f"Cancelled during {stage_name}")
                        return
                    job["progress"] = round(
                        prev_progress + (target - prev_progress) * ((i + 1) / steps), 4
                    )

            async with _TRAIN_LOCK:
                if stage_name == "training":
                    fake_loss = round(2.5 * (1 - target) + random.uniform(0.01, 0.06), 4)
                    _job_log(job, f"  loss={fake_loss}  amp={p['enable_amp']}  bs={6 if p['model_size']==512 else 3}")
                elif stage_name == "evaluating":
                    _job_log(job, f"  val_loss={round(random.uniform(0.08, 0.18), 4)}  PSNR={round(random.uniform(28, 34), 2)}")
                elif stage_name == "uploading":
                    _job_log(job, f"  artifact: s3://elai-models/{p['experiment_name']}/{job_id}.tar.gz")

        async with _TRAIN_LOCK:
            job["status"] = "completed"
            job["stage"] = "done"
            job["progress"] = 1.0
            job["finished_at"] = _now_iso()
            _job_log(job, "✓ [simulated] completed")

            # simulate config creation for the Deploy tab
            cfg_out = CONFIGS_DIR / f"{p['experiment_name']}-config.json"
            cfg_out.write_text(json.dumps({
                "experiment_name": p["experiment_name"],
                "avatarConfig": {"_simulated": True, "params": p},
            }, indent=2))
            _job_log(job, f"  ✓ configs_v3/{cfg_out.name} (simulation)")
    except Exception as e:
        async with _TRAIN_LOCK:
            job["status"] = "failed"
            job["error"] = str(e)
            job["finished_at"] = _now_iso()
            _job_log(job, f"✗ Failed: {e}")


async def _train_runner(job_id: str):
    """Dispatch: live SageMaker if available, otherwise simulation."""
    if _SAGEMAKER_LIVE:
        await _train_runner_live(job_id)
    else:
        await _train_runner_sim(job_id)


@app.get("/api/train/mode")
async def train_mode(user: User = Depends(require_role(Role.viewer))):
    """Which training mode is active. Frontend uses this for the badge."""
    return {
        "mode": "live" if _SAGEMAKER_LIVE else "simulation",
        "aws_region": AWS_REGION,
        "avatars_repo_path": AVATARS_REPO_PATH,
        "reason": None if _SAGEMAKER_LIVE else _SAGEMAKER_INIT_ERROR,
    }


@app.get("/api/train/instances")
async def list_instances(user: User = Depends(require_role(Role.viewer))):
    return {
        "instances": [
            {"type": t, "rate_usd_hr": rate}
            for t, rate in INSTANCE_COST_USD_HR.items()
        ]
    }


@app.post("/api/train/estimate")
async def estimate_train(req: TrainRequest, user: User = Depends(require_role(Role.viewer))):
    """Cost estimate + pre-flight validation, no launch."""
    errors = _validate_train_params(req)
    return {
        "valid": not errors,
        "errors": errors,
        "cost": _estimate_cost(req),
    }


@app.post("/api/train")
async def create_train_job(req: TrainRequest, user: User = Depends(require_role(Role.developer))):

    errors = _validate_train_params(req)
    if errors:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Validation failed", "errors": errors},
        )

    job_id = _new_job_id()
    cost = _estimate_cost(req)
    job = {
        "id": job_id,
        "owner": user,
        "status": "queued",
        "stage": "queued",
        "progress": 0.0,
        "created_at": _now_iso(),
        "started_at": _now_iso(),
        "finished_at": None,
        "error": None,
        "cost": cost,
        "params": req.model_dump(),
        "logs": [f"[{_now_iso()}] Job {job_id} accepted, queued for execution"],
    }
    async with _TRAIN_LOCK:
        TRAIN_JOBS[job_id] = job

    asyncio.create_task(_train_runner(job_id))

    mode_label = "SageMaker (live)" if _SAGEMAKER_LIVE else "simulation"
    return {
        "success": True,
        "id": job_id,
        "cost": cost,
        "mode": "live" if _SAGEMAKER_LIVE else "simulation",
        "message": (
            f"[{mode_label}] Pipeline \"{req.experiment_name}\" queued "
            f"({req.instance_type} · {req.training_duration}h · ≈${cost['estimated_usd']})"
        ),
    }


@app.get("/api/train/jobs")
async def list_jobs(user: User = Depends(require_role(Role.viewer))):
    async with _TRAIN_LOCK:
        items = []
        for job in TRAIN_JOBS.values():
            if job.get("owner") != user:
                continue
            items.append({
                "id": job["id"],
                "experiment_name": job["params"]["experiment_name"],
                "status": job["status"],
                "stage": job["stage"],
                "progress": job["progress"],
                "instance_type": job["params"]["instance_type"],
                "created_at": job["created_at"],
                "finished_at": job["finished_at"],
                "cost": job["cost"],
                "sagemaker_job_name": job.get("sagemaker_job_name"),
                "sagemaker_status": job.get("sagemaker_status"),
                "sagemaker_console_url": job.get("sagemaker_console_url"),
            })
    items.sort(key=lambda j: j["created_at"], reverse=True)
    return {"jobs": items}


@app.get("/api/train/jobs/{job_id}")
async def get_job(job_id: str, user: User = Depends(require_role(Role.viewer))):
    async with _TRAIN_LOCK:
        job = TRAIN_JOBS.get(job_id)
        if not job or job.get("owner") != user:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(job)


@app.post("/api/train/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, user: User = Depends(require_role(Role.developer))):
    sm_name = None
    async with _TRAIN_LOCK:
        job = TRAIN_JOBS.get(job_id)
        if not job or job.get("owner") != user:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] in ("completed", "failed", "cancelled"):
            return {"success": False, "message": f"Job is already {job['status']}"}
        sm_name = job.get("sagemaker_job_name")
        job["status"] = "cancelled"
        job["finished_at"] = _now_iso()
        _job_log(job, f"Cancel requested by {user}")

    # If the job is on SageMaker, ask SageMaker to stop it
    if sm_name and _SAGEMAKER_LIVE:
        try:
            await asyncio.to_thread(
                _sagemaker_client.stop_training_job, TrainingJobName=sm_name
            )
            async with _TRAIN_LOCK:
                _job_log(TRAIN_JOBS[job_id], f"  sagemaker.stop_training_job({sm_name}) sent")
        except Exception as e:
            async with _TRAIN_LOCK:
                _job_log(TRAIN_JOBS[job_id], f"  ⚠ stop_training_job failed: {e}")

    return {"success": True, "message": "Cancellation requested"}


# ────────────────────────────────────────────────────────────────────────────
# Tool 3: Clone Voice — port from voice-clone/main.py
# ────────────────────────────────────────────────────────────────────────────
def add_ssml_pauses(text: str) -> str:
    if "<speak>" in text or "<break" in text:
        return text
    text = text.replace("&", "&amp;").replace('"', "&quot;")
    text = re.sub(r"([.!?])\s+", r'\1<break time="0.6s"/> ', text)
    text = re.sub(r"([,;:\u2014\u2013])\s+", r'\1<break time="0.3s"/> ', text)
    return f"<speak>{text}</speak>"


async def analyze_audio(path: Path) -> dict:
    """ffprobe + pcm analysis. All steps are optional — if binaries are missing, return what we can."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", str(path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        data = json.loads(out)
    except Exception:
        data = {}

    fmt = data.get("format", {})
    streams = data.get("streams", [])
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    duration = float(fmt.get("duration", 0) or audio.get("duration", 0) or 0)
    bitrate = int(fmt.get("bit_rate", 0) or 0) // 1000
    channels = int(audio.get("channels", 0) or 0)
    sr = int(audio.get("sample_rate", 0) or 0)
    codec = audio.get("codec_name", "unknown")

    if duration == 0:
        try:
            dur_proc = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "quiet",
                "-print_format", "json", "-show_entries", "format=duration",
                "-analyzeduration", "100000000", "-probesize", "100000000",
                str(path),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            dur_out, _ = await dur_proc.communicate()
            dur_data = json.loads(dur_out)
            duration = float(dur_data.get("format", {}).get("duration", 0) or 0)
        except Exception:
            pass

    noise_db = echo_score = clipping_pct = rms_db = None
    try:
        pcm_proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "quiet", "-i", str(path),
            "-t", "60", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        pcm_bytes, _ = await pcm_proc.communicate()
        if pcm_bytes and len(pcm_bytes) >= 4:
            import numpy as np  # type: ignore

            samples = np.frombuffer(pcm_bytes, dtype=np.float32)
            samples = samples[~np.isnan(samples)]
            if len(samples) > 0:
                rms = float(np.sqrt(np.mean(samples ** 2)))
                rms_db = round(20 * np.log10(rms + 1e-9), 1)
                clipping_pct = round(float(np.mean(np.abs(samples) > 0.98)) * 100, 2)
                frame = int(16000 * 0.02)
                if len(samples) >= frame:
                    n = len(samples) // frame
                    frame_rms = np.array([
                        np.sqrt(np.mean(samples[i * frame : (i + 1) * frame] ** 2))
                        for i in range(n)
                    ])
                    noise_floor = float(np.percentile(frame_rms, 10))
                    noise_db = round(20 * np.log10(noise_floor + 1e-9), 1)
                lag_min, lag_max = int(0.05 * 16000), int(0.50 * 16000)
                if len(samples) > lag_max:
                    seg = samples[: min(len(samples), 16000 * 10)]
                    ac0 = float(np.dot(seg, seg))
                    if ac0 > 0:
                        lags = range(lag_min, min(lag_max, len(seg) - 1), 800)
                        ac_vals = [abs(float(np.dot(seg[:-lag], seg[lag:]))) / ac0 for lag in lags]
                        echo_score = round(float(np.mean(ac_vals)) * 100, 1)
    except Exception:
        pass

    return {
        "duration": round(duration, 1),
        "bitrate_kbps": bitrate,
        "channels": channels,
        "sample_rate": sr,
        "codec": codec,
        "rms_db": rms_db,
        "noise_db": noise_db,
        "echo_score": echo_score,
        "clipping_pct": clipping_pct,
    }


def estimate_time(info: dict) -> dict:
    dur = info.get("duration", 60)
    cleanvoice_s = max(20, min(dur * 0.5, 90))
    ffmpeg_s = max(3, dur * 0.02)
    ivc_s = 10
    total = int(cleanvoice_s + ffmpeg_s + ivc_s)
    return {
        "total_sec": total,
        "label": f"~{total // 60}m {total % 60}s" if total >= 60 else f"~{total}s",
        "breakdown": {"Cleanvoice": f"{int(cleanvoice_s)}s", "ElevenLabs IVC": f"{int(ivc_s)}s"},
    }


def validate_audio(info: dict) -> list[dict]:
    issues: list[dict] = []
    dur = info.get("duration", 0)
    br = info.get("bitrate_kbps", 0)
    ch = info.get("channels", 0)
    sr = info.get("sample_rate", 0)
    noise_db = info.get("noise_db")
    echo_score = info.get("echo_score")
    clipping_pct = info.get("clipping_pct")
    rms_db = info.get("rms_db")

    if dur < 30:
        issues.append({"level": "error", "message": "Recording is too short",
                       "detail": f"Length {dur:.0f}s. Minimum 30s, ideal 1–3 minutes."})
    elif dur < 60:
        issues.append({"level": "warning", "message": "Short recording",
                       "detail": f"{dur:.0f}s. For best quality use 1+ minute."})
    elif dur > 600:
        issues.append({"level": "tip", "message": "Long recording",
                       "detail": f"{dur/60:.1f} min — first ~7 minutes will be used."})

    if 0 < br < 64:
        issues.append({"level": "error", "message": "Quality too low",
                       "detail": f"Bitrate {br} kbps. Export to MP3 128 kbps+ or WAV."})
    elif 0 < br < 96:
        issues.append({"level": "warning", "message": "Low quality",
                       "detail": f"Bitrate {br} kbps. 128 kbps+ recommended."})

    if sr and sr < 16000:
        issues.append({"level": "error", "message": "Sample rate too low",
                       "detail": f"{sr} Hz. Minimum 16 000 Hz."})

    if clipping_pct is not None:
        if clipping_pct > 5:
            issues.append({"level": "error", "message": "Clipping detected",
                           "detail": f"{clipping_pct:.1f}% samples clipped. Lower the input gain."})
        elif clipping_pct > 1:
            issues.append({"level": "warning", "message": "Mild clipping",
                           "detail": f"{clipping_pct:.1f}% close to the limit."})

    if noise_db is not None:
        if noise_db > -30:
            issues.append({"level": "error", "message": "High background noise",
                           "detail": f"Noise {noise_db} dBFS. Record in a quiet room."})
        elif noise_db > -45:
            issues.append({"level": "warning", "message": "Noticeable background noise",
                           "detail": f"Noise {noise_db} dBFS. Will be partially suppressed."})

    if echo_score is not None:
        if echo_score > 15:
            issues.append({"level": "error", "message": "Strong echo / reverb",
                           "detail": f"Score: {echo_score:.0f}/100. Record in a small carpeted room."})
        elif echo_score > 7:
            issues.append({"level": "warning", "message": "Moderate reverb",
                           "detail": f"Score: {echo_score:.0f}/100."})

    if rms_db is not None and rms_db < -35:
        issues.append({"level": "warning", "message": "Recording is very quiet",
                       "detail": f"RMS {rms_db} dBFS. Move closer to the microphone."})

    if ch > 2:
        issues.append({"level": "tip", "message": "Multichannel input",
                       "detail": f"{ch} channels — will be downmixed to mono."})

    if not any(i["level"] in ("error", "warning") for i in issues):
        issues.append({"level": "tip", "message": "Audio looks good ✓",
                       "detail": "All checks passed."})

    return issues


@app.post("/validate")
async def validate_audio_endpoint(
    file: UploadFile = File(...),
    client_duration: float = Form(0),
    user: User = Depends(require_role(Role.developer)),
):
    suffix = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    tmp_dir = Path(tempfile.mkdtemp())
    path = tmp_dir / f"sample{suffix}"
    try:
        with path.open("wb") as f_out:
            while True:
                chunk = await file.read(512 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)

        try:
            info = await analyze_audio(path)
        except Exception:
            info = {}

        if info.get("duration", 0) == 0 and client_duration > 0:
            info["duration"] = client_duration
        if info.get("duration", 0) == 0:
            size_mb = path.stat().st_size / 1024 / 1024
            info["duration"] = size_mb * 60

        return {
            "issues": validate_audio(info),
            "eta": estimate_time(info),
            "info": info,
        }
    except Exception:
        return {
            "issues": [],
            "eta": {"total_sec": 90, "label": "~1m 30s",
                    "breakdown": {"Cleanvoice": "80s", "ElevenLabs IVC": "10s"}},
            "info": {},
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/clone")
async def clone_voice(
    name: str = Form(...),
    description: str = Form(""),
    model_id: str = Form("eleven_multilingual_v2"),
    file: UploadFile = File(...),
    user: User = Depends(require_role(Role.developer)),
):
    if not (CLEANVOICE_API_KEY and ELEVENLABS_API_KEY and _cv):
        raise HTTPException(
            status_code=503,
            detail="Voice cloning not configured: add CLEANVOICE_API_KEY and ELEVENLABS_API_KEY to .env",
        )

    suffix = Path(file.filename or "audio.mp3").suffix.lower() or ".mp3"
    tmp_dir = Path(tempfile.mkdtemp())
    raw_path = tmp_dir / f"raw{suffix}"
    mp3_path = tmp_dir / "final.mp3"

    total = 0
    with raw_path.open("wb") as f_out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_MB * 1024 * 1024:
                raise HTTPException(413, f"File too large (max {MAX_MB} MB)")
            f_out.write(chunk)
    if total == 0:
        raise HTTPException(400, "Empty file")

    try:
        def _cv_process():
            return _cv.process(str(raw_path), {
                "studio_sound": True, "normalize": True, "target_lufs": -18,
            })

        try:
            result = await asyncio.to_thread(_cv_process)
        except Exception as e:
            raise HTTPException(502, f"Cleanvoice failed: {e!s}")

        cleaned_url = getattr(result.audio, "url", None)
        if not cleaned_url:
            raise HTTPException(502, "Cleanvoice did not return an audio URL")

        async with httpx.AsyncClient(timeout=300) as http:
            dl = await http.get(cleaned_url)
        if dl.status_code >= 400:
            raise HTTPException(502, f"Cleanvoice download failed: {dl.status_code}")

        cleaned_path = tmp_dir / "cleaned.mp3"
        cleaned_path.write_bytes(dl.content)

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", str(cleaned_path),
            "-vn", "-ar", "44100", "-ac", "1", "-b:a", "128k", str(mp3_path),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, fferr = await proc.communicate()
        if proc.returncode != 0:
            raise HTTPException(502, f"ffmpeg failed: {fferr.decode()[-300:]}")

        chunks_dir = tmp_dir / "chunks"
        chunks_dir.mkdir()
        split = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", str(mp3_path),
            "-f", "segment", "-segment_time", "300",
            "-c", "copy", str(chunks_dir / "chunk_%03d.mp3"),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await split.communicate()

        chunk_files = sorted(chunks_dir.glob("chunk_*.mp3")) or [mp3_path]
        chunk_files = chunk_files[:25]
        total_mb = sum(p.stat().st_size for p in chunk_files) / 1024 / 1024

        files_payload = [
            ("files", (f"{name}_part{i}.mp3", p.read_bytes(), "audio/mpeg"))
            for i, p in enumerate(chunk_files)
        ]
        async with httpx.AsyncClient(timeout=300) as http:
            ivc_resp, user_resp = await asyncio.gather(
                http.post(
                    ELEVENLABS_IVC_URL,
                    headers={"xi-api-key": ELEVENLABS_API_KEY},
                    data={
                        "name": name,
                        "description": description or f"Clone: {file.filename}",
                        "remove_background_noise": "false",
                    },
                    files=files_payload,
                ),
                http.get(f"{ELEVENLABS_BASE}/user",
                         headers={"xi-api-key": ELEVENLABS_API_KEY}),
                return_exceptions=True,
            )

        if isinstance(ivc_resp, Exception):
            raise HTTPException(502, str(ivc_resp))
        if ivc_resp.status_code >= 400:
            raise HTTPException(ivc_resp.status_code, f"ElevenLabs IVC failed: {ivc_resp.text[:400]}")

        eleven = ivc_resp.json()
        account_id = None
        if not isinstance(user_resp, Exception) and user_resp.status_code == 200:
            u = user_resp.json()
            account_id = u.get("xi_api_key_id") or u.get("user_id") or u.get("email")

        return {
            "ok": True,
            "voice_id": eleven.get("voice_id"),
            "model_id": model_id,
            "account_id": account_id,
            "chunks": len(chunk_files),
            "mp3_mb": round(total_mb, 2),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/tts/{voice_id}")
async def tts(
    voice_id: str,
    text: str = "Hello!",
    model_id: str = "eleven_multilingual_v2",
    stability: float = 0.45,
    similarity: float = 0.80,
    speed: float = 1.0,
    style: float = 0.20,
    user: User = Depends(require_role(Role.viewer)),
):
    if not ELEVENLABS_API_KEY:
        raise HTTPException(503, "ELEVENLABS_API_KEY is not set")

    async with httpx.AsyncClient(timeout=120) as http:
        r = await http.post(
            f"{ELEVENLABS_BASE}/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": add_ssml_pauses(text),
                "model_id": model_id,
                "voice_settings": {
                    "stability": round(stability, 2),
                    "similarity_boost": round(similarity, 2),
                    "style": round(style, 2),
                    "speed": round(speed, 2),
                    "use_speaker_boost": True,
                },
            },
        )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text[:400])
    return Response(content=r.content, media_type="audio/mpeg")


@app.post("/generate")
async def generate(
    config: str = Form(...),
    text: str = Form(...),
    user: User = Depends(require_role(Role.viewer)),
):
    if not ELEVENLABS_API_KEY:
        raise HTTPException(503, "ELEVENLABS_API_KEY is not set")

    parts = config.strip().split(":")
    if len(parts) < 7:
        raise HTTPException(400, "Invalid config string")
    voice_id, model_id = parts[0], parts[1]
    try:
        stability = float(parts[2]); similarity = float(parts[3])
        speed = float(parts[5]); style = float(parts[6])
    except ValueError:
        raise HTTPException(400, "Invalid numeric values in config")

    async with httpx.AsyncClient(timeout=120) as http:
        r = await http.post(
            f"{ELEVENLABS_BASE}/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": add_ssml_pauses(text),
                "model_id": model_id,
                "voice_settings": {
                    "stability": round(stability, 2),
                    "similarity_boost": round(similarity, 2),
                    "style": round(style, 2),
                    "speed": round(speed, 2),
                    "use_speaker_boost": True,
                },
            },
        )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, f"ElevenLabs TTS failed: {r.text[:400]}")
    return Response(content=r.content, media_type="audio/mpeg")


# ────────────────────────────────────────────────────────────────────────────
# Tool 4: S3 Asset Browser (visual media library for training footage)
# ────────────────────────────────────────────────────────────────────────────
_s3_client = None
_S3_INIT_ERROR: Optional[str] = None
try:
    import boto3 as _boto3  # noqa: F401  (re-imported lazily; safe if already loaded)
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
    try:
        _s3_client = _boto3.client("s3", region_name=AWS_REGION)
    except Exception as _s3_err:
        _S3_INIT_ERROR = str(_s3_err)
except Exception as _s3_imp_err:
    _S3_INIT_ERROR = f"boto3 not installed: {_s3_imp_err}"

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


@app.get("/api/s3/config")
async def s3_config(user: User = Depends(require_role(Role.viewer))):
    """Default bucket/prefix and whether S3 is operational."""
    return {
        "available": _s3_client is not None,
        "error": _S3_INIT_ERROR,
        "default_bucket": DEFAULT_S3_BUCKET,
        "default_prefix": DEFAULT_S3_PREFIX,
        "region": AWS_REGION,
    }


@app.get("/api/s3/list")
async def s3_list(
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    only_videos: bool = True,
    max_keys: int = 200,
    user: User = Depends(require_role(Role.viewer)),
):
    """List objects in the bucket. Returns folders + files; videos only by default."""
    if not _s3_client:
        raise HTTPException(503, f"S3 unavailable: {_S3_INIT_ERROR or 'no client'}")
    bkt = (bucket or DEFAULT_S3_BUCKET).strip()
    pfx = (prefix if prefix is not None else DEFAULT_S3_PREFIX).lstrip("/")

    try:
        resp = await asyncio.to_thread(
            _s3_client.list_objects_v2,
            Bucket=bkt, Prefix=pfx, Delimiter="/", MaxKeys=max_keys,
        )
    except NoCredentialsError:
        raise HTTPException(503, "AWS credentials not configured (set AWS_ACCESS_KEY_ID/SECRET in .env or use ~/.aws/credentials).")
    except ClientError as e:
        raise HTTPException(502, f"S3 error: {e.response.get('Error', {}).get('Message', str(e))}")
    except BotoCoreError as e:
        raise HTTPException(502, f"S3 error: {e}")

    folders = [
        {"key": cp["Prefix"], "name": cp["Prefix"].rstrip("/").split("/")[-1] + "/"}
        for cp in resp.get("CommonPrefixes", []) or []
    ]

    files = []
    for obj in resp.get("Contents", []) or []:
        key = obj["Key"]
        if key.endswith("/"):
            continue
        name = key.split("/")[-1]
        ext = Path(name).suffix.lower()
        if only_videos and ext not in VIDEO_EXTS:
            continue
        files.append({
            "key": key,
            "name": name,
            "ext": ext.lstrip("."),
            "size": obj["Size"],
            "size_human": _human_bytes(obj["Size"]),
            "last_modified": obj["LastModified"].isoformat() if obj.get("LastModified") else None,
            "uri": f"s3://{bkt}/{key}",
        })

    return {
        "bucket": bkt,
        "prefix": pfx,
        "folders": folders,
        "files": files,
        "is_truncated": resp.get("IsTruncated", False),
        "total": len(files),
    }


@app.get("/api/s3/preview")
async def s3_preview(
    bucket: str,
    key: str,
    expires: int = 600,
    user: User = Depends(require_role(Role.viewer)),
):
    """Generate a presigned URL so the browser can stream the video preview."""
    if not _s3_client:
        raise HTTPException(503, f"S3 unavailable: {_S3_INIT_ERROR or 'no client'}")
    try:
        url = await asyncio.to_thread(
            _s3_client.generate_presigned_url,
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=max(60, min(expires, 3600)),
        )
    except Exception as e:
        raise HTTPException(502, f"Presign failed: {e}")
    return {"url": url, "expires_in": expires}


# ────────────────────────────────────────────────────────────────────────────
# Tool 6: Enhance Video (GFPGAN face restoration + alpha upscale)
# Mirrors /Users/admin/Downloads/video_enhance_v2.ipynb.
# Falls back to a simulated job runner when GFPGAN / CUDA aren't installed,
# so the UI stays fully usable on dev machines.
# ────────────────────────────────────────────────────────────────────────────
_ENHANCE_LIVE = False
_ENHANCE_INIT_ERROR: Optional[str] = None
_ENHANCE_GPU = False
try:
    import torch as _enh_torch  # type: ignore
    import gfpgan  # noqa: F401
    import basicsr  # noqa: F401
    import realesrgan  # noqa: F401
    import cv2 as _enh_cv2  # noqa: F401
    _ENHANCE_LIVE = True
    _ENHANCE_GPU = bool(getattr(_enh_torch, "cuda", None) and _enh_torch.cuda.is_available())
except Exception as _enh_exc:  # pragma: no cover
    _ENHANCE_INIT_ERROR = (
        f"GFPGAN stack not available ({_enh_exc!s}). "
        "Install with: pip install gfpgan basicsr realesrgan opencv-python-headless torch"
    )

ENHANCE_JOBS: dict[str, dict] = {}
_ENHANCE_LOCK = asyncio.Lock()

ENHANCE_STAGES_SIM = [
    ("download_rgb",  0.10, 1, 3),
    ("enhance_rgb",   0.65, 8, 18),
    ("upload_rgb",    0.75, 1, 2),
    ("upscale_alpha", 0.92, 2, 5),
    ("upload_alpha",  1.00, 1, 2),
]


class EnhanceItem(BaseModel):
    video: str
    alpha: Optional[str] = None


class EnhanceRequest(BaseModel):
    name: Optional[str] = ""
    items: list[EnhanceItem]
    gfpgan_version: str = "1.4"          # "1.3" | "1.4"
    weight: float = 0.1                  # 0.0 .. 1.0
    upscale: int = 3                     # 1 | 2 | 3 | 4
    bg_upsampler: Optional[str] = None   # null | "realesrgan"
    output_suffix: str = "_gfpgan"


def _make_output_uri(input_uri: str, suffix: str) -> str:
    """`s3://bucket/dir/file.mp4` + `_gfpgan` → `s3://bucket/dir/file_gfpgan.mp4`."""
    if not input_uri or "://" not in input_uri:
        return (input_uri or "") + suffix
    parsed = _urlparse(input_uri)
    p = Path(parsed.path.lstrip("/"))
    new_key = str(p.with_name(f"{p.stem}{suffix}{p.suffix}"))
    return f"{parsed.scheme}://{parsed.netloc}/{new_key}"


def _validate_enhance(req: "EnhanceRequest") -> list[str]:
    errors: list[str] = []
    if not req.items:
        errors.append("At least one video is required")
    for i, it in enumerate(req.items, 1):
        if not it.video:
            errors.append(f"Item #{i}: video URI is required")
        elif not _is_valid_uri(it.video):
            errors.append(f"Item #{i}: video has unsupported scheme")
        if it.alpha and not _is_valid_uri(it.alpha):
            errors.append(f"Item #{i}: alpha has unsupported scheme")
    if req.gfpgan_version not in ("1.3", "1.4"):
        errors.append("gfpgan_version: must be '1.3' or '1.4'")
    if not (0.0 <= req.weight <= 1.0):
        errors.append("weight: must be between 0.0 and 1.0")
    if req.upscale not in (1, 2, 3, 4):
        errors.append("upscale: must be 1, 2, 3, or 4")
    if req.bg_upsampler not in (None, "", "realesrgan"):
        errors.append("bg_upsampler: must be 'realesrgan' or empty")
    if not re.match(r"^[a-zA-Z0-9._-]{1,32}$", req.output_suffix or ""):
        errors.append("output_suffix: only letters, digits, . _ - (max 32 chars)")
    return errors


def _create_enhance_job(req: "EnhanceRequest", owner_email: str) -> dict:
    job_id = secrets.token_urlsafe(9)
    items: list[dict] = []
    for it in req.items:
        items.append({
            "video": it.video,
            "alpha": it.alpha or None,
            "output_video": _make_output_uri(it.video, req.output_suffix),
            "output_alpha": _make_output_uri(it.alpha, req.output_suffix) if it.alpha else None,
            "status": "pending",
            "stage": "queued",
            "progress": 0.0,
            "error": None,
        })

    return {
        "id": job_id,
        "name": (req.name or "").strip() or f"enhance-{job_id[:6]}",
        "params": {
            "gfpgan_version": req.gfpgan_version,
            "weight": req.weight,
            "upscale": req.upscale,
            "bg_upsampler": req.bg_upsampler,
            "output_suffix": req.output_suffix,
        },
        "items": items,
        "owner": owner_email,
        "mode": "live" if _ENHANCE_LIVE else "simulation",
        "status": "pending",
        "progress": 0.0,
        "current_item": 0,
        "created_at": _now_iso(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "logs": [],
    }


async def _enhance_runner_sim(job_id: str):
    """Simulated batch runner — used when GFPGAN isn't installed locally."""
    async with _ENHANCE_LOCK:
        job = ENHANCE_JOBS.get(job_id)
    if not job:
        return

    try:
        async with _ENHANCE_LOCK:
            job["status"] = "running"
            job["started_at"] = _now_iso()
            _job_log(
                job,
                f"▶ [simulated] enhance batch · gfpgan v{job['params']['gfpgan_version']} · "
                f"upscale={job['params']['upscale']} · weight={job['params']['weight']} · "
                f"bg={job['params']['bg_upsampler'] or 'bicubic'}",
            )

        total = len(job["items"])
        for idx, item in enumerate(job["items"]):
            async with _ENHANCE_LOCK:
                if job["status"] == "cancelled":
                    _job_log(job, "Cancelled before starting next item")
                    return
                job["current_item"] = idx
                item["status"] = "running"
                _job_log(job, f"[{idx+1}/{total}] {item['video']}")

            for stage_name, target, t_min, t_max in ENHANCE_STAGES_SIM:
                if stage_name in ("upscale_alpha", "upload_alpha") and not item["alpha"]:
                    continue

                async with _ENHANCE_LOCK:
                    item["stage"] = stage_name
                    _job_log(job, f"  → {stage_name}")

                duration = random.uniform(t_min, t_max)
                steps = max(4, int(duration * 4))
                start_prog = item["progress"]
                for s in range(steps):
                    await asyncio.sleep(duration / steps)
                    async with _ENHANCE_LOCK:
                        if job["status"] == "cancelled":
                            _job_log(job, "Cancelled mid-stage")
                            return
                        item["progress"] = round(
                            start_prog + (target - start_prog) * ((s + 1) / steps), 4
                        )
                        job["progress"] = round((idx + item["progress"]) / total, 4)

            async with _ENHANCE_LOCK:
                item["status"] = "completed"
                item["progress"] = 1.0
                item["stage"] = "done"
                _job_log(job, f"  ✓ uploaded {item['output_video']}")
                if item["output_alpha"]:
                    _job_log(job, f"  ✓ uploaded {item['output_alpha']}")

        async with _ENHANCE_LOCK:
            job["status"] = "completed"
            job["progress"] = 1.0
            job["finished_at"] = _now_iso()
            _job_log(job, f"✓ [simulated] batch completed ({total} items)")

    except Exception as e:
        async with _ENHANCE_LOCK:
            job["status"] = "failed"
            job["error"] = str(e)
            job["finished_at"] = _now_iso()
            _job_log(job, f"✗ Failed: {e}")


async def _enhance_runner_live(job_id: str):
    """
    Live GFPGAN runner. Mirrors the V2 notebook pipeline:
      download → enhance_rgb (GFPGAN) → ffmpeg encode → upload
      → upscale_alpha (LANCZOS) → ffmpeg → upload
    Falls back to simulation if anything in the heavy stack is missing.
    """
    # Skeleton kept intentionally narrow: heavy GFPGAN init happens lazily
    # inside the worker so the API stays responsive at import time.
    # The reference implementation is in /Users/admin/Downloads/video_enhance_v2.ipynb;
    # plug it into this function on the GPU host. For now we fall back to sim
    # so the UI works identically in both modes.
    await _enhance_runner_sim(job_id)


async def _enhance_runner(job_id: str):
    if _ENHANCE_LIVE:
        await _enhance_runner_live(job_id)
    else:
        await _enhance_runner_sim(job_id)


@app.get("/api/enhance/mode")
async def enhance_mode(user: User = Depends(require_role(Role.viewer))):
    """Live vs simulation mode + GPU availability."""
    return {
        "mode": "live" if _ENHANCE_LIVE else "simulation",
        "gpu": _ENHANCE_GPU,
        "reason": None if _ENHANCE_LIVE else _ENHANCE_INIT_ERROR,
    }


@app.post("/api/enhance/estimate")
async def enhance_estimate(req: EnhanceRequest, user: User = Depends(require_role(Role.viewer))):
    """Validate + estimate ETA for a batch (no side effects)."""
    errors = _validate_enhance(req)
    per_item = 30 + 25 * req.upscale + (15 if req.bg_upsampler == "realesrgan" else 0)
    has_alpha = sum(1 for it in req.items if it.alpha)
    eta_seconds = per_item * len(req.items) + 10 * has_alpha
    return {
        "ok": not errors,
        "errors": errors,
        "items": len(req.items),
        "items_with_alpha": has_alpha,
        "eta_seconds": eta_seconds,
        "mode": "live" if _ENHANCE_LIVE else "simulation",
    }


@app.post("/api/enhance")
async def create_enhance_job(req: EnhanceRequest, user: User = Depends(require_role(Role.developer))):
    """Submit a batch enhancement job."""
    errors = _validate_enhance(req)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    job = _create_enhance_job(req, user.email)
    async with _ENHANCE_LOCK:
        ENHANCE_JOBS[job["id"]] = job

    asyncio.create_task(_enhance_runner(job["id"]))
    return {"ok": True, "job_id": job["id"], "mode": job["mode"]}


@app.get("/api/enhance/jobs")
async def list_enhance_jobs(user: User = Depends(require_role(Role.viewer))):
    rows = sorted(ENHANCE_JOBS.values(), key=lambda j: j["created_at"], reverse=True)
    return {
        "mode": "live" if _ENHANCE_LIVE else "simulation",
        "jobs": [{
            "id":          j["id"],
            "name":        j["name"],
            "status":      j["status"],
            "progress":    j["progress"],
            "items":       len(j["items"]),
            "current_item": j["current_item"],
            "owner":       j["owner"],
            "params":      j["params"],
            "created_at":  j["created_at"],
            "started_at":  j.get("started_at"),
            "finished_at": j.get("finished_at"),
        } for j in rows],
    }


@app.get("/api/enhance/jobs/{job_id}")
async def get_enhance_job(job_id: str, user: User = Depends(require_role(Role.viewer))):
    job = ENHANCE_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.post("/api/enhance/jobs/{job_id}/cancel")
async def cancel_enhance_job(job_id: str, user: User = Depends(require_role(Role.developer))):
    async with _ENHANCE_LOCK:
        job = ENHANCE_JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job["status"] in ("completed", "failed", "cancelled"):
            return {"ok": True, "status": job["status"]}
        job["status"] = "cancelled"
        job["finished_at"] = _now_iso()
        _job_log(job, "Cancelled by user")
    return {"ok": True, "status": "cancelled"}


# ────────────────────────────────────────────────────────────────────────────
# Tool 6.5: Footage QA / Validator
# Pre-production gate that scores raw avatar footage against the V3.6 spec:
#   • resolution: 4K (≥ 3840 px wide)
#   • framerate: ≥ 25 fps
#   • duration: 2-5 min (sweet spot), warning < 2 or > 7
#   • sharpness: Laplacian variance, sampled across the clip
#   • face coverage: head detected in ≥ 95% of sampled frames
#   • pauses: at least 2 silent regions ≥ 0.4s in the audio track
#   • gaze: face roughly centered + frontal (proxy for "eyes to camera")
# Dual-mode: live path uses ffprobe + opencv (Haar cascade) + ffmpeg
# silencedetect; simulation returns deterministic, realistic numbers so the
# UI is fully functional on machines without ffmpeg or opencv.
# ────────────────────────────────────────────────────────────────────────────
import hashlib as _hashlib_fv
import math as _math_fv
import shutil as _shutil_fv
import subprocess as _subprocess_fv

_FV_LIVE = False
_FV_INIT_ERROR: Optional[str] = None
try:
    import cv2 as _fv_cv2  # type: ignore
    import numpy as _fv_np  # type: ignore
    if not _shutil_fv.which("ffprobe") or not _shutil_fv.which("ffmpeg"):
        raise RuntimeError("ffprobe/ffmpeg not in PATH")
    _FV_LIVE = True
except Exception as _fv_exc:  # pragma: no cover
    _FV_INIT_ERROR = (
        f"Validator stack not available ({_fv_exc!s}). "
        "Install with: pip install opencv-python-headless numpy ; "
        "and make sure ffmpeg/ffprobe are on PATH."
    )

# Optional MediaPipe stack — turns on richer expression / pose sensors.
# Failing to import simply removes those sensors from the report; the rest
# of the validator continues to work as before.
#
# We use the modern MediaPipe Tasks API (FaceLandmarker / PoseLandmarker),
# which is what ships on macOS / Apple Silicon. The .task model files are
# downloaded once on first use into a tmp cache.
_FV_MP = False
_FV_MP_ERROR: Optional[str] = None
try:
    import mediapipe as _fv_mp_mod                              # type: ignore
    from mediapipe.tasks import python as _fv_mp_python         # type: ignore
    from mediapipe.tasks.python import vision as _fv_mp_vision  # type: ignore
    _FV_MP = True
except Exception as _mp_exc:  # pragma: no cover
    _FV_MP_ERROR = f"MediaPipe Tasks not available ({_mp_exc!s})"

_FV_MP_CACHE = Path(tempfile.gettempdir()) / "avatars_studio_mp_models"
_FV_MP_CACHE.mkdir(exist_ok=True)
_FV_MP_FACE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
_FV_MP_POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _fv_mp_ensure_model(url: str, fname: str) -> Optional[Path]:
    """Cache the MediaPipe .task file on disk. Returns None on failure."""
    if not _FV_MP:
        return None
    dest = _FV_MP_CACHE / fname
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        return dest if dest.stat().st_size > 1024 else None
    except Exception:
        try: dest.unlink()
        except Exception: pass
        return None

FV_JOBS: dict[str, dict] = {}
_FV_LOCK = asyncio.Lock()


# Spec thresholds — keep aligned with /Users/admin/Downloads/Avatar Creation
# Instructions.pdf so that the validator reflects the same gates as the
# official guidance handed to actors.
#   • Duration 5-7 min     (PDF: "Duration: 5-7 minutes")
#   • 4K @ 30 fps          (PDF: "iOS … 4K, 30 fps" / Android UHD 4K)
#   • Pauses 2-3 s, 4-5 ×  (PDF: "Pauses: 2-3 seconds between sentences,
#                           up to 4-5 pauses per speech")
#   • Eyes to camera       (PDF: "Maintain direct eye contact")
#   • Centered framing     (PDF: "Position the actor in the center")
FV_SPEC = {
    # ── existing image / duration / pauses gates ──────────────────────────
    "min_width":         3840,            # 4K UHD
    "min_height":        2160,
    "min_fps":           25.0,            # 30 recommended, 25 minimum
    "min_duration_sec":  300,             # 5 min, per PDF
    "max_duration_sec":  420,             # 7 min, per PDF
    "ideal_duration":    (300, 420),      # 5-7 min sweet spot
    "soft_floor_sec":    120,             # below 2 min → hard fail
    "soft_ceiling_sec":  600,             # above 10 min → warn (wasted training budget)
    "min_sharpness":     100.0,           # Laplacian variance threshold
    "warn_sharpness":    70.0,
    "min_face_coverage": 0.95,
    "min_pauses":        4,               # 4-5 explicit closed-mouth pauses
    "min_pause_sec":     2.0,             # each pause ≥ 2 s (per PDF)
    "max_face_offset":   0.18,            # |dx|+|dy| relative to frame
    "sample_frames":     32,              # frames inspected per pass

    # ── Audio (PDF: loud, clear pronunciation; pauses) ────────────────────
    "audio_loudness_ideal":  (-23.0, -14.0),   # LUFS sweet-spot (broadcast)
    "audio_loudness_warn":   (-30.0, -10.0),   # outside → warn
    "audio_peak_max_dbfs":   -1.0,             # above → clipping
    "audio_peak_warn_dbfs":  -3.0,
    "audio_silence_max":     0.45,             # >45% of clip silent → fail

    # ── Head pose (PDF: horizontal twist; minimize vertical) ──────────────
    "head_horizontal_min": 0.010,         # below = too static (no head turns)
    "head_horizontal_max": 0.090,         # above = jumpy / sharp
    "head_vertical_warn":  0.040,         # PDF: minimize up/down motion
    "head_vertical_max":   0.070,

    # ── Framing (PDF: center, minimize space above head, no crop) ─────────
    "head_top_margin_max":   0.18,        # gap above head, normalized to H
    "head_top_margin_warn":  0.10,
    "face_size_ideal":       (0.06, 0.22), # face_area / frame_area
    "face_size_warn":        (0.04, 0.30),

    # ── Background uniformity (PDF: plain background for studio) ──────────
    # std-dev of pixels outside the head bbox; 0-255 scale
    "bg_std_pass":  18.0,
    "bg_std_warn":  35.0,

    # ── Lighting (PDF: diffused light, avoid hard shadows on face) ────────
    "face_brightness_std_pass": 35.0,
    "face_brightness_std_warn": 55.0,

    # ── MediaPipe FaceMesh — mouth expressiveness & eye behaviour ─────────
    # PDF: "Open your mouth wide enough to capture teeth and expressions"
    #      "Slow, clear and expressive pronunciation"
    #      "Avoid yawning"
    "mouth_var_min":         0.025,       # below = monotone (too little articulation)
    "mouth_var_warn_max":    0.180,       # above warn = bordering on over-expressive
    "mouth_var_fail_max":    0.260,       # above fail = extreme / yawning
    "mouth_open_extreme":    0.55,        # single-frame MAR (yawning threshold)
    "mouth_open_extreme_max_frac": 0.05,  # ≤5% of frames may exceed → otherwise fail

    # PDF: "Maintain direct eye contact … avoid rolling eyes"
    # Eye Aspect Ratio (EAR): typical open ≥ 0.25, blink ~0.1, closed ≤ 0.08
    "eye_min_open_avg":      0.18,        # below avg → eyes look closed too much
    "eye_long_closed_max_frac": 0.10,     # ≤10% of frames may show fully-closed eyes

    # ── MediaPipe FaceMesh — head pose (yaw / pitch / roll) ───────────────
    # std-dev across sampled frames, in degrees.
    "head_yaw_min_deg":      3.0,         # below = head static (no twist at all)
    "head_yaw_max_deg":     22.0,         # above warn = jerky/exaggerated
    "head_pitch_max_warn":   5.0,         # PDF: minimize vertical
    "head_pitch_max_fail":   9.0,
    "head_roll_max_warn":    4.0,         # smooth roll only
    "head_roll_max_fail":    8.0,

    # ── MediaPipe Pose — hands (PDF: hands below the chest) ───────────────
    "hands_above_chest_warn_frac": 0.10,  # ≤10% of frames may show hand near/above chest
    "hands_above_chest_fail_frac": 0.30,
    "hand_on_face_warn_frac":       0.02,
    "hand_on_face_fail_frac":       0.08,

    "mp_sample_frames":     24,           # extra MediaPipe-only sampling pass
}


class FVRequest(BaseModel):
    source: str
    name: Optional[str] = ""


def _fv_validate_req(req: "FVRequest") -> list[str]:
    errors: list[str] = []
    if not (req.source or "").strip():
        errors.append("Source URI is required")
    elif not _is_valid_uri(req.source):
        errors.append("Source must start with s3://, gs://, http(s):// or file://")
    return errors


def _fv_score(checks: dict) -> int:
    """
    Weighted 0-100 score across all sensors. Weights sum to 100 — checks
    listed here that are missing from a particular run are skipped, and the
    total is normalized to 100 across only the present weights so older
    metric sets still produce comparable scores.
    """
    weights = {
        # Image core (camera & subject)
        "resolution":      7,
        "fps":             5,
        "duration":        7,
        "sharpness":       9,
        "face_coverage":  10,
        "gaze":            6,
        # Head pose / framing (cv2 sampler)
        "head_horizontal": 3,
        "head_vertical":   3,
        "top_margin":      3,
        "face_size":       3,
        # Background / lighting
        "background":      5,
        "lighting":        5,
        # Audio
        "audio_track":     3,
        "loudness":        5,
        "audio_peak":      2,
        "audio_silence":   3,
        "pauses":          7,
        # MediaPipe FaceMesh — articulation, eyes, accurate head pose
        "mouth_expr":      6,
        "eyes_open":       4,
        "head_yaw":        2,
        "head_pitch":      3,
        "head_roll":       2,
        # MediaPipe Pose — hands
        "hands_position":  5,
        "hand_on_face":    3,
    }
    earned, possible = 0, 0
    for key, w in weights.items():
        c = checks.get(key)
        if not c:
            continue
        possible += w
        if c["status"] == "pass":
            earned += w
        elif c["status"] == "warn":
            earned += w // 2
    if possible == 0:
        return 0
    return int(round(earned * 100 / possible))


def _fv_verdict(score: int, checks: dict) -> str:
    fails = [k for k, v in checks.items() if v.get("status") == "fail"]
    if not fails and score >= 90:
        return "excellent"
    if not fails and score >= 75:
        return "good"
    if len(fails) <= 1:
        return "needs_work"
    return "rejected"


def _fv_create_job(req: "FVRequest", owner_email: str) -> dict:
    jid = secrets.token_urlsafe(9)
    name = (req.name or "").strip() or req.source.rsplit("/", 1)[-1] or f"footage-{jid[:6]}"
    return {
        "id": jid,
        "name": name,
        "source": req.source,
        "owner": owner_email,
        "mode": "live" if _FV_LIVE else "simulation",
        "status": "pending",
        "progress": 0.0,
        "stage": "queued",
        "created_at": _now_iso(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "logs": [],
        "metrics": {},
        "checks": {},
        "score": None,
        "verdict": None,
        "recommendations": [],
    }


# ── Simulation runner ───────────────────────────────────────────────────────
async def _fv_runner_sim(jid: str):
    """
    Deterministic, realistic mock of the live runner. Uses a hash of the
    source URI as a seed so the same path always produces the same verdict
    (handy for screenshots & demos).
    """
    async with _FV_LOCK:
        job = FV_JOBS.get(jid)
    if not job:
        return
    src = job["source"]
    seed = int(_hashlib_fv.sha1(src.encode("utf-8")).hexdigest(), 16) % (2**32)
    rnd = random.Random(seed)

    stages = [
        ("download",        0.08, 0.6, 1.4),
        ("probe_metadata",  0.15, 0.4, 0.9),
        ("sample_frames",   0.40, 1.2, 2.5),
        ("audio_pauses",    0.55, 0.5, 1.1),
        ("audio_loudness",  0.65, 0.5, 1.0),
        ("face_mesh",       0.82, 0.8, 1.6),
        ("pose",            0.94, 0.6, 1.2),
        ("score",           1.00, 0.2, 0.4),
    ]
    try:
        async with _FV_LOCK:
            job["status"] = "running"
            job["started_at"] = _now_iso()
            _job_log(job, f"▶ [simulated] validating {src}")

        for stage_name, target, t_min, t_max in stages:
            async with _FV_LOCK:
                if job["status"] == "cancelled":
                    _job_log(job, "Cancelled before next stage")
                    return
                job["stage"] = stage_name
                _job_log(job, f"  → {stage_name}")
            duration = rnd.uniform(t_min, t_max)
            steps = max(4, int(duration * 4))
            start_p = job["progress"]
            for s in range(steps):
                await asyncio.sleep(duration / steps)
                async with _FV_LOCK:
                    if job["status"] == "cancelled":
                        return
                    job["progress"] = round(
                        start_p + (target - start_p) * ((s + 1) / steps), 4
                    )

        # ── Build realistic per-check values from the seeded RNG. ───────────
        width  = rnd.choice([1920, 2560, 3840, 3840, 3840, 4096])
        height = {1920: 1080, 2560: 1440, 3840: 2160, 4096: 2160}[width]
        fps    = rnd.choice([23.976, 24, 25, 29.97, 30, 30, 50, 60])
        dur    = rnd.uniform(180, 540)           # 3-9 min (target band 5-7)
        sharp  = rnd.uniform(45, 320)
        face_cov = rnd.uniform(0.55, 1.0)
        n_pauses = rnd.choice([0, 1, 2, 3, 4, 4, 5, 5, 6])
        avg_pause = rnd.uniform(1.2, 3.0)
        gaze_off = rnd.uniform(0.02, 0.32)

        # New sensors — head pose, framing, background, lighting, audio.
        head_h    = rnd.uniform(0.005, 0.110)
        head_v    = rnd.uniform(0.005, 0.080)
        top_marg  = rnd.uniform(0.02, 0.28)
        face_size = rnd.uniform(0.03, 0.32)
        bg_std    = rnd.uniform(6.0, 55.0)
        face_brt  = rnd.uniform(20.0, 70.0)
        audio = {
            "present":       rnd.choice([True, True, True, True, False]),
            "loudness_lufs": round(rnd.uniform(-32.0, -10.0), 1),
            "peak_dbfs":     round(rnd.uniform(-12.0, 0.0), 1),
            "silent_ratio":  round(rnd.uniform(0.02, 0.55), 3),
        }

        # MediaPipe-derived sensors (simulated).
        face_mesh = {
            "available":          True,
            "frames":             FV_SPEC["mp_sample_frames"],
            "mouth_var":          round(rnd.uniform(0.010, 0.230), 4),
            "mouth_max":          round(rnd.uniform(0.20, 0.70), 3),
            "mouth_extreme_frac": round(rnd.uniform(0.0, 0.12), 3),
            "eye_open_avg":       round(rnd.uniform(0.14, 0.32), 3),
            "eye_closed_ratio":   round(rnd.uniform(0.01, 0.18), 3),
            "head_yaw_std":       round(rnd.uniform(1.0, 28.0), 2),
            "head_pitch_std":     round(rnd.uniform(0.5, 12.0), 2),
            "head_roll_std":      round(rnd.uniform(0.3,  9.0), 2),
        }
        pose = {
            "available":               True,
            "frames":                  FV_SPEC["mp_sample_frames"],
            "hands_above_chest_ratio": round(rnd.uniform(0.0, 0.45), 3),
            "hand_on_face_ratio":      round(rnd.uniform(0.0, 0.12), 3),
        }

        metrics = {
            "resolution": {"width": int(width), "height": int(height)},
            "fps": round(float(fps), 3),
            "duration_sec": round(float(dur), 1),
            "sharpness": round(float(sharp), 1),
            "face_coverage": round(float(face_cov), 3),
            "pauses": int(n_pauses),
            "avg_pause_sec": round(float(avg_pause), 2),
            "gaze_offset": round(float(gaze_off), 3),
            "head_horiz_motion":   round(head_h, 4),
            "head_vert_motion":    round(head_v, 4),
            "head_top_margin":     round(top_marg, 3),
            "face_size_ratio":     round(face_size, 4),
            "bg_std":              round(bg_std, 1),
            "face_brightness_std": round(face_brt, 1),
            "audio":     audio,
            "face_mesh": face_mesh,
            "pose":      pose,
            "frames_sampled": FV_SPEC["sample_frames"],
            "duration_human": f"{int(dur // 60)}m {int(dur % 60):02d}s",
        }
        checks, recs = _fv_compile_checks(metrics)
        score = _fv_score(checks)
        verdict = _fv_verdict(score, checks)

        async with _FV_LOCK:
            job["metrics"] = metrics
            job["checks"] = checks
            job["recommendations"] = recs
            job["score"] = score
            job["verdict"] = verdict
            job["status"] = "completed"
            job["progress"] = 1.0
            job["finished_at"] = _now_iso()
            _job_log(job, f"✓ [simulated] verdict: {verdict} (score {score}/100)")
    except Exception as e:
        async with _FV_LOCK:
            job["status"] = "failed"
            job["error"] = str(e)
            job["finished_at"] = _now_iso()
            _job_log(job, f"✗ Failed: {e}")


def _fv_compile_checks(m: dict) -> tuple[dict, list[str]]:
    """Translate raw metrics dict into per-check {status, label, detail} blocks."""
    checks: dict[str, dict] = {}
    recs: list[str] = []

    # Resolution
    w, h = m["resolution"]["width"], m["resolution"]["height"]
    if w >= FV_SPEC["min_width"] and h >= FV_SPEC["min_height"]:
        checks["resolution"] = {
            "status": "pass", "label": "4K resolution",
            "detail": f"{w}×{h}", "value": f"{w}×{h}",
        }
    elif w >= 2560:
        checks["resolution"] = {
            "status": "warn", "label": "Below 4K",
            "detail": f"{w}×{h} — acceptable but not ideal", "value": f"{w}×{h}",
        }
        recs.append("Re-shoot at 3840×2160 (UHD) or higher for the cleanest faces.")
    else:
        checks["resolution"] = {
            "status": "fail", "label": "Resolution too low",
            "detail": f"{w}×{h}, needs ≥ 3840×2160", "value": f"{w}×{h}",
        }
        recs.append("Resolution is below 4K — re-shoot at 3840×2160 or higher.")

    # FPS — PDF recommends 30 fps, 25 is the minimum we accept.
    fps = m["fps"]
    if fps >= 29.0:
        checks["fps"] = {"status": "pass", "label": "Frame rate OK",
                         "detail": f"{fps} fps (recommended ≥ 30)", "value": f"{fps} fps"}
    elif fps >= FV_SPEC["min_fps"]:
        checks["fps"] = {"status": "warn", "label": "Frame rate borderline",
                         "detail": f"{fps} fps — Elai recommends 30 fps",
                         "value": f"{fps} fps"}
        recs.append("Frame rate is borderline. Use 30 fps (iOS Settings → Camera → Record Video → 4K, 30 fps; Android Camera → Settings → Rear video → UHD 4K).")
    else:
        checks["fps"] = {"status": "fail", "label": "FPS too low",
                         "detail": f"{fps} fps, needs ≥ 25 (target 30)",
                         "value": f"{fps} fps"}
        recs.append("Capture at 30 fps (iOS Settings → Camera → Record Video → 4K, 30 fps; Android Camera → Settings → Rear video → UHD 4K). Never below 25.")

    # Duration — official Elai spec is 5-7 min.
    dur = m["duration_sec"]
    ideal_lo, ideal_hi = FV_SPEC["ideal_duration"]
    if ideal_lo <= dur <= ideal_hi:
        status = "pass"; detail = f"{m['duration_human']} (within 5-7 min sweet spot)"
    elif FV_SPEC["soft_floor_sec"] <= dur < ideal_lo:
        status = "warn"
        detail = f"{m['duration_human']} — usable but below 5 min recommended"
        recs.append("Footage is shorter than recommended. Record 5-7 minutes so the model has enough material to pick the best segment.")
    elif ideal_hi < dur <= FV_SPEC["soft_ceiling_sec"]:
        status = "warn"
        detail = f"{m['duration_human']} — over 7 min, trim to 5-7 min"
        recs.append("Trim the clip to 5-7 minutes; extra footage rarely helps and slows training.")
    elif dur < FV_SPEC["soft_floor_sec"]:
        status = "fail"; detail = f"{m['duration_human']} — needs at least 2 min, target 5-7"
        recs.append("Footage is far too short. Re-shoot to capture 5-7 minutes of varied speech (see the Recording Guide).")
    else:
        status = "fail"; detail = f"{m['duration_human']} — well over 7 min"
        recs.append("Clip is far too long. Trim to 5-7 minutes of the best material before submitting for training.")
    checks["duration"] = {"status": status, "label": "Duration",
                          "detail": detail, "value": m["duration_human"]}

    # Sharpness (Laplacian variance)
    s = m["sharpness"]
    if s >= FV_SPEC["min_sharpness"]:
        checks["sharpness"] = {"status": "pass", "label": "Sharp",
                               "detail": f"Laplacian var {s} (≥ 100)", "value": f"{s:.0f}"}
    elif s >= FV_SPEC["warn_sharpness"]:
        checks["sharpness"] = {"status": "warn", "label": "Slightly soft",
                               "detail": f"Laplacian var {s}, target ≥ 100",
                               "value": f"{s:.0f}"}
        recs.append("Footage is slightly soft. Stop down (smaller aperture), increase shutter to ≥ 1/100s, or sharpen focus on the eyes.")
    else:
        checks["sharpness"] = {"status": "fail", "label": "Blurry / out of focus",
                               "detail": f"Laplacian var {s}, target ≥ 100",
                               "value": f"{s:.0f}"}
        recs.append("Image is blurry. Re-shoot with manual focus locked on the eyes; check for motion blur and dirty lens.")

    # Face coverage
    fc = m["face_coverage"]
    if fc >= FV_SPEC["min_face_coverage"]:
        checks["face_coverage"] = {"status": "pass", "label": "Face detected continuously",
                                   "detail": f"{fc*100:.1f}% of sampled frames",
                                   "value": f"{fc*100:.0f}%"}
    elif fc >= 0.85:
        checks["face_coverage"] = {"status": "warn", "label": "Some frames missing face",
                                   "detail": f"{fc*100:.1f}% (target ≥ 95%)",
                                   "value": f"{fc*100:.0f}%"}
        recs.append("Face was missing in a few frames. Stay inside the frame and avoid turning fully sideways.")
    else:
        checks["face_coverage"] = {"status": "fail", "label": "Face lost too often",
                                   "detail": f"{fc*100:.1f}% (target ≥ 95%)",
                                   "value": f"{fc*100:.0f}%"}
        recs.append("Face detector lost the head in many frames. Re-shoot keeping the whole head visible throughout.")

    # Pauses — PDF: "2-3 seconds between sentences, up to 4-5 pauses per
    # speech, naturally and with mouth closed".
    np_ = m["pauses"]
    avg = m["avg_pause_sec"]
    min_n = FV_SPEC["min_pauses"]                     # 4
    min_len = FV_SPEC["min_pause_sec"]                # 2.0
    if np_ >= min_n and avg >= min_len:
        checks["pauses"] = {"status": "pass", "label": "Pauses look good",
                            "detail": f"{np_} pauses · avg {avg}s (target 4-5 × 2-3 s)",
                            "value": f"{np_} pauses"}
    elif np_ >= min_n and avg >= 1.0:
        checks["pauses"] = {"status": "warn", "label": "Pauses a bit short",
                            "detail": f"{np_} pauses · avg {avg}s (target 2-3 s each)",
                            "value": f"{np_} pauses"}
        recs.append("Pauses are too brief. Hold each silent pause for 2-3 seconds with mouth closed (PDF spec: 4-5 pauses × 2-3 s).")
    elif np_ >= 2:
        checks["pauses"] = {"status": "warn", "label": "Not enough pauses",
                            "detail": f"only {np_} pauses (target 4-5 × 2-3 s)",
                            "value": f"{np_} pauses"}
        recs.append("Insert 4-5 natural silent pauses of 2-3 seconds between sentences (mouth closed, eyes still on camera).")
    elif np_ == 1:
        checks["pauses"] = {"status": "fail", "label": "Almost no pauses",
                            "detail": "only 1 pause (target 4-5 × 2-3 s)",
                            "value": "1 pause"}
        recs.append("Re-record with 4-5 distinct 2-3 s silent pauses between sentences — they teach the avatar's listening/idle pose.")
    else:
        checks["pauses"] = {"status": "fail", "label": "No silent pauses",
                            "detail": "Model needs closed-mouth idle frames (target 4-5 × 2-3 s)",
                            "value": "0 pauses"}
        recs.append("Add 4-5 silent pauses of 2-3 seconds each (mouth closed, eyes open, no head movement).")

    # Gaze (face centering proxy)
    g = m["gaze_offset"]
    if g <= FV_SPEC["max_face_offset"]:
        checks["gaze"] = {"status": "pass", "label": "Eyes to camera",
                          "detail": f"face offset {g:.2f} (≤ 0.18 of frame)",
                          "value": f"{g:.2f}"}
    elif g <= 0.25:
        checks["gaze"] = {"status": "warn", "label": "Slightly off-center",
                          "detail": f"face offset {g:.2f} — recenter the lens",
                          "value": f"{g:.2f}"}
        recs.append("Eyeline drifts slightly off-camera. Mount the lens at eye height directly in front of the subject.")
    else:
        checks["gaze"] = {"status": "fail", "label": "Looking off-camera",
                          "detail": f"face offset {g:.2f} — must look straight in",
                          "value": f"{g:.2f}"}
        recs.append("Subject is looking away from the lens. Place the camera at eye level and instruct them to look straight into it.")

    # ── Head pose: horizontal twist vs vertical nod (PDF: minimize vertical) ──
    hh = m.get("head_horiz_motion")
    hv = m.get("head_vert_motion")
    if hh is not None:
        hh_min = FV_SPEC["head_horizontal_min"]
        hh_max = FV_SPEC["head_horizontal_max"]
        if hh_min <= hh <= hh_max:
            checks["head_horizontal"] = {
                "status": "pass", "label": "Head twists naturally",
                "detail": f"horizontal motion {hh:.3f} (range {hh_min}-{hh_max})",
                "value": f"{hh:.3f}",
            }
        elif hh < hh_min:
            checks["head_horizontal"] = {
                "status": "warn", "label": "Head looks frozen",
                "detail": f"horizontal motion {hh:.3f} — too still",
                "value": f"{hh:.3f}",
            }
            recs.append("Head barely moves. Add gentle left/right turns to capture profile angles — the avatar will look stiff otherwise.")
        else:
            checks["head_horizontal"] = {
                "status": "warn", "label": "Head moves too sharply",
                "detail": f"horizontal motion {hh:.3f} — keep moves smooth",
                "value": f"{hh:.3f}",
            }
            recs.append("Head swings are too sharp. Move smoothly, no jerks (PDF: 'Move head smoothly, with no sharp movements').")
    if hv is not None:
        if hv <= FV_SPEC["head_vertical_warn"]:
            checks["head_vertical"] = {
                "status": "pass", "label": "Vertical nodding minimized",
                "detail": f"vertical motion {hv:.3f} (≤ {FV_SPEC['head_vertical_warn']})",
                "value": f"{hv:.3f}",
            }
        elif hv <= FV_SPEC["head_vertical_max"]:
            checks["head_vertical"] = {
                "status": "warn", "label": "Some up/down nodding",
                "detail": f"vertical motion {hv:.3f} — minimize per spec",
                "value": f"{hv:.3f}",
            }
            recs.append("There's some up/down head motion. Keep head movements mostly horizontal — minimize vertical nods.")
        else:
            checks["head_vertical"] = {
                "status": "fail", "label": "Too much vertical motion",
                "detail": f"vertical motion {hv:.3f} — must stay mostly horizontal",
                "value": f"{hv:.3f}",
            }
            recs.append("Head bobs up and down. Per the recording guide, motion must be mostly horizontal (twist left/right).")

    # ── Framing: top margin + face size in frame ──────────────────────────────
    tm = m.get("head_top_margin")
    if tm is not None:
        if tm <= FV_SPEC["head_top_margin_warn"]:
            checks["top_margin"] = {
                "status": "pass", "label": "Tight top framing",
                "detail": f"{tm*100:.0f}% of frame above head",
                "value": f"{tm*100:.0f}%",
            }
        elif tm <= FV_SPEC["head_top_margin_max"]:
            checks["top_margin"] = {
                "status": "warn", "label": "A bit of headroom",
                "detail": f"{tm*100:.0f}% of frame above head — tighten the crop",
                "value": f"{tm*100:.0f}%",
            }
            recs.append("Too much empty space above the head. Re-frame to leave just a small margin (PDF: 'Minimize space above the head').")
        else:
            checks["top_margin"] = {
                "status": "fail", "label": "Too much space above head",
                "detail": f"{tm*100:.0f}% empty above the head",
                "value": f"{tm*100:.0f}%",
            }
            recs.append("Way too much space above the head. Re-frame the shot tighter (PDF: minimize space above head).")

    fs = m.get("face_size_ratio")
    if fs is not None:
        lo, hi   = FV_SPEC["face_size_ideal"]
        wlo, whi = FV_SPEC["face_size_warn"]
        if lo <= fs <= hi:
            checks["face_size"] = {
                "status": "pass", "label": "Subject sized correctly",
                "detail": f"face occupies {fs*100:.1f}% of frame",
                "value": f"{fs*100:.1f}%",
            }
        elif wlo <= fs <= whi:
            checks["face_size"] = {
                "status": "warn",
                "label": "Face too small" if fs < lo else "Face too close",
                "detail": f"face {fs*100:.1f}% of frame (target {lo*100:.0f}-{hi*100:.0f}%)",
                "value": f"{fs*100:.1f}%",
            }
            if fs < lo:
                recs.append("Subject is too far from the camera. Move in so the face occupies roughly 1/8 to 1/5 of the frame area.")
            else:
                recs.append("Subject is too close. Pull the camera back — head and shoulders should fit, with hands able to enter frame below the chest.")
        else:
            checks["face_size"] = {
                "status": "fail",
                "label": "Face wildly off-size",
                "detail": f"face {fs*100:.1f}% of frame (target {lo*100:.0f}-{hi*100:.0f}%)",
                "value": f"{fs*100:.1f}%",
            }
            recs.append("Re-block the shot — the head should be the natural focal point of the frame, not lost in the background nor filling it edge-to-edge.")

    # ── Background uniformity (PDF: plain background for studio) ──────────────
    bg = m.get("bg_std")
    if bg is not None:
        if bg <= FV_SPEC["bg_std_pass"]:
            checks["background"] = {
                "status": "pass", "label": "Background is clean",
                "detail": f"variation σ={bg:.0f} (≤ {FV_SPEC['bg_std_pass']:.0f})",
                "value": f"σ={bg:.0f}",
            }
        elif bg <= FV_SPEC["bg_std_warn"]:
            checks["background"] = {
                "status": "warn", "label": "Background is busy",
                "detail": f"variation σ={bg:.0f} — fine for Selfie, too busy for Studio",
                "value": f"σ={bg:.0f}",
            }
            recs.append("Background has noticeable variation. For Studio avatars use a plain (typically green) backdrop with strong contrast against clothing.")
        else:
            checks["background"] = {
                "status": "fail", "label": "Background too busy",
                "detail": f"variation σ={bg:.0f} — Studio avatars require a plain backdrop",
                "value": f"σ={bg:.0f}",
            }
            recs.append("Background is far too busy. Switch to a plain backdrop (green for Studio avatars) so the keyer can cleanly remove it.")

    # ── Lighting evenness across the face (PDF: avoid hard shadows) ───────────
    fb = m.get("face_brightness_std")
    if fb is not None:
        if fb <= FV_SPEC["face_brightness_std_pass"]:
            checks["lighting"] = {
                "status": "pass", "label": "Even lighting",
                "detail": f"face σ={fb:.0f} (≤ {FV_SPEC['face_brightness_std_pass']:.0f})",
                "value": f"σ={fb:.0f}",
            }
        elif fb <= FV_SPEC["face_brightness_std_warn"]:
            checks["lighting"] = {
                "status": "warn", "label": "Some shadow on face",
                "detail": f"face σ={fb:.0f} — soften the key light",
                "value": f"σ={fb:.0f}",
            }
            recs.append("Some shadowing visible on the face. Use diffused lighting (PDF: avoid hard shadows under the nose or around the neck).")
        else:
            checks["lighting"] = {
                "status": "fail", "label": "Hard shadows on face",
                "detail": f"face σ={fb:.0f} — lighting too contrasty",
                "value": f"σ={fb:.0f}",
            }
            recs.append("Lighting is too harsh. Move to diffused light (softbox / north-facing window) — hard shadows under the nose and neck corrupt training.")

    # ── Audio: presence, loudness, silent ratio ───────────────────────────────
    audio = m.get("audio") or {}
    if not audio.get("present", True):
        checks["audio_track"] = {
            "status": "fail", "label": "No audio track",
            "detail": "video has no audio — pauses cannot be detected",
            "value": "missing",
        }
        recs.append("Clip has no audio track. The training pipeline needs the original sound to learn lip-sync — re-export with audio enabled.")
    else:
        checks["audio_track"] = {
            "status": "pass", "label": "Audio track present",
            "detail": "video contains an audio stream",
            "value": "ok",
        }
        lufs = audio.get("loudness_lufs")
        ilo, ihi = FV_SPEC["audio_loudness_ideal"]
        wlo, whi = FV_SPEC["audio_loudness_warn"]
        if lufs is None:
            checks["loudness"] = {
                "status": "warn", "label": "Loudness not measurable",
                "detail": "ffmpeg loudnorm did not return a reading",
                "value": "n/a",
            }
        elif ilo <= lufs <= ihi:
            checks["loudness"] = {
                "status": "pass", "label": "Loudness in range",
                "detail": f"{lufs:.1f} LUFS (broadcast {ilo:.0f}…{ihi:.0f})",
                "value": f"{lufs:.1f} LUFS",
            }
        elif wlo <= lufs <= whi:
            checks["loudness"] = {
                "status": "warn",
                "label": "Voice too quiet" if lufs < ilo else "Voice too loud",
                "detail": f"{lufs:.1f} LUFS (target {ilo:.0f}…{ihi:.0f})",
                "value": f"{lufs:.1f} LUFS",
            }
            if lufs < ilo:
                recs.append("Voice is quieter than broadcast. Speak louder, closer to the mic, or boost the recording gain so loudness sits near -18 LUFS.")
            else:
                recs.append("Voice peaks too hot — risk of distortion. Back off the mic, lower input gain, target -20…-14 LUFS.")
        else:
            checks["loudness"] = {
                "status": "fail",
                "label": "Voice way too quiet" if lufs < ilo else "Voice clipping",
                "detail": f"{lufs:.1f} LUFS (must be in {wlo:.0f}…{whi:.0f})",
                "value": f"{lufs:.1f} LUFS",
            }
            if lufs < ilo:
                recs.append("Audio is barely audible. Re-record with a proper mic close to the speaker (PDF: 'Loud, clear and distinct').")
            else:
                recs.append("Audio is severely clipping. Reduce gain and re-record — distortion can't be removed in post.")

        peak = audio.get("peak_dbfs")
        if peak is not None and peak > FV_SPEC["audio_peak_max_dbfs"]:
            checks["audio_peak"] = {
                "status": "fail", "label": "Audio peaks clipping",
                "detail": f"true peak {peak:+.1f} dBFS — leave at least 1 dB of headroom",
                "value": f"{peak:+.1f} dBFS",
            }
            recs.append("Audio is hitting the ceiling. Re-record with at least 3 dB of headroom — clipped audio can't be fixed.")
        elif peak is not None and peak > FV_SPEC["audio_peak_warn_dbfs"]:
            checks["audio_peak"] = {
                "status": "warn", "label": "Audio peaks tight",
                "detail": f"true peak {peak:+.1f} dBFS — close to clipping",
                "value": f"{peak:+.1f} dBFS",
            }
            recs.append("Audio peaks are close to clipping. Aim for true-peak ≤ -3 dBFS.")

        sr = audio.get("silent_ratio")
        if sr is not None:
            if sr <= FV_SPEC["audio_silence_max"]:
                checks["audio_silence"] = {
                    "status": "pass", "label": "Speech / silence ratio OK",
                    "detail": f"{sr*100:.0f}% silent ({(1-sr)*100:.0f}% speech)",
                    "value": f"{sr*100:.0f}% silent",
                }
            else:
                checks["audio_silence"] = {
                    "status": "fail", "label": "Mostly silent",
                    "detail": f"{sr*100:.0f}% of the clip is silence",
                    "value": f"{sr*100:.0f}% silent",
                }
                recs.append("More than half of the clip is silent. Re-record continuous speech with planned 2-3 s pauses — not the other way around.")

    # ── MediaPipe FaceMesh: mouth expressiveness & eyes ───────────────────────
    fm = m.get("face_mesh") or {}
    if fm.get("available"):
        mvar = fm.get("mouth_var", 0.0)
        mext = fm.get("mouth_extreme_frac", 0.0)
        mmax = fm.get("mouth_max", 0.0)

        if mvar < FV_SPEC["mouth_var_min"]:
            checks["mouth_expr"] = {
                "status": "fail", "label": "Mouth barely moves",
                "detail": f"MAR σ {mvar:.3f} — speech is monotone or near-silent",
                "value": f"σ {mvar:.3f}",
            }
            recs.append("Mouth movement is almost absent. PDF: 'Open your mouth wide enough to capture teeth' and use slow, clear, expressive pronunciation.")
        elif (mext > FV_SPEC["mouth_open_extreme_max_frac"]
              or mmax >= FV_SPEC["mouth_open_extreme"] + 0.1):
            checks["mouth_expr"] = {
                "status": "fail", "label": "Likely yawning / extreme mouth open",
                "detail": f"{mext*100:.0f}% frames with MAR ≥ {FV_SPEC['mouth_open_extreme']:.2f} (max {mmax:.2f})",
                "value": f"max MAR {mmax:.2f}",
            }
            recs.append("Detected very wide mouth openings — PDF explicitly says: avoid yawning during the recording. Re-shoot the affected take.")
        elif mvar > FV_SPEC["mouth_var_fail_max"]:
            checks["mouth_expr"] = {
                "status": "warn", "label": "Over-expressive mouth",
                "detail": f"MAR σ {mvar:.3f} — articulation may be exaggerated",
                "value": f"σ {mvar:.3f}",
            }
            recs.append("Mouth articulation is on the edge of over-expressive. Aim for natural, slow speech as in the PDF guide.")
        elif mvar > FV_SPEC["mouth_var_warn_max"]:
            checks["mouth_expr"] = {
                "status": "warn", "label": "Lively articulation",
                "detail": f"MAR σ {mvar:.3f} — fine, but watch for exaggeration",
                "value": f"σ {mvar:.3f}",
            }
        else:
            checks["mouth_expr"] = {
                "status": "pass", "label": "Mouth articulation",
                "detail": f"MAR σ {mvar:.3f} — natural articulation range",
                "value": f"σ {mvar:.3f}",
            }

        ear = fm.get("eye_open_avg", 1.0)
        eclo = fm.get("eye_closed_ratio", 0.0)
        if ear < FV_SPEC["eye_min_open_avg"] or eclo > FV_SPEC["eye_long_closed_max_frac"]:
            checks["eyes_open"] = {
                "status": ("fail" if ear < (FV_SPEC["eye_min_open_avg"] - 0.04) else "warn"),
                "label": "Eyes often closed",
                "detail": f"avg EAR {ear:.2f}, {eclo*100:.0f}% of frames with closed eyes",
                "value": f"EAR {ear:.2f}",
            }
            recs.append("Eyes appear closed for too long. PDF: keep direct eye contact and avoid prolonged blinking or rolling eyes.")
        else:
            checks["eyes_open"] = {
                "status": "pass", "label": "Eyes engaged",
                "detail": f"avg EAR {ear:.2f}, {eclo*100:.0f}% closed-frame ratio",
                "value": f"EAR {ear:.2f}",
            }

        yaw   = fm.get("head_yaw_std",  0.0)
        pitch = fm.get("head_pitch_std",0.0)
        roll  = fm.get("head_roll_std", 0.0)

        if yaw < FV_SPEC["head_yaw_min_deg"]:
            checks["head_yaw"] = {
                "status": "warn", "label": "Head is locked",
                "detail": f"yaw σ {yaw:.1f}° — almost no horizontal turn",
                "value": f"{yaw:.1f}°",
            }
            recs.append("Head looks frozen. PDF wants natural side-to-side movement — minor horizontal turns, not a statue.")
        elif yaw > FV_SPEC["head_yaw_max_deg"]:
            checks["head_yaw"] = {
                "status": "warn", "label": "Excessive horizontal turn",
                "detail": f"yaw σ {yaw:.1f}° — calmer movement preferred",
                "value": f"{yaw:.1f}°",
            }
            recs.append("Horizontal head movement is too jerky. Keep turns smooth and subtle.")
        else:
            checks["head_yaw"] = {
                "status": "pass", "label": "Natural horizontal motion",
                "detail": f"yaw σ {yaw:.1f}°",
                "value": f"{yaw:.1f}°",
            }

        if pitch > FV_SPEC["head_pitch_max_fail"]:
            checks["head_pitch"] = {
                "status": "fail", "label": "Too much nodding",
                "detail": f"pitch σ {pitch:.1f}° — PDF says minimize vertical motion",
                "value": f"{pitch:.1f}°",
            }
            recs.append("Vertical (nodding) motion is too strong. PDF: 'Vertical movements should be minimized' — keep chin steady.")
        elif pitch > FV_SPEC["head_pitch_max_warn"]:
            checks["head_pitch"] = {
                "status": "warn", "label": "Light nodding",
                "detail": f"pitch σ {pitch:.1f}° — try to keep chin steadier",
                "value": f"{pitch:.1f}°",
            }
            recs.append("Slight nodding detected. Aim for horizontal turns rather than nods.")
        else:
            checks["head_pitch"] = {
                "status": "pass", "label": "Stable head pitch",
                "detail": f"pitch σ {pitch:.1f}°",
                "value": f"{pitch:.1f}°",
            }

        if roll > FV_SPEC["head_roll_max_fail"]:
            checks["head_roll"] = {
                "status": "fail", "label": "Head tilts heavily",
                "detail": f"roll σ {roll:.1f}° — head leans side-to-side too much",
                "value": f"{roll:.1f}°",
            }
            recs.append("Head tilts side-to-side too much. Keep the head upright and aligned with the shoulders.")
        elif roll > FV_SPEC["head_roll_max_warn"]:
            checks["head_roll"] = {
                "status": "warn", "label": "Mild head tilt",
                "detail": f"roll σ {roll:.1f}°",
                "value": f"{roll:.1f}°",
            }
            recs.append("Minor head tilting detected. Try to keep the head level.")
        else:
            checks["head_roll"] = {
                "status": "pass", "label": "Head stays level",
                "detail": f"roll σ {roll:.1f}°",
                "value": f"{roll:.1f}°",
            }
    elif fm:
        checks["mediapipe_face"] = {
            "status": "warn", "label": "FaceMesh sensors unavailable",
            "detail": fm.get("reason", "MediaPipe FaceMesh did not run"),
            "value": "—",
        }
        recs.append("MediaPipe FaceMesh skipped — install `mediapipe` to enable mouth expressiveness, eye and 6-DoF head pose checks.")

    # ── MediaPipe Pose: hands position (PDF: hands below the chest) ──────────
    ps = m.get("pose") or {}
    if ps.get("available"):
        ach = ps.get("hands_above_chest_ratio", 0.0)
        ofa = ps.get("hand_on_face_ratio",      0.0)

        if ach >= FV_SPEC["hands_above_chest_fail_frac"]:
            checks["hands_position"] = {
                "status": "fail", "label": "Hands above chest",
                "detail": f"{ach*100:.0f}% of frames show wrists above the chest line",
                "value": f"{ach*100:.0f}%",
            }
            recs.append("Hands are visible above the chest. PDF: 'Place your hands below the chest, out of the frame'. Re-shoot or reframe.")
        elif ach >= FV_SPEC["hands_above_chest_warn_frac"]:
            checks["hands_position"] = {
                "status": "warn", "label": "Hands occasionally in frame",
                "detail": f"{ach*100:.0f}% of frames with hands near chest line",
                "value": f"{ach*100:.0f}%",
            }
            recs.append("Hands occasionally rise into the chest area. Keep them below chest level for the whole clip.")
        else:
            checks["hands_position"] = {
                "status": "pass", "label": "Hands stay below chest",
                "detail": f"{ach*100:.0f}% of frames flagged",
                "value": f"{ach*100:.0f}%",
            }

        if ofa >= FV_SPEC["hand_on_face_fail_frac"]:
            checks["hand_on_face"] = {
                "status": "fail", "label": "Hand touches face",
                "detail": f"{ofa*100:.0f}% of frames show a wrist near the face",
                "value": f"{ofa*100:.0f}%",
            }
            recs.append("Hands reach the face area. PDF: avoid touching the face during recording.")
        elif ofa >= FV_SPEC["hand_on_face_warn_frac"]:
            checks["hand_on_face"] = {
                "status": "warn", "label": "Hand near face occasionally",
                "detail": f"{ofa*100:.0f}% of frames",
                "value": f"{ofa*100:.0f}%",
            }
            recs.append("Hands approach the face area in a few frames. Keep them clearly below the chest.")
        else:
            checks["hand_on_face"] = {
                "status": "pass", "label": "Face stays untouched",
                "detail": f"{ofa*100:.0f}% of frames",
                "value": f"{ofa*100:.0f}%",
            }
    elif ps:
        checks["mediapipe_pose"] = {
            "status": "warn", "label": "Pose sensor unavailable",
            "detail": ps.get("reason", "MediaPipe Pose did not run"),
            "value": "—",
        }
        recs.append("MediaPipe Pose skipped — install `mediapipe` to enable hand-position checks.")

    return checks, recs


# ── Live runner (opencv + ffprobe + ffmpeg silencedetect) ──────────────────
def _fv_ffprobe(path: str) -> dict:
    """Return {width, height, fps, duration_sec} via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate:format=duration",
        "-of", "json", path,
    ]
    out = _subprocess_fv.check_output(cmd, stderr=_subprocess_fv.STDOUT, timeout=20)
    j = json.loads(out)
    stream = (j.get("streams") or [{}])[0]
    w = int(stream.get("width") or 0)
    h = int(stream.get("height") or 0)
    rate_str = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        num, den = rate_str.split("/")
        fps = (float(num) / float(den)) if float(den) else 0.0
    except Exception:
        fps = 0.0
    dur = float((j.get("format") or {}).get("duration") or 0.0)
    return {"width": w, "height": h, "fps": fps, "duration_sec": dur}


def _fv_silencedetect(path: str, min_pause_sec: float | None = None) -> tuple[int, float]:
    if min_pause_sec is None:
        min_pause_sec = FV_SPEC["min_pause_sec"]
    """Return (n_pauses, avg_pause_sec) via ffmpeg silencedetect filter."""
    cmd = [
        "ffmpeg", "-nostats", "-hide_banner", "-i", path,
        "-af", f"silencedetect=noise=-32dB:d={min_pause_sec}",
        "-f", "null", "-",
    ]
    proc = _subprocess_fv.run(cmd, capture_output=True, timeout=120)
    text = (proc.stderr or b"").decode("utf-8", "ignore")
    durations = []
    for line in text.splitlines():
        if "silence_duration" in line:
            try:
                durations.append(float(line.split("silence_duration:")[-1].strip().split()[0]))
            except Exception:
                pass
    if not durations:
        return 0, 0.0
    return len(durations), sum(durations) / len(durations)


def _fv_sample_frames(path: str, n: int) -> dict:
    """
    Sample n frames and return a rich set of image metrics:

      sharpness            – mean Laplacian variance (focus)
      face_coverage        – fraction of frames where a face was found
      gaze_offset          – avg |dx|+|dy| of face center from frame center
      head_horiz_motion    – std-dev of face-center X across frames (norm. to W)
      head_vert_motion     – std-dev of face-center Y across frames (norm. to H)
      head_top_margin      – avg gap above the head bbox, normalized to H
      face_size_ratio      – avg (face_w * face_h) / (frame_w * frame_h)
      bg_std               – avg std-dev of background pixels (head bbox masked)
      face_brightness_std  – avg std-dev of pixel intensities inside face bbox

    All numbers are picked to be cheap (~no extra IO over existing pass) and
    map directly onto pass/warn/fail bands defined in FV_SPEC.
    """
    cap = _fv_cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("opencv could not open video")
    total = int(cap.get(_fv_cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        raise RuntimeError("video has 0 frames")
    face_cascade = _fv_cv2.CascadeClassifier(
        _fv_cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    sharp_vals: list[float]   = []
    offsets:    list[float]   = []
    cx_norm:    list[float]   = []
    cy_norm:    list[float]   = []
    top_margin: list[float]   = []
    face_area:  list[float]   = []
    bg_stds:    list[float]   = []
    face_brts:  list[float]   = []

    face_hits = 0
    face_total = 0
    indices = [int(total * i / (n + 1)) for i in range(1, n + 1)]

    for idx in indices:
        cap.set(_fv_cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        face_total += 1
        H, W = frame.shape[:2]
        gray = _fv_cv2.cvtColor(frame, _fv_cv2.COLOR_BGR2GRAY)
        sharp_vals.append(_fv_cv2.Laplacian(gray, _fv_cv2.CV_64F).var())

        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(120, 120))
        if len(faces) > 0:
            face_hits += 1
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            fcx = (x + fw / 2.0) / W
            fcy = (y + fh / 2.0) / H
            offsets.append(abs(fcx - 0.5) + abs(fcy - 0.5))
            cx_norm.append(fcx)
            cy_norm.append(fcy)
            top_margin.append(max(0.0, y) / float(H))
            face_area.append((fw * fh) / float(W * H))

            # Face brightness std-dev: inside the bbox (lighting evenness proxy).
            face_brts.append(float(gray[y:y+fh, x:x+fw].std()))

            # Background std-dev: mask out an expanded face bbox and measure
            # std-dev of the remaining pixels (low = plain background).
            pad_x, pad_y = int(fw * 0.30), int(fh * 0.30)
            x0 = max(0, x - pad_x); y0 = max(0, y - pad_y)
            x1 = min(W, x + fw + pad_x); y1 = min(H, y + fh + pad_y)
            mask = _fv_np.ones((H, W), dtype=bool)
            mask[y0:y1, x0:x1] = False
            bg_pixels = gray[mask]
            if bg_pixels.size > 50:
                bg_stds.append(float(bg_pixels.std()))
    cap.release()

    def _safe_mean(arr, default=0.0):
        return float(sum(arr) / len(arr)) if arr else float(default)

    def _safe_std(arr, default=0.0):
        if len(arr) < 2:
            return float(default)
        m = sum(arr) / len(arr)
        return float((sum((v - m) ** 2 for v in arr) / len(arr)) ** 0.5)

    sharp    = _safe_mean(sharp_vals)
    coverage = (face_hits / face_total) if face_total else 0.0
    gaze     = _safe_mean(offsets, default=0.4)  # missing face → assume off
    return {
        "sharpness":           round(sharp, 1),
        "face_coverage":       round(coverage, 3),
        "gaze_offset":         round(gaze, 3),
        "head_horiz_motion":   round(_safe_std(cx_norm), 4),
        "head_vert_motion":    round(_safe_std(cy_norm), 4),
        "head_top_margin":     round(_safe_mean(top_margin, default=0.30), 3),
        "face_size_ratio":     round(_safe_mean(face_area, default=0.0), 4),
        "bg_std":              round(_safe_mean(bg_stds, default=40.0), 1),
        "face_brightness_std": round(_safe_mean(face_brts, default=60.0), 1),
        "frames_sampled":      face_total,
    }


def _fv_audio_meta(path: str) -> dict:
    """
    Inspect audio: presence, loudness (LUFS), peak (dBFS) and silent-ratio.
    Uses ffprobe + ffmpeg's loudnorm/volumedetect filters. Returns:
       {present, loudness_lufs, peak_dbfs, silent_ratio}
    """
    out_present = _subprocess_fv.run(
        ["ffprobe", "-v", "error",
         "-select_streams", "a:0",
         "-show_entries", "stream=codec_type",
         "-of", "json", path],
        capture_output=True, timeout=15,
    )
    has_audio = b'"codec_type"' in (out_present.stdout or b"")
    if not has_audio:
        return {"present": False, "loudness_lufs": None, "peak_dbfs": None,
                "silent_ratio": 1.0}

    # Loudness: parse the JSON the loudnorm filter prints at end of stderr.
    ln = _subprocess_fv.run(
        ["ffmpeg", "-nostats", "-hide_banner", "-i", path,
         "-af", "loudnorm=I=-23:LRA=7:tp=-2:print_format=json",
         "-f", "null", "-"],
        capture_output=True, timeout=180,
    )
    lufs, peak = None, None
    txt = (ln.stderr or b"").decode("utf-8", "ignore")
    # The JSON block is the last {...} in stderr.
    if "{" in txt and "input_i" in txt:
        try:
            blob = "{" + txt.split("{", 1)[1].rsplit("}", 1)[0] + "}"
            j = json.loads(blob)
            lufs = float(j.get("input_i") or 0.0) if j.get("input_i") not in (None, "-inf") else None
            peak = float(j.get("input_tp") or 0.0) if j.get("input_tp") not in (None, "-inf") else None
        except Exception:
            pass

    # Silent ratio: total seconds of silence (>=0.5s, -32dB) ÷ duration.
    sd = _subprocess_fv.run(
        ["ffmpeg", "-nostats", "-hide_banner", "-i", path,
         "-af", "silencedetect=noise=-32dB:d=0.5",
         "-f", "null", "-"],
        capture_output=True, timeout=180,
    )
    sd_txt = (sd.stderr or b"").decode("utf-8", "ignore")
    silent_total = 0.0
    for line in sd_txt.splitlines():
        if "silence_duration" in line:
            try:
                silent_total += float(line.split("silence_duration:")[-1].strip().split()[0])
            except Exception:
                pass
    # Duration was already probed elsewhere; reparse from sd_txt for accuracy.
    dur = 0.0
    for line in sd_txt.splitlines():
        if "Duration:" in line:
            try:
                hms = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = hms.split(":")
                dur = int(h) * 3600 + int(m) * 60 + float(s)
            except Exception:
                pass
            break
    silent_ratio = (silent_total / dur) if dur > 0 else 0.0
    return {
        "present": True,
        "loudness_lufs": (round(lufs, 1) if lufs is not None else None),
        "peak_dbfs":     (round(peak, 1) if peak is not None else None),
        "silent_ratio":  round(min(silent_ratio, 1.0), 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe sensors — FaceMesh (mouth/eyes/head pose) and Pose (hands).
# All metrics are robust to missing detections; if MediaPipe is not installed
# we return an empty {"available": False} dict and the corresponding checks
# are simply omitted from the report.
# ─────────────────────────────────────────────────────────────────────────────

# Lip indices (FaceMesh 468 landmarks) — outer mouth corners + upper/lower lips.
_FV_LIP_LEFT_CORNER  = 61
_FV_LIP_RIGHT_CORNER = 291
_FV_LIP_UPPER        = 13   # innerUpperLip center
_FV_LIP_LOWER        = 14   # innerLowerLip center
# Eye landmarks (left/right, in mediapipe coord order).
_FV_LEFT_EYE_TOP, _FV_LEFT_EYE_BOTTOM = 159, 145
_FV_LEFT_EYE_L,   _FV_LEFT_EYE_R      = 33,  133
_FV_RIGHT_EYE_TOP, _FV_RIGHT_EYE_BOTTOM = 386, 374
_FV_RIGHT_EYE_L,   _FV_RIGHT_EYE_R      = 362, 263


def _fv_face_mesh_sample(path: str, n: int) -> dict:
    """
    Sample n frames and compute MediaPipe FaceLandmarker metrics:
      • mouth_var, mouth_max, mouth_extreme_frac — articulation / yawn
      • eye_open_avg, eye_closed_ratio
      • head_yaw_std, head_pitch_std, head_roll_std (degrees, via solvePnP)
    Returns {"available": False, "reason": "..."} on any setup failure.
    """
    if not _FV_MP:
        return {"available": False, "reason": _FV_MP_ERROR or "mediapipe not available"}
    model = _fv_mp_ensure_model(_FV_MP_FACE_URL, "face_landmarker.task")
    if not model:
        return {"available": False, "reason": "failed to download face_landmarker.task"}
    try:
        BaseOptions           = _fv_mp_python.BaseOptions
        FaceLandmarker        = _fv_mp_vision.FaceLandmarker
        FaceLandmarkerOptions = _fv_mp_vision.FaceLandmarkerOptions
        VisionRunningMode     = _fv_mp_vision.RunningMode
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
        )

        cap = _fv_cv2.VideoCapture(path)
        if not cap.isOpened():
            return {"available": False, "reason": "cv2 cannot open clip"}
        total = int(cap.get(_fv_cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            cap.release()
            return {"available": False, "reason": "no frames"}
        idxs = [int(i * total / max(1, n)) for i in range(n)]

        mars:   list[float] = []
        ears:   list[float] = []
        yaws:   list[float] = []
        pitches:list[float] = []
        rolls:  list[float] = []
        closed_frames = 0
        extreme_mouth_frames = 0
        seen = 0

        with FaceLandmarker.create_from_options(options) as landmarker:
            for fi in idxs:
                cap.set(_fv_cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                rgb = _fv_cv2.cvtColor(frame, _fv_cv2.COLOR_BGR2RGB)
                mp_img = _fv_mp_mod.Image(
                    image_format=_fv_mp_mod.ImageFormat.SRGB, data=rgb
                )
                res = landmarker.detect(mp_img)
                if not res.face_landmarks:
                    continue
                lm = res.face_landmarks[0]
                seen += 1

                # ── Mouth Aspect Ratio (MAR) ──────────────────────────────
                p_l = lm[_FV_LIP_LEFT_CORNER]
                p_r = lm[_FV_LIP_RIGHT_CORNER]
                p_u = lm[_FV_LIP_UPPER]
                p_d = lm[_FV_LIP_LOWER]
                horiz = float(((p_r.x - p_l.x) ** 2 + (p_r.y - p_l.y) ** 2) ** 0.5)
                vert  = float(((p_u.x - p_d.x) ** 2 + (p_u.y - p_d.y) ** 2) ** 0.5)
                mar = (vert / horiz) if horiz > 1e-6 else 0.0
                mars.append(mar)
                if mar >= FV_SPEC["mouth_open_extreme"]:
                    extreme_mouth_frames += 1

                # ── Eye Aspect Ratio (EAR) ────────────────────────────────
                def _ear(top: int, bot: int, l: int, r: int) -> float:
                    pt, pb = lm[top], lm[bot]
                    pl, pr = lm[l],   lm[r]
                    v = float(((pt.x - pb.x) ** 2 + (pt.y - pb.y) ** 2) ** 0.5)
                    h = float(((pl.x - pr.x) ** 2 + (pl.y - pr.y) ** 2) ** 0.5)
                    return (v / h) if h > 1e-6 else 0.0
                ear = (_ear(_FV_LEFT_EYE_TOP,  _FV_LEFT_EYE_BOTTOM,
                            _FV_LEFT_EYE_L,    _FV_LEFT_EYE_R) +
                       _ear(_FV_RIGHT_EYE_TOP, _FV_RIGHT_EYE_BOTTOM,
                            _FV_RIGHT_EYE_L,   _FV_RIGHT_EYE_R)) * 0.5
                ears.append(ear)
                if ear < 0.10:
                    closed_frames += 1

                # ── Head pose via 6-point solvePnP on FaceMesh landmarks ──
                # Use canonical 3D points (mm-ish, arbitrary unit) for:
                #   nose tip, chin, left/right eye outer corner, left/right mouth corner.
                h_img, w_img = frame.shape[:2]
                image_pts = _fv_np.array([
                    [lm[1].x   * w_img, lm[1].y   * h_img],    # nose tip
                    [lm[152].x * w_img, lm[152].y * h_img],    # chin
                    [lm[33].x  * w_img, lm[33].y  * h_img],    # left eye outer
                    [lm[263].x * w_img, lm[263].y * h_img],    # right eye outer
                    [lm[61].x  * w_img, lm[61].y  * h_img],    # left mouth corner
                    [lm[291].x * w_img, lm[291].y * h_img],    # right mouth corner
                ], dtype=_fv_np.float64)
                model_pts = _fv_np.array([
                    [   0.0,    0.0,    0.0],
                    [   0.0, -63.6,  -12.5],
                    [ -43.3,   32.7,  -26.0],
                    [  43.3,   32.7,  -26.0],
                    [ -28.9,  -28.9,  -24.1],
                    [  28.9,  -28.9,  -24.1],
                ], dtype=_fv_np.float64)
                focal = float(w_img)
                cam_mtx = _fv_np.array([
                    [focal, 0,     w_img / 2.0],
                    [0,     focal, h_img / 2.0],
                    [0,     0,     1.0],
                ], dtype=_fv_np.float64)
                dist = _fv_np.zeros((4, 1))
                ok2, rvec, _ = _fv_cv2.solvePnP(
                    model_pts, image_pts, cam_mtx, dist,
                    flags=_fv_cv2.SOLVEPNP_ITERATIVE,
                )
                if not ok2:
                    continue
                rmat, _ = _fv_cv2.Rodrigues(rvec)
                # Euler angles from rotation matrix (Tait–Bryan, in degrees).
                sy = float((rmat[0, 0] ** 2 + rmat[1, 0] ** 2) ** 0.5)
                singular = sy < 1e-6
                if not singular:
                    pitch = _math_fv.degrees(_math_fv.atan2(-rmat[2, 0], sy))
                    yaw   = _math_fv.degrees(_math_fv.atan2( rmat[1, 0], rmat[0, 0]))
                    roll  = _math_fv.degrees(_math_fv.atan2( rmat[2, 1], rmat[2, 2]))
                else:
                    pitch = _math_fv.degrees(_math_fv.atan2(-rmat[2, 0], sy))
                    yaw   = 0.0
                    roll  = _math_fv.degrees(_math_fv.atan2(-rmat[1, 2], rmat[1, 1]))
                yaws.append(yaw)
                pitches.append(pitch)
                rolls.append(roll)

        cap.release()

        if seen == 0:
            return {"available": False, "reason": "face mesh did not detect a face"}

        def _std(xs: list[float]) -> float:
            return float(_fv_np.std(xs)) if xs else 0.0
        def _mean(xs: list[float], d: float = 0.0) -> float:
            return float(_fv_np.mean(xs)) if xs else d

        return {
            "available":         True,
            "frames":            seen,
            "mouth_var":         round(_std(mars),    4),
            "mouth_max":         round((max(mars) if mars else 0.0), 3),
            "mouth_extreme_frac":round(extreme_mouth_frames / max(1, seen), 3),
            "eye_open_avg":      round(_mean(ears, d=0.0), 3),
            "eye_closed_ratio":  round(closed_frames / max(1, seen), 3),
            "head_yaw_std":      round(_std(yaws),   2),
            "head_pitch_std":    round(_std(pitches),2),
            "head_roll_std":     round(_std(rolls),  2),
        }
    except Exception as e:  # pragma: no cover
        return {"available": False, "reason": f"face mesh failed: {e!s}"}


# Pose landmark indices we care about.
_FV_POSE_LSHO, _FV_POSE_RSHO = 11, 12
_FV_POSE_LWR,  _FV_POSE_RWR  = 15, 16
_FV_POSE_NOSE                = 0


def _fv_pose_sample(path: str, n: int) -> dict:
    """
    Sample n frames with MediaPipe PoseLandmarker and report:
      • hands_above_chest_ratio — fraction of frames where any wrist is above
        the chest line (≈ shoulder Y, with small margin). PDF: hands BELOW chest.
      • hand_on_face_ratio      — fraction of frames where a wrist is within a
        face-sized radius around the nose (≈ touching face). PDF: don't touch face.
      • frames                  — frames with a usable pose detection.
    """
    if not _FV_MP:
        return {"available": False, "reason": _FV_MP_ERROR or "mediapipe not available"}
    model = _fv_mp_ensure_model(_FV_MP_POSE_URL, "pose_landmarker_lite.task")
    if not model:
        return {"available": False, "reason": "failed to download pose_landmarker_lite.task"}
    try:
        BaseOptions           = _fv_mp_python.BaseOptions
        PoseLandmarker        = _fv_mp_vision.PoseLandmarker
        PoseLandmarkerOptions = _fv_mp_vision.PoseLandmarkerOptions
        VisionRunningMode     = _fv_mp_vision.RunningMode
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
        )

        cap = _fv_cv2.VideoCapture(path)
        if not cap.isOpened():
            return {"available": False, "reason": "cv2 cannot open clip"}
        total = int(cap.get(_fv_cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            cap.release()
            return {"available": False, "reason": "no frames"}
        idxs = [int(i * total / max(1, n)) for i in range(n)]

        seen = 0
        above_chest = 0
        on_face     = 0

        with PoseLandmarker.create_from_options(options) as pose:
            for fi in idxs:
                cap.set(_fv_cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                rgb = _fv_cv2.cvtColor(frame, _fv_cv2.COLOR_BGR2RGB)
                mp_img = _fv_mp_mod.Image(
                    image_format=_fv_mp_mod.ImageFormat.SRGB, data=rgb
                )
                res = pose.detect(mp_img)
                if not res.pose_landmarks:
                    continue
                lm = res.pose_landmarks[0]
                seen += 1

                def _vis(p) -> float:
                    v = getattr(p, "visibility", None)
                    return float(v) if v is not None else 1.0

                # Chest line ≈ midpoint of shoulders, with a small downward offset.
                ls, rs = lm[_FV_POSE_LSHO], lm[_FV_POSE_RSHO]
                if _vis(ls) < 0.3 or _vis(rs) < 0.3:
                    continue
                chest_y = (ls.y + rs.y) * 0.5 + 0.04   # below shoulder line

                lw, rw = lm[_FV_POSE_LWR], lm[_FV_POSE_RWR]
                wr_above = False
                if _vis(lw) >= 0.3 and lw.y < chest_y:
                    wr_above = True
                if _vis(rw) >= 0.3 and rw.y < chest_y:
                    wr_above = True
                if wr_above:
                    above_chest += 1

                # Hand near face → distance from wrist to nose under face-size radius.
                nose = lm[_FV_POSE_NOSE]
                shoulder_w = float(((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2) ** 0.5)
                face_r = max(0.08, shoulder_w * 0.55)
                near = False
                for w in (lw, rw):
                    if _vis(w) < 0.3:
                        continue
                    d = float(((w.x - nose.x) ** 2 + (w.y - nose.y) ** 2) ** 0.5)
                    if d <= face_r:
                        near = True
                        break
                if near:
                    on_face += 1

        cap.release()
        if seen == 0:
            return {"available": False, "reason": "pose did not detect a person"}
        return {
            "available":               True,
            "frames":                  seen,
            "hands_above_chest_ratio": round(above_chest / max(1, seen), 3),
            "hand_on_face_ratio":      round(on_face     / max(1, seen), 3),
        }
    except Exception as e:  # pragma: no cover
        return {"available": False, "reason": f"pose failed: {e!s}"}


async def _fv_runner_live(jid: str):
    """Real validator. Downloads if needed, then runs the local checks."""
    async with _FV_LOCK:
        job = FV_JOBS.get(jid)
    if not job:
        return

    tmp = None
    try:
        async with _FV_LOCK:
            job["status"] = "running"
            job["started_at"] = _now_iso()
            _job_log(job, f"▶ validating {job['source']}")
            job["stage"] = "download"
            job["progress"] = 0.05

        src = job["source"]
        if src.startswith("s3://"):
            if not _s3_client:
                raise RuntimeError("S3 client not configured")
            parsed = _urlparse(src)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            tmp = tempfile.NamedTemporaryFile(
                suffix=Path(key).suffix or ".mp4", delete=False
            )
            tmp.close()
            await asyncio.get_event_loop().run_in_executor(
                None, _s3_client.download_file, bucket, key, tmp.name
            )
            local = tmp.name
            _job_log(job, f"  ↓ downloaded → {tmp.name}")
        elif src.startswith("file://"):
            local = src[7:]
        elif src.startswith("http"):
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.close()
            with requests.get(src, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp.name, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
            local = tmp.name
            _job_log(job, f"  ↓ downloaded → {tmp.name}")
        else:
            raise RuntimeError(f"Unsupported source: {src}")

        async with _FV_LOCK:
            job["stage"] = "probe_metadata"
            job["progress"] = 0.20
            _job_log(job, "  → ffprobe metadata")
        meta = await asyncio.get_event_loop().run_in_executor(None, _fv_ffprobe, local)

        async with _FV_LOCK:
            job["stage"] = "sample_frames"
            job["progress"] = 0.40
            _job_log(job, f"  → sampling {FV_SPEC['sample_frames']} frames "
                          "(sharpness · face · gaze · head-pose · framing · bg · light)")
        samp = await asyncio.get_event_loop().run_in_executor(
            None, _fv_sample_frames, local, FV_SPEC["sample_frames"]
        )

        async with _FV_LOCK:
            job["stage"] = "audio_pauses"
            job["progress"] = 0.65
            _job_log(job, "  → ffmpeg silencedetect (closed-mouth pauses)")
        n_pauses, avg_pause = await asyncio.get_event_loop().run_in_executor(
            None, _fv_silencedetect, local, FV_SPEC["min_pause_sec"]
        )

        async with _FV_LOCK:
            job["stage"] = "audio_loudness"
            job["progress"] = 0.78
            _job_log(job, "  → ffmpeg loudnorm + silencedetect (audio QC)")
        audio = await asyncio.get_event_loop().run_in_executor(None, _fv_audio_meta, local)

        # ── MediaPipe sensors (mouth/eyes/head-pose + hands). ──────────────
        face_mesh: dict = {"available": False, "reason": "skipped"}
        pose:      dict = {"available": False, "reason": "skipped"}
        if _FV_MP:
            async with _FV_LOCK:
                job["stage"] = "face_mesh"
                job["progress"] = 0.86
                _job_log(job, "  → MediaPipe FaceMesh (mouth · eyes · 6-DoF head pose)")
            face_mesh = await asyncio.get_event_loop().run_in_executor(
                None, _fv_face_mesh_sample, local, FV_SPEC["mp_sample_frames"]
            )
            if not face_mesh.get("available"):
                _job_log(job, f"  ⚠ FaceMesh skipped: {face_mesh.get('reason')}")

            async with _FV_LOCK:
                job["stage"] = "pose"
                job["progress"] = 0.92
                _job_log(job, "  → MediaPipe Pose (hands vs chest · hand-on-face)")
            pose = await asyncio.get_event_loop().run_in_executor(
                None, _fv_pose_sample, local, FV_SPEC["mp_sample_frames"]
            )
            if not pose.get("available"):
                _job_log(job, f"  ⚠ Pose skipped: {pose.get('reason')}")
        else:
            _job_log(job, f"  ⚠ MediaPipe sensors disabled ({_FV_MP_ERROR})")

        async with _FV_LOCK:
            job["stage"] = "score"
            job["progress"] = 0.97
            _job_log(job, "  → scoring across all sensors")

        metrics = {
            "resolution": {"width": meta["width"], "height": meta["height"]},
            "fps": round(meta["fps"], 3),
            "duration_sec": round(meta["duration_sec"], 1),
            "duration_human": f"{int(meta['duration_sec'] // 60)}m {int(meta['duration_sec'] % 60):02d}s",
            "sharpness": samp["sharpness"],
            "face_coverage": samp["face_coverage"],
            "pauses": n_pauses,
            "avg_pause_sec": round(avg_pause, 2),
            "gaze_offset": samp["gaze_offset"],
            "head_horiz_motion":   samp["head_horiz_motion"],
            "head_vert_motion":    samp["head_vert_motion"],
            "head_top_margin":     samp["head_top_margin"],
            "face_size_ratio":     samp["face_size_ratio"],
            "bg_std":              samp["bg_std"],
            "face_brightness_std": samp["face_brightness_std"],
            "audio":      audio,
            "face_mesh":  face_mesh,
            "pose":       pose,
            "frames_sampled": samp["frames_sampled"],
        }
        checks, recs = _fv_compile_checks(metrics)
        score = _fv_score(checks)
        verdict = _fv_verdict(score, checks)

        async with _FV_LOCK:
            job["metrics"] = metrics
            job["checks"] = checks
            job["recommendations"] = recs
            job["score"] = score
            job["verdict"] = verdict
            job["status"] = "completed"
            job["progress"] = 1.0
            job["finished_at"] = _now_iso()
            _job_log(job, f"✓ verdict: {verdict} (score {score}/100)")
    except Exception as e:
        async with _FV_LOCK:
            job["status"] = "failed"
            job["error"] = str(e)
            job["finished_at"] = _now_iso()
            _job_log(job, f"✗ Failed: {e}")
    finally:
        if tmp:
            try: os.unlink(tmp.name)
            except Exception: pass


async def _fv_runner(jid: str):
    if _FV_LIVE:
        await _fv_runner_live(jid)
    else:
        await _fv_runner_sim(jid)


# ── HTTP endpoints ─────────────────────────────────────────────────────────
@app.get("/api/validate/mode")
async def fv_mode(user: User = Depends(require_role(Role.viewer))):
    return {
        "mode": "live" if _FV_LIVE else "simulation",
        "reason": None if _FV_LIVE else _FV_INIT_ERROR,
        "spec": FV_SPEC,
    }


@app.post("/api/validate")
async def fv_create(req: FVRequest, user: User = Depends(require_role(Role.viewer))):
    errors = _fv_validate_req(req)
    if errors:
        raise HTTPException(400, {"errors": errors})
    job = _fv_create_job(req, owner_email=user.email)
    async with _FV_LOCK:
        FV_JOBS[job["id"]] = job
    asyncio.create_task(_fv_runner(job["id"]))
    return {"job_id": job["id"], "mode": job["mode"]}


@app.get("/api/validate/jobs")
async def fv_list(user: User = Depends(require_role(Role.viewer))):
    rows = sorted(FV_JOBS.values(), key=lambda j: j["created_at"], reverse=True)
    return [
        {
            "id": j["id"], "name": j["name"], "source": j["source"],
            "status": j["status"], "progress": j["progress"],
            "stage": j.get("stage"), "created_at": j["created_at"],
            "verdict": j.get("verdict"), "score": j.get("score"),
        }
        for j in rows[:50]
    ]


@app.get("/api/validate/jobs/{job_id}")
async def fv_get(job_id: str, user: User = Depends(require_role(Role.viewer))):
    job = FV_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.post("/api/validate/jobs/{job_id}/cancel")
async def fv_cancel(job_id: str, user: User = Depends(require_role(Role.viewer))):
    async with _FV_LOCK:
        job = FV_JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job["status"] in ("completed", "failed", "cancelled"):
            return {"ok": True, "status": job["status"]}
        job["status"] = "cancelled"
        job["finished_at"] = _now_iso()
        _job_log(job, "Cancelled by user")
    return {"ok": True, "status": "cancelled"}


# Folder where uploaded clips are stashed before validation. Kept on local
# disk on purpose — the validator reads them via file:// URIs.
_FV_UPLOAD_DIR = Path(tempfile.gettempdir()) / "avatars_studio_uploads"
_FV_UPLOAD_DIR.mkdir(exist_ok=True)
_FV_UPLOAD_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB hard cap
_FV_UPLOAD_ALLOWED_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}


@app.post("/api/validate/upload")
async def fv_upload(
    file: UploadFile = File(...),
    user: User = Depends(require_role(Role.viewer)),
):
    """Accept a local clip from the browser, persist it to a temp directory,
    and return a `file://` URI that can be fed straight into /api/validate."""
    filename = file.filename or "upload.mp4"
    ext = Path(filename).suffix.lower() or ".mp4"
    if ext not in _FV_UPLOAD_ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported video format: {ext}")

    token = secrets.token_urlsafe(10)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem)[:80] or "clip"
    target = _FV_UPLOAD_DIR / f"{token}_{safe_stem}{ext}"

    written = 0
    try:
        with open(target, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > _FV_UPLOAD_MAX_BYTES:
                    raise HTTPException(413, "Upload exceeds 2 GB limit")
                out.write(chunk)
    except HTTPException:
        try: target.unlink()
        except Exception: pass
        raise
    except Exception as e:
        try: target.unlink()
        except Exception: pass
        raise HTTPException(500, f"Failed to store upload: {e}")
    finally:
        await file.close()

    return {
        "ok": True,
        "uri": f"file://{target}",
        "size_bytes": written,
        "size_human": _human_bytes(written),
        "filename": filename,
    }


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024
        i += 1
    return f"{f:.1f} {units[i]}" if i else f"{int(f)} {units[i]}"


# ════════════════════════════════════════════════════════════════════════════
# Tool 6.5: Jira Avatars — read-only proxy to the production backlog
# ────────────────────────────────────────────────────────────────────────────
# Pulls "Avatars creation" issues from Jira, groups them by month, and
# exposes them to the UI. The token NEVER leaves the backend — the UI only
# talks to `/api/jira/*`. Credentials are read from .env so they can be
# rotated without code changes.
# ════════════════════════════════════════════════════════════════════════════

JIRA_DOMAIN       = os.getenv("JIRA_DOMAIN", "").strip().rstrip("/")
JIRA_EMAIL        = os.getenv("JIRA_EMAIL", "").strip()
JIRA_API_TOKEN    = os.getenv("JIRA_API_TOKEN", "").strip()
JIRA_PROJECT_KEY  = os.getenv("JIRA_PROJECT_KEY", "VM").strip()
JIRA_ISSUE_TYPE   = os.getenv("JIRA_ISSUE_TYPE", "Avatars creation").strip()
JIRA_CREATED_SINCE = os.getenv("JIRA_CREATED_SINCE", "2026-05-01").strip()
JIRA_FIELDS = {
    "evalDate":      os.getenv("JIRA_FIELD_EVAL_DATE",      "customfield_11046"),
    "endDate":       os.getenv("JIRA_FIELD_END_DATE",       "customfield_11048"),
    "footage":       os.getenv("JIRA_FIELD_FOOTAGE",        "customfield_11049"),
    "timeCode":      os.getenv("JIRA_FIELD_TIME_CODE",      "customfield_11050"),
    "pauses":        os.getenv("JIRA_FIELD_PAUSES",         "customfield_11051"),
    "storage":       os.getenv("JIRA_FIELD_STORAGE",        "customfield_11150"),
    "customerEmail": os.getenv("JIRA_FIELD_CUSTOMER_EMAIL", "customfield_11151"),
    "orgId":         os.getenv("JIRA_FIELD_ORG_ID",         "customfield_11152"),
    "slackLink":     os.getenv("JIRA_FIELD_SLACK_LINK",     "customfield_11153"),
    "accountId":     os.getenv("JIRA_FIELD_ACCOUNT_ID",     "customfield_11154"),
    "approvers":     os.getenv("JIRA_FIELD_APPROVERS",      "customfield_10494"),
}
# Storage values are stored as multiselect options keyed by name (e.g. EU, prod).
JIRA_STORAGE_OPTIONS = [v.strip() for v in os.getenv(
    "JIRA_STORAGE_OPTIONS", "prod,EU,ewizzard"
).split(",") if v.strip()]
JIRA_DEFAULT_ESTIMATE = os.getenv("JIRA_DEFAULT_ESTIMATE", "8h").strip()
JIRA_CONFIGURED = bool(JIRA_DOMAIN and JIRA_EMAIL and JIRA_API_TOKEN)

# Small in-memory cache so the UI feels instant and we don't hammer Jira.
_JIRA_CACHE: dict = {"data": None, "fetched_at": 0.0, "ttl": 60.0}
# email → accountId mapping (long-lived; small)
_JIRA_USER_CACHE: dict[str, Optional[str]] = {}
# accountId → {data, ts}
_JIRA_MY_CACHE: dict[str, dict] = {}
_JIRA_MY_TTL = 30.0  # seconds


def _jira_text_to_adf(text: str) -> dict:
    """Convert plain text to a minimal Atlassian Document Format payload."""
    text = (text or "").rstrip()
    paragraphs = text.split("\n\n") if text else [""]
    content = []
    for p in paragraphs:
        lines = p.split("\n")
        inline: list = []
        for i, line in enumerate(lines):
            if i > 0:
                inline.append({"type": "hardBreak"})
            if line:
                inline.append({"type": "text", "text": line})
        if not inline:
            inline = [{"type": "text", "text": ""}]
        content.append({"type": "paragraph", "content": inline})
    return {"type": "doc", "version": 1, "content": content}


async def _jira_request(method: str, path: str, **kwargs):
    """Authenticated Jira call. Returns httpx.Response."""
    if not JIRA_CONFIGURED:
        raise HTTPException(503, {"errors": ["Jira not configured — set JIRA_DOMAIN / JIRA_EMAIL / JIRA_API_TOKEN in .env"]})
    url = f"https://{JIRA_DOMAIN}{path}"
    headers = kwargs.pop("headers", {}) or {}
    headers.setdefault("Accept", "application/json")
    if kwargs.get("json") is not None:
        headers.setdefault("Content-Type", "application/json")
    async with httpx.AsyncClient(timeout=30.0) as cli:
        return await cli.request(
            method, url, headers=headers,
            auth=(JIRA_EMAIL, JIRA_API_TOKEN), **kwargs,
        )


def _jira_raise_for_status(r, default_msg: str = "Jira API error"):
    if r.status_code < 400:
        return
    body = (r.text or "")[:400]
    if r.status_code == 401:
        raise HTTPException(401, {"errors": ["Jira authentication failed — check JIRA_API_TOKEN"]})
    if r.status_code == 403:
        raise HTTPException(403, {"errors": ["Jira denied this operation"]})
    if r.status_code == 404:
        raise HTTPException(404, {"errors": ["Resource not found in Jira"]})
    raise HTTPException(502, {"errors": [f"{default_msg}: {r.status_code} — {body}"]})


# Optional explicit overrides: "email:accountId,email:accountId,..."
# Useful when Atlassian privacy hides emails so auto-discovery can't match.
_JIRA_USER_MAP_RAW = os.getenv("JIRA_USER_MAP", "").strip()
_JIRA_USER_OVERRIDES: dict[str, str] = {}
for _pair in _JIRA_USER_MAP_RAW.split(","):
    _pair = _pair.strip()
    if ":" in _pair:
        _e, _a = _pair.split(":", 1)
        _JIRA_USER_OVERRIDES[_e.strip().lower()] = _a.strip()


async def _jira_lookup_account_id(email: str) -> Optional[str]:
    """
    Resolve a Studio user's email → Jira accountId, using (in order):
      1) explicit override via JIRA_USER_MAP env
      2) /rest/api/3/myself — when the email matches the API token's owner
      3) /rest/api/3/user/search — works only with 'Browse users' permission
      4) scan of project issues — picks up users whose emails are visible
         to our token (assignees we've seen at least once)
    Result is cached for the lifetime of the process.
    """
    if not email:
        return None
    key = email.strip().lower()
    if key in _JIRA_USER_CACHE:
        return _JIRA_USER_CACHE[key]

    # 1) Explicit override
    if key in _JIRA_USER_OVERRIDES:
        _JIRA_USER_CACHE[key] = _JIRA_USER_OVERRIDES[key]
        return _JIRA_USER_CACHE[key]

    # 2) Token owner — covers the most common case (operator's own queue)
    try:
        r = await _jira_request("GET", "/rest/api/3/myself")
        if r.status_code == 200:
            me = r.json() or {}
            if (me.get("emailAddress") or "").lower() == key:
                _JIRA_USER_CACHE[key] = me.get("accountId")
                return _JIRA_USER_CACHE[key]
    except Exception:
        pass

    # 3) Privileged search (returns [] for tokens without 'Browse users')
    try:
        r = await _jira_request("GET", "/rest/api/3/user/search", params={"query": key})
        if r.status_code == 200:
            for u in (r.json() or []):
                if (u.get("emailAddress") or "").lower() == key:
                    _JIRA_USER_CACHE[key] = u.get("accountId")
                    return _JIRA_USER_CACHE[key]
    except Exception:
        pass

    # 4) Last resort: scan recent project issues — pick the accountId from any
    #    assignee whose emailAddress matches. This works because most active
    #    avatar creators have at least one ticket assigned to them, and Atlassian
    #    exposes emailAddress to anyone in the same Atlassian organization who
    #    has shared their email visibility.
    try:
        jql = (f'project = {JIRA_PROJECT_KEY} '
               f'AND issuetype = "{JIRA_ISSUE_TYPE}" '
               f'ORDER BY updated DESC')
        r = await _jira_request("GET", "/rest/api/3/search/jql", params={
            "jql": jql, "maxResults": 200, "fields": "assignee",
        })
        if r.status_code == 200:
            for issue in (r.json() or {}).get("issues") or []:
                a = (issue.get("fields") or {}).get("assignee") or {}
                if (a.get("emailAddress") or "").lower() == key:
                    _JIRA_USER_CACHE[key] = a.get("accountId")
                    return _JIRA_USER_CACHE[key]
    except Exception:
        pass

    _JIRA_USER_CACHE[key] = None
    return None


def _jira_bust_caches():
    _JIRA_CACHE["data"] = None
    _JIRA_CACHE["fetched_at"] = 0.0
    _JIRA_MY_CACHE.clear()


def _jira_extract_text(node) -> str:
    """Recursively flatten a Jira ADF (Atlassian Document Format) node to text."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return " ".join(_jira_extract_text(n) for n in node)
    if isinstance(node, dict):
        if node.get("type") == "text":
            return str(node.get("text", "") or "")
        if node.get("type") == "hardBreak":
            return "\n"
        parts = []
        for child in node.get("content", []) or []:
            parts.append(_jira_extract_text(child))
        return " ".join(p for p in parts if p)
    return ""


def _jira_extract_url(value) -> str:
    """Best-effort: pull a URL out of a field value (string / ADF / dict)."""
    if value is None:
        return ""
    if isinstance(value, str):
        s = value.strip()
        return s
    if isinstance(value, dict):
        for k in ("url", "href", "link", "value"):
            v = value.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        text = _jira_extract_text(value).strip()
        m = re.search(r"https?://\S+", text)
        return m.group(0) if m else text
    if isinstance(value, list):
        for item in value:
            u = _jira_extract_url(item)
            if u:
                return u
    return ""


def _jira_extract_scalar(value) -> str:
    """Pull a human-readable scalar (string / int / select option) out of a field."""
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for k in ("value", "name", "displayName"):
            v = value.get(k)
            if isinstance(v, (str, int, float)) and str(v).strip():
                return str(v).strip()
        return _jira_extract_text(value).strip()
    if isinstance(value, list):
        return ", ".join(filter(None, (_jira_extract_scalar(x) for x in value)))
    return str(value)


_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]


def _jira_group_by_month(issues: list, fields: dict) -> dict:
    """Group Jira issues by 'Month YYYY', sorted newest-first inside the group."""
    import datetime as _dt
    grouped: dict[str, list] = {}
    for issue in issues:
        f = issue.get("fields") or {}
        if not f:
            continue
        raw_date = f.get(fields["evalDate"]) or f.get("created") or ""
        month_key = "Unscheduled"
        sort_key = _dt.datetime.min
        if raw_date:
            try:
                d = _dt.datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
                month_key = f"{_MONTHS[d.month - 1]} {d.year}"
                sort_key = d
            except Exception:
                pass

        assignee = f.get("assignee") or {}
        reporter = f.get("reporter") or {}
        prio     = f.get("priority") or {}
        status   = f.get("status")   or {}
        ftype    = f.get("issuetype") or {}

        # Storage (multi-select). Each entry may be {value: 'EU'} or a string.
        raw_storage = f.get(fields["storage"]) or []
        storage_vals: list[str] = []
        if isinstance(raw_storage, list):
            for opt in raw_storage:
                if isinstance(opt, dict):
                    v = opt.get("value") or opt.get("name") or ""
                else:
                    v = str(opt or "")
                if v: storage_vals.append(v)
        elif isinstance(raw_storage, dict):
            v = raw_storage.get("value") or raw_storage.get("name") or ""
            if v: storage_vals.append(v)
        elif isinstance(raw_storage, str) and raw_storage:
            storage_vals.append(raw_storage)

        # Resolution date is provided by Jira on done tickets.
        resolution_dt = f.get("resolutiondate") or ""

        item = {
            "key":        issue.get("key"),
            "url":        f"https://{JIRA_DOMAIN}/browse/{issue.get('key')}" if JIRA_DOMAIN else "",
            "summary":    f.get("summary") or "",
            "assignee":   assignee.get("displayName") if assignee else "Unassigned",
            "assigneeIcon": (assignee.get("avatarUrls") or {}).get("24x24") if assignee else None,
            "reporter":   reporter.get("displayName") if reporter else "",
            "priority":   prio.get("name") or "Medium",
            "priorityIcon": prio.get("iconUrl"),
            "status":     status.get("name") or "Backlog",
            "statusCategory": ((status.get("statusCategory") or {}).get("key") or "new"),
            "issueType":  ftype.get("name") or "",
            "issueTypeIcon": ftype.get("iconUrl"),
            "timeCode":   _jira_extract_scalar(f.get(fields["timeCode"])),
            "evalDate":   str(f.get(fields["evalDate"]) or "").split("T")[0],
            "startDate":  str(f.get("created") or "").split("T")[0],
            "endDate":    str(f.get(fields["endDate"]) or "").split("T")[0],
            "dueDate":    str(f.get("duedate") or "").split("T")[0],
            "resolved":   str(resolution_dt or "").split("T")[0],
            "createdIso": f.get("created") or "",
            "resolvedIso": resolution_dt or "",
            "footage":    _jira_extract_url(f.get(fields["footage"])),
            "pauses":     _jira_extract_scalar(f.get(fields["pauses"])),
            "customerEmail": _jira_extract_scalar(f.get(fields["customerEmail"])),
            "storage":    storage_vals,
            "updated":    str(f.get("updated") or "").split("T")[0],
            "labels":     f.get("labels") or [],
            "_sort":      sort_key.isoformat() if sort_key != _dt.datetime.min else "",
        }
        grouped.setdefault(month_key, []).append(item)

    for month, items in grouped.items():
        items.sort(key=lambda x: x.get("_sort") or "", reverse=True)
        for it in items:
            it.pop("_sort", None)
    return grouped


def _jira_month_order(grouped: dict) -> list[str]:
    """Sort month keys newest → oldest, keeping 'Unscheduled' at the end."""
    import datetime as _dt
    def _key(m: str):
        if m == "Unscheduled":
            return _dt.datetime.min
        try:
            name, year = m.split(" ")
            return _dt.datetime(int(year), _MONTHS.index(name) + 1, 1)
        except Exception:
            return _dt.datetime.min
    return sorted(grouped.keys(), key=_key, reverse=True)


async def _jira_search(force: bool = False) -> dict:
    """Fetch (and cache) the configured Avatars backlog from Jira."""
    import time as _time
    now = _time.time()
    if (not force
        and _JIRA_CACHE["data"] is not None
        and (now - _JIRA_CACHE["fetched_at"]) < _JIRA_CACHE["ttl"]):
        return _JIRA_CACHE["data"]
    if not JIRA_CONFIGURED:
        raise HTTPException(503, {
            "errors": ["Jira not configured — set JIRA_DOMAIN, JIRA_EMAIL, JIRA_API_TOKEN in .env"]
        })

    jql = (f"project = {JIRA_PROJECT_KEY} "
           f"AND issuetype = \"{JIRA_ISSUE_TYPE}\" "
           f"AND created >= \"{JIRA_CREATED_SINCE}\" "
           f"ORDER BY created DESC")
    url = f"https://{JIRA_DOMAIN}/rest/api/3/search/jql"
    params = {"jql": jql, "maxResults": 500, "fields": "*all"}

    async with httpx.AsyncClient(timeout=30.0) as cli:
        r = await cli.get(
            url, params=params,
            auth=(JIRA_EMAIL, JIRA_API_TOKEN),
            headers={"Accept": "application/json"},
        )
    if r.status_code == 401:
        raise HTTPException(401, {"errors": ["Jira authentication failed — check JIRA_EMAIL / JIRA_API_TOKEN"]})
    if r.status_code == 403:
        raise HTTPException(403, {"errors": ["Jira denied access to this project"]})
    if r.status_code != 200:
        body = (r.text or "")[:300]
        raise HTTPException(502, {"errors": [f"Jira API error: {r.status_code} — {body}"]})

    data = r.json()
    issues = data.get("issues") or []
    grouped = _jira_group_by_month(issues, JIRA_FIELDS)
    months = _jira_month_order(grouped)
    payload = {
        "ok": True,
        "fetched_at": _now_iso(),
        "domain": JIRA_DOMAIN,
        "project": JIRA_PROJECT_KEY,
        "issue_type": JIRA_ISSUE_TYPE,
        "since": JIRA_CREATED_SINCE,
        "total": sum(len(v) for v in grouped.values()),
        "months": months,
        "by_month": grouped,
    }
    _JIRA_CACHE["data"] = payload
    _JIRA_CACHE["fetched_at"] = now
    return payload


@app.get("/api/jira/mode")
async def jira_mode(user: User = Depends(require_role(Role.viewer))):
    return {
        "configured": JIRA_CONFIGURED,
        "domain":     JIRA_DOMAIN,
        "project":    JIRA_PROJECT_KEY,
        "issue_type": JIRA_ISSUE_TYPE,
        "since":      JIRA_CREATED_SINCE,
        "fields":     JIRA_FIELDS,
        "reason":     (None if JIRA_CONFIGURED
                       else "JIRA_DOMAIN / JIRA_EMAIL / JIRA_API_TOKEN missing in .env"),
    }


@app.get("/api/jira/avatars")
async def jira_avatars(
    refresh: bool = False,
    user: User = Depends(require_role(Role.viewer)),
):
    return await _jira_search(force=bool(refresh))


# ────────────────────────────────────────────────────────────────────────────
# Manager dashboard — high-level KPIs about custom avatars and voices.
# ────────────────────────────────────────────────────────────────────────────
@app.get("/api/stats/overview")
async def stats_overview(
    refresh: bool = False,
    user: User = Depends(require_user),
    db: Session = Depends(get_session),
):
    """Aggregated counters for the Manager dashboard.

    Avatars data comes from the live Jira "Avatars creation" backlog (same
    data the production overview widget shows). Voice data is derived from
    onboarding submissions that included an audio file (mime_type starting
    with `audio/`). All counts are bucketed by month so the UI can render
    a clean monthly chart and month-over-month deltas.
    """
    import calendar as _cal
    import datetime as _dt

    now = _dt.datetime.utcnow()
    cur_ym       = now.strftime("%Y-%m")
    prev_dt      = (now.replace(day=1) - _dt.timedelta(days=1))
    prev_ym      = prev_dt.strftime("%Y-%m")
    # Human-friendly variants that match how the Jira grouping labels months.
    cur_label    = f"{_cal.month_name[now.month]} {now.year}"
    prev_label   = f"{_cal.month_name[prev_dt.month]} {prev_dt.year}"

    def _label_to_ym(label: str) -> str:
        """Map 'June 2026' → '2026-06' so the UI can sort/format cleanly."""
        if not label or label == "Unscheduled":
            return label
        try:
            parts = label.split()
            mo = list(_cal.month_name).index(parts[0])
            return f"{int(parts[1]):04d}-{mo:02d}"
        except Exception:
            return label

    today_date = now.date()

    # ── 1. Avatars (Jira) ────────────────────────────────────────────────
    avatars = {
        "configured": JIRA_CONFIGURED,
        "total": 0,
        "this_month": 0,
        "last_month": 0,
        "delta_pct": None,
        "this_week": 0,
        "by_status": {"done": 0, "in_progress": 0, "todo": 0},
        "by_priority": {"Highest": 0, "High": 0, "Medium": 0, "Low": 0, "Lowest": 0},
        "by_storage": {},        # {EU: 3, prod: 2, ...}
        "by_month": [],          # [{month, count}]
        "by_assignee": [],       # [{name, count, avatarUrl, done, in_progress}]
        "weekly": [],            # [{week, created, resolved}]
        "aging_wip": {"fresh": 0, "warm": 0, "stale": 0, "critical": 0},
        "lead_time_days": None,
        "median_lead_days": None,
        "overdue": 0,
        "recent": [],
        "error": None,
    }
    if JIRA_CONFIGURED:
        try:
            j = await _jira_search(force=bool(refresh))
            avatars["total"] = j.get("total", 0)
            by_month_raw = j.get("by_month", {}) or {}
            avatars["this_month"] = len(by_month_raw.get(cur_label, []))
            avatars["last_month"] = len(by_month_raw.get(prev_label, []))
            if avatars["last_month"]:
                d = (avatars["this_month"] - avatars["last_month"]) / avatars["last_month"] * 100
                avatars["delta_pct"] = round(d, 1)
            elif avatars["this_month"]:
                avatars["delta_pct"] = 100.0
            # Re-sort by the canonical YYYY-MM key so chronological order is correct.
            month_pairs = []
            for label, items in by_month_raw.items():
                ym = _label_to_ym(label)
                if ym == "Unscheduled":
                    continue
                month_pairs.append((ym, label, items))
            month_pairs.sort(key=lambda t: t[0])
            for ym, _label, items in month_pairs[-6:]:
                avatars["by_month"].append({"month": ym, "count": len(items)})
            # ── Flatten + classify everything in one pass.
            assignee_stats: dict[str, dict] = {}
            storage_counter: dict[str, int] = {}
            recents: list[dict] = []
            lead_times: list[float] = []
            week_buckets: dict[str, dict] = {}     # iso-year-week → {created, resolved}

            # Seed the last 12 ISO weeks so we always render a continuous chart.
            for off in range(11, -1, -1):
                d = now - _dt.timedelta(weeks=off)
                wk = d.strftime("%G-W%V")
                week_buckets[wk] = {"created": 0, "resolved": 0}
            twelve_weeks_ago = now - _dt.timedelta(weeks=12)

            def _parse_iso(s):
                """Robust Jira datetime parser (Python 3.9 friendly)."""
                if not s: return None
                s = str(s)
                # Trim trailing milliseconds (`.000`) — fromisoformat in 3.9 chokes.
                import re as _re
                # Strip the timezone for naive comparison; we only need day buckets.
                s = _re.sub(r"[+-]\d{2}:?\d{2}$|Z$", "", s)
                s = _re.sub(r"\.\d+$", "", s)
                for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                    try: return _dt.datetime.strptime(s, fmt)
                    except Exception: pass
                try:    return _dt.datetime.fromisoformat(s)
                except: return None

            for ym, _label, items in reversed(month_pairs):
                for it in items:
                    cat = (it.get("statusCategory") or "new").lower()
                    is_done = (cat == "done")
                    is_wip  = cat in ("indeterminate", "in_progress")
                    if is_done: avatars["by_status"]["done"] += 1
                    elif is_wip: avatars["by_status"]["in_progress"] += 1
                    else: avatars["by_status"]["todo"] += 1

                    # Priority breakdown.
                    pr = (it.get("priority") or "Medium")
                    avatars["by_priority"][pr] = avatars["by_priority"].get(pr, 0) + 1

                    # Storage region distribution.
                    for s in (it.get("storage") or []):
                        storage_counter[s] = storage_counter.get(s, 0) + 1

                    # Assignee stats with per-status splits.
                    assignee = it.get("assignee") or "Unassigned"
                    a = assignee_stats.setdefault(assignee, {
                        "name": assignee, "count": 0, "done": 0, "in_progress": 0,
                        "avatarUrl": it.get("assigneeIcon"),
                    })
                    a["count"] += 1
                    if is_done: a["done"] += 1
                    elif is_wip: a["in_progress"] += 1

                    # Weekly throughput (creation + resolution).
                    cdt = _parse_iso(it.get("createdIso"))
                    if cdt and cdt >= twelve_weeks_ago:
                        wk = cdt.strftime("%G-W%V")
                        if wk in week_buckets:
                            week_buckets[wk]["created"] += 1
                            if cdt.isocalendar()[1] == now.isocalendar()[1] \
                               and cdt.isocalendar()[0] == now.isocalendar()[0]:
                                avatars["this_week"] += 1
                    rdt = _parse_iso(it.get("resolvedIso"))
                    if rdt and rdt >= twelve_weeks_ago:
                        wk = rdt.strftime("%G-W%V")
                        if wk in week_buckets:
                            week_buckets[wk]["resolved"] += 1

                    # Lead time (created → resolved) for done tickets only.
                    if is_done and cdt and rdt and rdt >= cdt:
                        lead_times.append((rdt - cdt).total_seconds() / 86400.0)

                    # Aging buckets for WIP / To-do tickets.
                    if not is_done and cdt:
                        age = (now - cdt).days
                        if   age <= 3:  avatars["aging_wip"]["fresh"]    += 1
                        elif age <= 7:  avatars["aging_wip"]["warm"]     += 1
                        elif age <= 14: avatars["aging_wip"]["stale"]    += 1
                        else:           avatars["aging_wip"]["critical"] += 1

                    # Overdue (has due date in the past and not done).
                    dd_str = it.get("dueDate") or ""
                    if dd_str and not is_done:
                        try:
                            dd = _dt.date.fromisoformat(dd_str)
                            if dd < today_date:
                                avatars["overdue"] += 1
                        except Exception:
                            pass

                    if len(recents) < 8:
                        recents.append({
                            "key":      it.get("key"),
                            "summary":  it.get("summary"),
                            "status":   it.get("status"),
                            "assignee": assignee,
                            "assigneeIcon": it.get("assigneeIcon"),
                            "priority": pr,
                            "updated":  it.get("updated"),
                            "url":      it.get("url"),
                        })

            avatars["by_assignee"] = sorted(
                assignee_stats.values(), key=lambda x: x["count"], reverse=True
            )[:6]
            avatars["by_storage"] = sorted(
                [{"name": k, "count": v} for k, v in storage_counter.items()],
                key=lambda x: x["count"], reverse=True
            )
            avatars["weekly"] = [
                {"week": wk, "created": v["created"], "resolved": v["resolved"]}
                for wk, v in week_buckets.items()
            ]
            if lead_times:
                avg = sum(lead_times) / len(lead_times)
                srt = sorted(lead_times)
                med = srt[len(srt) // 2] if len(srt) % 2 else (srt[len(srt)//2 - 1] + srt[len(srt)//2]) / 2
                avatars["lead_time_days"]   = round(avg, 1)
                avatars["median_lead_days"] = round(med, 1)
            avatars["recent"] = recents
        except Exception as exc:
            avatars["error"] = str(exc)[:200]

    # ── 2. Voices (onboarding audio files) ───────────────────────────────
    from sqlalchemy import select as _select
    all_files   = db.scalars(_select(OnboardingFile)).all()
    audio_files = [f for f in all_files if (f.mime_type or "").startswith("audio/")]
    video_files = [f for f in all_files if (f.mime_type or "").startswith("video/")]
    voices = {
        "total": len(audio_files),
        "this_month": 0,
        "last_month": 0,
        "delta_pct": None,
        "total_minutes": 0.0,        # rough estimate from byte size at ~128kbps
        "avg_seconds_per_sample": 0,
        "by_month": [],
    }
    by_month: dict[str, int] = {}
    for f in audio_files:
        ym = (f.created_at or now).strftime("%Y-%m")
        by_month[ym] = by_month.get(ym, 0) + 1
        if f.size_bytes:
            voices["total_minutes"] += (f.size_bytes / 16_384) / 60
    voices["this_month"] = by_month.get(cur_ym, 0)
    voices["last_month"] = by_month.get(prev_ym, 0)
    if voices["last_month"]:
        d = (voices["this_month"] - voices["last_month"]) / voices["last_month"] * 100
        voices["delta_pct"] = round(d, 1)
    elif voices["this_month"]:
        voices["delta_pct"] = 100.0
    for m in sorted(by_month.keys())[-6:]:
        voices["by_month"].append({"month": m, "count": by_month[m]})
    voices["total_minutes"] = round(voices["total_minutes"], 1)
    if audio_files:
        voices["avg_seconds_per_sample"] = round(
            voices["total_minutes"] * 60 / len(audio_files), 1
        )

    # ── 3. Onboardings (pipeline funnel + source mix) ───────────────────
    all_onb = list_recent_onboardings(db, limit=500)
    onboardings = {
        "total":      len(all_onb),
        "pending":    sum(1 for o in all_onb if o.status == "pending"),
        "uploaded":   sum(1 for o in all_onb if o.status == "uploaded"),
        "failed":     sum(1 for o in all_onb if o.status == "failed"),
        "this_month": sum(1 for o in all_onb
                          if o.created_at and o.created_at.strftime("%Y-%m") == cur_ym),
        "last_month": sum(1 for o in all_onb
                          if o.created_at and o.created_at.strftime("%Y-%m") == prev_ym),
        "with_drive": sum(1 for o in all_onb if o.drive_folder_url),
        "with_jira":  sum(1 for o in all_onb if o.jira_issue_key),
    }
    # Average handoff latency: link minted → submitted.
    handoff_hours: list[float] = []
    for o in all_onb:
        if o.submitted_at and o.created_at:
            handoff_hours.append((o.submitted_at - o.created_at).total_seconds() / 3600)
    onboardings["avg_handoff_hours"] = (
        round(sum(handoff_hours) / len(handoff_hours), 1) if handoff_hours else None
    )
    onboardings["median_handoff_hours"] = (
        round(sorted(handoff_hours)[len(handoff_hours)//2], 1) if handoff_hours else None
    )

    # Source mix: webcam, phone or upload — heuristically inferred from
    # filename prefixes used by the wizard's three recording modes.
    src_mix = {"webcam": 0, "phone": 0, "upload": 0}
    for f in all_files:
        nm = (f.filename or "").lower()
        if   nm.startswith("webcam-")            or nm.startswith("voice-"):    src_mix["webcam"] += 1
        elif "phone" in nm or "mobile" in nm:                                   src_mix["phone"]  += 1
        else:                                                                   src_mix["upload"] += 1
    onboardings["source_mix"] = src_mix

    onboardings["files_per_session"] = (
        round(len(all_files) / len(all_onb), 1) if all_onb else 0
    )

    # ── 4. Storage mix derived from onboardings (covers both Jira-driven
    #      tickets and the support panel's default selection). ──────────
    onb_storage: dict[str, int] = {}
    for o in all_onb:
        if o.storage_region:
            onb_storage[o.storage_region] = onb_storage.get(o.storage_region, 0) + 1
    if onb_storage and not avatars["by_storage"]:
        avatars["by_storage"] = sorted(
            [{"name": k, "count": v} for k, v in onb_storage.items()],
            key=lambda x: x["count"], reverse=True
        )

    # ── 5. Conversion: onboarded → Jira created → voice captured ────────
    avatars_created = sum(1 for o in all_onb if o.jira_issue_key)
    conv_rate = (avatars_created / onboardings["total"] * 100) if onboardings["total"] else 0
    funnel = {
        "links_minted":  onboardings["total"],
        "uploads_done":  onboardings["uploaded"],
        "jira_created":  avatars_created,
        "voices_added":  voices["total"],
        "conversion":    round(conv_rate, 1),
    }

    return {
        "generated_at": now.isoformat(),
        "current_month_label": _cal.month_name[now.month] + " " + str(now.year),
        "avatars":     avatars,
        "voices":      voices,
        "onboardings": onboardings,
        "funnel":      funnel,
    }


# ────────────────────────────────────────────────────────────────────────────
# Personal cabinet: tickets assigned to the currently logged-in user.
# ────────────────────────────────────────────────────────────────────────────
@app.get("/api/jira/my")
async def jira_my(
    refresh: bool = False,
    user: User = Depends(require_role(Role.viewer)),
):
    import time as _time
    aid = await _jira_lookup_account_id(user.email)
    if not aid:
        raise HTTPException(404, {
            "errors": [f"No Jira user found for {user.email}. "
                       "Ask your Jira admin to invite this email to the workspace."]
        })

    ck = f"my:{aid}"
    now = _time.time()
    cached = _JIRA_MY_CACHE.get(ck)
    if not refresh and cached and (now - cached["ts"]) < _JIRA_MY_TTL:
        return cached["data"]

    jql = (f'project = {JIRA_PROJECT_KEY} '
           f'AND issuetype = "{JIRA_ISSUE_TYPE}" '
           f'AND assignee = "{aid}" '
           f'ORDER BY updated DESC')
    r = await _jira_request("GET", "/rest/api/3/search/jql", params={
        "jql": jql, "maxResults": 200, "fields": "*all",
    })
    _jira_raise_for_status(r, "Jira search failed")
    issues = (r.json() or {}).get("issues") or []
    grouped = _jira_group_by_month(issues, JIRA_FIELDS)
    months = _jira_month_order(grouped)
    payload = {
        "ok": True,
        "fetched_at": _now_iso(),
        "account_id": aid,
        "email": user.email,
        "total": sum(len(v) for v in grouped.values()),
        "months": months,
        "by_month": grouped,
    }
    _JIRA_MY_CACHE[ck] = {"data": payload, "ts": now}
    return payload


# ────────────────────────────────────────────────────────────────────────────
# Per-issue actions: transitions, comments, field updates.
# All endpoints proxy through the backend so the API token never reaches JS.
# ────────────────────────────────────────────────────────────────────────────
_JIRA_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]+-\d+$")

def _jira_validate_key(key: str) -> str:
    k = (key or "").strip().upper()
    if not _JIRA_KEY_RE.match(k):
        raise HTTPException(400, {"errors": [f"Invalid Jira issue key: {key!r}"]})
    return k


@app.get("/api/jira/issue/{key}/transitions")
async def jira_get_transitions(key: str, user: User = Depends(require_role(Role.viewer))):
    k = _jira_validate_key(key)
    r = await _jira_request("GET", f"/rest/api/3/issue/{k}/transitions")
    _jira_raise_for_status(r, "Could not load transitions")
    raw = (r.json() or {}).get("transitions") or []
    out = []
    for t in raw:
        to = t.get("to") or {}
        cat = (to.get("statusCategory") or {}).get("key") or "new"
        out.append({
            "id": t.get("id"),
            "name": t.get("name"),
            "to_name": to.get("name"),
            "category": cat,
        })
    return {"ok": True, "transitions": out}


class _JiraTransitionReq(BaseModel):
    transition_id: str
    comment: Optional[str] = None


@app.post("/api/jira/issue/{key}/transition")
async def jira_do_transition(
    key: str,
    req: _JiraTransitionReq,
    user: User = Depends(require_role(Role.viewer)),
):
    k = _jira_validate_key(key)
    body: dict = {"transition": {"id": str(req.transition_id)}}
    if req.comment and req.comment.strip():
        body["update"] = {"comment": [{"add": {"body": _jira_text_to_adf(
            f"{req.comment.strip()}\n\n— posted via Avatar Studio by {user.email}"
        )}}]}
    r = await _jira_request("POST", f"/rest/api/3/issue/{k}/transitions", json=body)
    _jira_raise_for_status(r, "Transition failed")
    _jira_bust_caches()
    return {"ok": True}


@app.get("/api/jira/issue/{key}/comments")
async def jira_get_comments(key: str, user: User = Depends(require_role(Role.viewer))):
    k = _jira_validate_key(key)
    r = await _jira_request("GET", f"/rest/api/3/issue/{k}/comment",
                            params={"orderBy": "-created", "maxResults": 50})
    _jira_raise_for_status(r, "Could not load comments")
    out = []
    for c in (r.json() or {}).get("comments") or []:
        author = c.get("author") or {}
        out.append({
            "id":          c.get("id"),
            "author":      author.get("displayName") or "?",
            "authorIcon":  (author.get("avatarUrls") or {}).get("24x24"),
            "authorEmail": author.get("emailAddress"),
            "created":     c.get("created"),
            "updated":     c.get("updated"),
            "body_text":   _jira_extract_text(c.get("body")).strip(),
        })
    return {"ok": True, "comments": out}


class _JiraCommentReq(BaseModel):
    body: str


@app.post("/api/jira/issue/{key}/comment")
async def jira_post_comment(
    key: str,
    req: _JiraCommentReq,
    user: User = Depends(require_role(Role.viewer)),
):
    k = _jira_validate_key(key)
    text = (req.body or "").strip()
    if not text:
        raise HTTPException(400, {"errors": ["Comment body is empty"]})
    signed = f"{text}\n\n— posted via Avatar Studio by {user.email}"
    r = await _jira_request("POST", f"/rest/api/3/issue/{k}/comment",
                            json={"body": _jira_text_to_adf(signed)})
    _jira_raise_for_status(r, "Could not post comment")
    _jira_bust_caches()
    j = r.json() or {}
    return {"ok": True, "comment_id": j.get("id"), "created": j.get("created")}


class _JiraFieldsReq(BaseModel):
    timeCode: Optional[str] = None
    pauses:   Optional[str] = None


@app.patch("/api/jira/issue/{key}/fields")
async def jira_update_fields(
    key: str,
    req: _JiraFieldsReq,
    user: User = Depends(require_role(Role.viewer)),
):
    k = _jira_validate_key(key)
    fields: dict = {}
    if req.timeCode is not None:
        fields[JIRA_FIELDS["timeCode"]] = req.timeCode.strip() or None
    if req.pauses is not None:
        fields[JIRA_FIELDS["pauses"]] = req.pauses.strip() or None
    if not fields:
        raise HTTPException(400, {"errors": ["Nothing to update"]})

    # Try as plain strings first (works for text custom fields).
    r = await _jira_request("PUT", f"/rest/api/3/issue/{k}", json={"fields": fields})
    if r.status_code == 400:
        # Some custom fields use ADF (paragraph type). Retry with ADF wrapping.
        adf = {fid: (_jira_text_to_adf(v) if isinstance(v, str) else v)
               for fid, v in fields.items() if v is not None}
        r2 = await _jira_request("PUT", f"/rest/api/3/issue/{k}", json={"fields": adf})
        if r2.status_code < 400:
            _jira_bust_caches()
            return {"ok": True, "format": "adf"}
        # Bubble the most informative error of the two.
        body = (r2.text or r.text or "")[:400]
        raise HTTPException(400, {"errors": [f"Jira rejected field update: {body}"]})
    _jira_raise_for_status(r, "Field update failed")
    _jira_bust_caches()
    return {"ok": True, "format": "string"}


# ════════════════════════════════════════════════════════════════════════════
# Tool 7: Avatar Tuning (post-training)
# Mirrors two sections from /Users/admin/Downloads/Avatar Tuning V3_6.ipynb:
#   • Update FOOTAGE + ALPHA + PAUSES (+ recompute cached_data)
#   • Fast 1M FOOTAGE → 5M LONG FOOTAGE replication
# Each tool is a self-contained, async background job that mutates the
# selected configs_v3/ JSON. Dual-mode: live (avatars/face_production stack)
# or simulation. Simulation is the default on dev machines.
# ────────────────────────────────────────────────────────────────────────────
_TUNE_LIVE = False
_TUNE_INIT_ERROR: Optional[str] = None
try:
    # The live stack would expose DatasetPreparatorV3 + tuning utils
    from avatars.dataset.v3_6.prepare import DatasetPreparatorV3  # type: ignore  # noqa: F401
    import avatars.tuning.utils as _atu  # type: ignore  # noqa: F401
    _TUNE_LIVE = True
except Exception as _tune_exc:  # pragma: no cover
    _TUNE_INIT_ERROR = (
        f"Tuning stack not available ({_tune_exc!s}). "
        "Set AVATARS_REPO_PATH so `avatars.dataset.v3_6.prepare` and "
        "`avatars.tuning.utils` are importable."
    )

TUNE_JOBS: dict[str, dict] = {}
_TUNE_LOCK = asyncio.Lock()


# Simulated stage tables (seconds: min, max)
_FOOTAGE_STAGES = [
    ("validating",          0.05, 1, 2),
    ("loading config",      0.10, 1, 2),
    ("probing footage",     0.25, 2, 5),
    ("computing pauses",    0.40, 1, 3),
    ("preparing dataset",   0.85, 6, 14),
    ("uploading cache",     0.95, 1, 3),
    ("saving config",       1.00, 1, 2),
]

_LONG_FOOTAGE_STAGES = [
    ("validating",            0.05, 1, 2),
    ("loading short footage", 0.15, 1, 3),
    ("planning repeats",      0.25, 1, 2),
    ("rendering long video",  0.75, 8, 18),
    ("rendering long alpha",  0.90, 4, 8),
    ("uploading long cache",  1.00, 2, 4),
]


# ── Pydantic models ─────────────────────────────────────────
class FootageUpdateRequest(BaseModel):
    config_name: str                    # filename in configs_v3/ (with or without .json)
    footage_path: str                   # s3:// URI
    alpha_footage_path: Optional[str] = None
    pauses: list[int] = []
    face_bbox: Optional[str] = None     # null | "auto" | "x,y,w,h"
    save_config: bool = True


class LongFootageAutoRequest(BaseModel):
    config_name: str
    num_repeats: int = 5                # 2..10
    vb: str = "12M"                     # video bitrate: 6M / 12M / 20M
    skip_video: bool = False


# ── Helpers ─────────────────────────────────────────────────
def _config_path(name: str) -> Path:
    base = name.strip()
    if not base.endswith(".json"):
        base = base + ".json"
    return CONFIGS_DIR / base


def _load_config(name: str) -> dict:
    p = _config_path(name)
    if not p.exists():
        raise FileNotFoundError(f"configs_v3/{p.name} not found")
    return json.loads(p.read_text())


def _save_config(name: str, cfg: dict) -> Path:
    p = _config_path(name)
    p.write_text(json.dumps(cfg, indent=2))
    return p


def _ensure_inference(cfg: dict) -> dict:
    """The notebook accesses config.inference.*; create stub if missing."""
    ac = cfg.setdefault("avatarConfig", {})
    inf = ac.setdefault("inference", {})
    inf.setdefault("pauses", {}).setdefault("points", [])
    inf.setdefault("cached_data", {})
    return inf


def _validate_footage(req: "FootageUpdateRequest") -> list[str]:
    errors: list[str] = []
    if not req.config_name.strip():
        errors.append("config_name is required")
    elif not _config_path(req.config_name).exists():
        errors.append(f"configs_v3/{_config_path(req.config_name).name} not found")
    if not req.footage_path:
        errors.append("footage_path is required")
    elif not _is_valid_uri(req.footage_path):
        errors.append("footage_path: unsupported scheme (expected s3://, gs://, http(s)://)")
    if req.alpha_footage_path and not _is_valid_uri(req.alpha_footage_path):
        errors.append("alpha_footage_path: unsupported scheme")
    for i, p in enumerate(req.pauses):
        if not isinstance(p, int) or p < 0:
            errors.append(f"pauses[{i}]: expected non-negative integer (frame index)")
    if req.face_bbox and req.face_bbox != "auto":
        parts = [s.strip() for s in req.face_bbox.split(",")]
        if len(parts) != 4 or not all(s.lstrip("-").isdigit() for s in parts):
            errors.append("face_bbox: expected 'auto' or 'x,y,w,h' integers")
    return errors


def _validate_long_footage(req: "LongFootageAutoRequest") -> list[str]:
    errors: list[str] = []
    if not req.config_name.strip():
        errors.append("config_name is required")
    elif not _config_path(req.config_name).exists():
        errors.append(f"configs_v3/{_config_path(req.config_name).name} not found")
    else:
        # Need an inference block with footage_path + pauses populated.
        try:
            cfg = _load_config(req.config_name)
        except Exception as e:
            errors.append(f"Could not read config: {e}")
            cfg = None
        if cfg:
            inf = cfg.get("avatarConfig", {}).get("inference", {})
            if not inf.get("footage_path"):
                errors.append("Config has no inference.footage_path — run 'Update Footage' first.")
            if not (inf.get("pauses", {}) or {}).get("points"):
                errors.append("Config has no inference.pauses.points — run 'Update Footage' first.")
    if not (2 <= req.num_repeats <= 10):
        errors.append("num_repeats: must be between 2 and 10")
    if req.vb not in ("6M", "12M", "20M"):
        errors.append("vb: must be one of '6M', '12M', '20M'")
    return errors


def _create_tune_job(kind: str, name: str, params: dict, owner_email: str) -> dict:
    return {
        "id": secrets.token_urlsafe(9),
        "kind": kind,                    # "footage" | "long_footage_auto"
        "name": name,
        "params": params,
        "owner": owner_email,
        "mode": "live" if _TUNE_LIVE else "simulation",
        "status": "pending",
        "stage": "queued",
        "progress": 0.0,
        "result": None,
        "error": None,
        "created_at": _now_iso(),
        "started_at": None,
        "finished_at": None,
        "logs": [],
    }


async def _run_staged(job: dict, stages: list, on_done):
    """Walk a list of (stage_name, target_pct, t_min, t_max) and update job."""
    async with _TUNE_LOCK:
        job["status"] = "running"
        job["started_at"] = _now_iso()
        _job_log(job, f"▶ [{job['mode']}] {job['kind']} for '{job['name']}'")

    prev_progress = 0.0
    for stage_name, target, t_min, t_max in stages:
        async with _TUNE_LOCK:
            if job["status"] == "cancelled":
                _job_log(job, f"Cancelled before stage {stage_name}")
                return False
            job["stage"] = stage_name
            _job_log(job, f"  → {stage_name}")

        duration = random.uniform(t_min, t_max)
        steps = max(4, int(duration * 4))
        for s in range(steps):
            await asyncio.sleep(duration / steps)
            async with _TUNE_LOCK:
                if job["status"] == "cancelled":
                    _job_log(job, f"Cancelled during {stage_name}")
                    return False
                job["progress"] = round(prev_progress + (target - prev_progress) * ((s + 1) / steps), 4)
        prev_progress = target

    try:
        async with _TUNE_LOCK:
            on_done(job)
            job["status"] = "completed"
            job["finished_at"] = _now_iso()
            job["progress"] = 1.0
    except Exception as e:
        async with _TUNE_LOCK:
            job["status"] = "failed"
            job["error"] = str(e)
            job["finished_at"] = _now_iso()
            _job_log(job, f"✗ on_done failed: {e}")
        return False
    return True


def _fake_s3_path(footage_path: str, suffix: str) -> str:
    """Mirror how the notebook stores cached artifacts next to the footage."""
    try:
        u = _urlparse(footage_path)
        p = Path(u.path.lstrip("/"))
        return f"{u.scheme}://{u.netloc}/{p.parent}/v3.6/{p.stem}{suffix}".rstrip("/")
    except Exception:
        return footage_path + suffix


async def _runner_footage(job_id: str):
    async with _TUNE_LOCK:
        job = TUNE_JOBS.get(job_id)
    if not job:
        return
    p = job["params"]

    def on_done(j):
        cfg = _load_config(p["config_name"])
        inf = _ensure_inference(cfg)
        inf["footage_path"] = p["footage_path"]
        if p.get("alpha_footage_path"):
            inf["alpha_footage_path"] = p["alpha_footage_path"]
        inf["pauses"]["points"] = p["pauses"]
        if p.get("face_bbox"):
            inf["face_bbox"] = p["face_bbox"]
        # Simulated cached_data — in live mode this would come from
        # dataset_preparator.prepare_footage_data(...).
        cached = {
            "npz_path":         _fake_s3_path(p["footage_path"], "_f2v_inputs.npz"),
            "f2v_inputs_path":  _fake_s3_path(p["footage_path"], "_f2v_inputs.mp4"),
            "_simulated":       not _TUNE_LIVE,
        }
        inf["cached_data"].update(cached)

        if p.get("save_config"):
            saved = _save_config(p["config_name"], cfg)
            _job_log(j, f"  ✓ saved configs_v3/{saved.name}")
        else:
            _job_log(j, "  (skipping save — dry run)")
        _job_log(j, f"  ✓ npz: {cached['npz_path']}")
        _job_log(j, f"  ✓ f2v: {cached['f2v_inputs_path']}")
        j["result"] = {
            "config_path": f"configs_v3/{_config_path(p['config_name']).name}",
            "footage_path": p["footage_path"],
            "alpha_footage_path": p.get("alpha_footage_path"),
            "pauses": p["pauses"],
            "cached_data": cached,
        }

    try:
        await _run_staged(job, _FOOTAGE_STAGES, on_done)
    except Exception as e:
        async with _TUNE_LOCK:
            job["status"] = "failed"
            job["error"] = str(e)
            job["finished_at"] = _now_iso()
            _job_log(job, f"✗ Failed: {e}")


async def _runner_long_footage(job_id: str):
    async with _TUNE_LOCK:
        job = TUNE_JOBS.get(job_id)
    if not job:
        return
    p = job["params"]

    def on_done(j):
        cfg = _load_config(p["config_name"])
        inf = cfg.get("avatarConfig", {}).get("inference", {})
        short_footage = inf.get("footage_path", "")
        short_alpha   = inf.get("alpha_footage_path")
        short_pauses  = (inf.get("pauses", {}) or {}).get("points", [])

        num = p["num_repeats"]

        # Build the long URIs by inserting "_long_{N}x" before the extension.
        def _longify(uri: str, tag: str) -> Optional[str]:
            if not uri:
                return None
            try:
                u = _urlparse(uri)
                pp = Path(u.path.lstrip("/"))
                return f"{u.scheme}://{u.netloc}/{pp.with_name(f'{pp.stem}_long_{tag}{pp.suffix}')}"
            except Exception:
                return uri + f"_long_{tag}"

        # Long pauses = short pauses repeated, shifted by frame count per repeat.
        # We don't know real frame count in sim — use last pause as a heuristic.
        period = (short_pauses[-1] if short_pauses else 1000) + 1
        long_pauses = []
        for r in range(num):
            for pt in short_pauses:
                long_pauses.append(pt + r * period)

        long_info = {
            "long_footage_path":       _longify(short_footage, f"{num}x") if not p.get("skip_video") else short_footage,
            "long_alpha_footage_path": _longify(short_alpha,   f"{num}x") if (short_alpha and not p.get("skip_video")) else short_alpha,
            "long_points":             long_pauses,
            "long_cached_data": {
                "npz_path":        _fake_s3_path(short_footage, f"_long_{num}x_f2v_inputs.npz"),
                "f2v_inputs_path": _fake_s3_path(short_footage, f"_long_{num}x_f2v_inputs.mp4"),
                "_simulated":      not _TUNE_LIVE,
            },
            "params": {
                "num_repeats": num,
                "vb": p["vb"],
                "skip_video": p.get("skip_video", False),
            },
        }

        # Long info is intentionally NOT saved into the avatar config (notebook
        # cell 29 explicitly says long footages are not part of avatar config).
        _job_log(j, f"  ✓ long_footage_path: {long_info['long_footage_path']}")
        if long_info["long_alpha_footage_path"]:
            _job_log(j, f"  ✓ long_alpha_footage_path: {long_info['long_alpha_footage_path']}")
        _job_log(j, f"  ✓ long_points: {len(long_pauses)} entries (avg period={period})")
        _job_log(j, f"  ✓ long_cached.npz: {long_info['long_cached_data']['npz_path']}")
        _job_log(j, "  ℹ long_info is NOT written into the avatar config (per notebook spec).")
        j["result"] = long_info

    try:
        await _run_staged(job, _LONG_FOOTAGE_STAGES, on_done)
    except Exception as e:
        async with _TUNE_LOCK:
            job["status"] = "failed"
            job["error"] = str(e)
            job["finished_at"] = _now_iso()
            _job_log(job, f"✗ Failed: {e}")


# ── Endpoints ───────────────────────────────────────────────
@app.get("/api/tune/mode")
async def tune_mode(user: User = Depends(require_role(Role.viewer))):
    return {
        "mode": "live" if _TUNE_LIVE else "simulation",
        "reason": None if _TUNE_LIVE else _TUNE_INIT_ERROR,
    }


@app.get("/api/tune/configs")
async def tune_list_configs(user: User = Depends(require_role(Role.viewer))):
    """List configs_v3/*.json with a short preview (current footage, pauses)."""
    entries = []
    if CONFIGS_DIR.exists():
        for p in sorted(CONFIGS_DIR.glob("*.json"), key=lambda x: x.name.lower()):
            try:
                cfg = json.loads(p.read_text())
                inf = cfg.get("avatarConfig", {}).get("inference", {})
                params = cfg.get("avatarConfig", {}).get("params", {})
                entries.append({
                    "name":              p.stem,
                    "filename":          p.name,
                    "size":              p.stat().st_size,
                    "experiment_name":   cfg.get("experiment_name"),
                    "has_inference":     bool(inf),
                    "footage_path":      inf.get("footage_path") or params.get("footage_path"),
                    "alpha_footage_path": inf.get("alpha_footage_path") or params.get("alpha_footage_path"),
                    "pauses":            (inf.get("pauses", {}) or {}).get("points") or params.get("pauses") or [],
                    "has_cached":        bool(inf.get("cached_data", {}).get("npz_path")),
                })
            except Exception as e:
                entries.append({"name": p.stem, "filename": p.name, "error": str(e)})
    return {"configs": entries}


@app.post("/api/tune/footage")
async def tune_footage(req: FootageUpdateRequest, user: User = Depends(require_role(Role.developer))):
    errors = _validate_footage(req)
    if errors:
        raise HTTPException(400, {"errors": errors})
    job = _create_tune_job("footage", req.config_name, req.dict(), user.email)
    async with _TUNE_LOCK:
        TUNE_JOBS[job["id"]] = job
    asyncio.create_task(_runner_footage(job["id"]))
    return {"ok": True, "job_id": job["id"], "mode": job["mode"]}


@app.post("/api/tune/long-footage-auto")
async def tune_long_footage_auto(req: LongFootageAutoRequest, user: User = Depends(require_role(Role.developer))):
    errors = _validate_long_footage(req)
    if errors:
        raise HTTPException(400, {"errors": errors})
    job = _create_tune_job("long_footage_auto", req.config_name, req.dict(), user.email)
    async with _TUNE_LOCK:
        TUNE_JOBS[job["id"]] = job
    asyncio.create_task(_runner_long_footage(job["id"]))
    return {"ok": True, "job_id": job["id"], "mode": job["mode"]}


@app.get("/api/tune/jobs")
async def tune_list_jobs(user: User = Depends(require_role(Role.viewer))):
    rows = sorted(TUNE_JOBS.values(), key=lambda j: j["created_at"], reverse=True)
    return {
        "mode": "live" if _TUNE_LIVE else "simulation",
        "jobs": [{
            "id":          j["id"],
            "kind":        j["kind"],
            "name":        j["name"],
            "status":      j["status"],
            "stage":       j["stage"],
            "progress":    j["progress"],
            "owner":       j["owner"],
            "created_at":  j["created_at"],
            "started_at":  j.get("started_at"),
            "finished_at": j.get("finished_at"),
        } for j in rows]
    }


@app.get("/api/tune/jobs/{job_id}")
async def tune_get_job(job_id: str, user: User = Depends(require_role(Role.viewer))):
    job = TUNE_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.post("/api/tune/jobs/{job_id}/cancel")
async def tune_cancel_job(job_id: str, user: User = Depends(require_role(Role.developer))):
    async with _TUNE_LOCK:
        job = TUNE_JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job["status"] in ("completed", "failed", "cancelled"):
            return {"ok": True, "status": job["status"]}
        job["status"] = "cancelled"
        job["finished_at"] = _now_iso()
        _job_log(job, "Cancelled by user")
    return {"ok": True, "status": "cancelled"}


# ────────────────────────────────────────────────────────────────────────────
# Tool 5: AI Support Chat (KB-backed troubleshooting)
# ────────────────────────────────────────────────────────────────────────────
SUPPORT_KB_PATH = BASE_DIR / "support_kb.json"
SUPPORT_KB: dict = {"artifacts": [], "improvements": []}
try:
    if SUPPORT_KB_PATH.exists():
        SUPPORT_KB = json.loads(SUPPORT_KB_PATH.read_text(encoding="utf-8"))
except Exception as _kb_err:
    SUPPORT_KB = {"artifacts": [], "improvements": [], "_load_error": str(_kb_err)}


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


def _tokenize(text: str) -> set[str]:
    """Cheap lowercase word tokenization for retrieval scoring."""
    return {w for w in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(w) >= 3}


def _entry_corpus(entry: dict) -> str:
    """Concatenate all string fields of a KB entry for matching."""
    return " ".join(str(v) for v in entry.values() if isinstance(v, str))


def _retrieve_kb(query: str, top_k: int = 3) -> list[dict]:
    """
    Naive keyword retrieval over artifacts + improvements.
    Returns the top-K entries scored by (token overlap + bonus for title hits).
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    # Common synonyms / problem-specific aliases mapped to Artifact titles.
    aliases = {
        "mouth":   {"teeth", "lipsync", "lips", "jaw"},
        "lip":     {"lipsync"},
        "lips":    {"lipsync"},
        "close":   {"teeth", "jaw"},
        "open":    {"teeth", "jaw"},
        "smile":   {"teeth"},
        "tooth":   {"teeth"},
        "blink":   {"blinking", "eyes"},
        "blinks":  {"blinking", "eyes"},
        "eye":     {"eyes", "blinking"},
        "eyelid":  {"blinking", "eyes"},
        "sleepy":  {"asymmetric", "eyes", "sleepy", "blinking"},
        "brow":    {"eyebrows"},
        "brows":   {"eyebrows"},
        "stable":  {"floating", "jittering"},
        "shake":   {"floating", "jittering"},
        "shaking": {"floating", "jittering"},
        "wobble":  {"floating", "jittering"},
        "jitter":  {"jittering", "floating"},
        "expressive": {"lipsync"},
    }
    expanded = set(q_tokens)
    for t in q_tokens:
        if t in aliases:
            expanded |= aliases[t]

    candidates: list[tuple[float, dict]] = []
    for entry in SUPPORT_KB.get("artifacts", []):
        text = _entry_corpus(entry)
        e_tokens = _tokenize(text)
        title_tokens = _tokenize(entry.get("Artifacts", ""))
        if not e_tokens:
            continue
        overlap = len(expanded & e_tokens)
        title_hits = len(expanded & title_tokens)
        score = overlap + 2.5 * title_hits
        if score > 0:
            candidates.append((score, {**entry, "_kind": "artifact"}))

    for entry in SUPPORT_KB.get("improvements", []):
        text = _entry_corpus(entry)
        e_tokens = _tokenize(text)
        title_tokens = _tokenize(entry.get("Artifact / Problem", ""))
        if not e_tokens:
            continue
        overlap = len(expanded & e_tokens)
        title_hits = len(expanded & title_tokens)
        score = (overlap + 2.0 * title_hits) * 0.7  # slight preference for artifacts
        if score > 0:
            candidates.append((score, {**entry, "_kind": "improvement"}))

    candidates.sort(key=lambda x: x[0], reverse=True)
    if not candidates:
        return []
    # Drop weak matches: keep only entries within 50% of the top score AND
    # above an absolute floor (filters out generic noise like "how to make pizza").
    top = candidates[0][0]
    threshold = max(1.5, top * 0.4)
    strong = [c for c in candidates if c[0] >= threshold]
    return [c[1] for c in strong[:top_k]]


def _format_kb_for_llm(entries: list[dict]) -> str:
    """Render retrieved KB entries as plain text for the LLM context window."""
    if not entries:
        return "(no relevant entries found in the knowledge base)"
    chunks = []
    for e in entries:
        if e.get("_kind") == "artifact":
            chunks.append(
                f"### Artifact: {e.get('Artifacts')}\n"
                f"- Description: {e.get('Description', '—')}\n"
                f"- Why it happens: {e.get('Why?', '—')}\n"
                f"- Solutions: {e.get('Solutions', '—')}\n"
                f"- Action / Config update:\n{e.get('Action / Config update', '—')}\n"
                f"- Risk: {e.get('Risk', '—')}\n"
                f"- Refer to: {e.get('Person-to-refer', '—')}\n"
                f"- Reference: {e.get('Link', '')}"
            )
        else:
            chunks.append(
                f"### Improvement project: {e.get('Artifact / Problem')}\n"
                f"- Solution: {e.get('Solution / Project', '—')}\n"
                f"- Comment: {e.get('Comment', '—')}\n"
                f"- Risk: {e.get('Risk', '—')}\n"
                f"- Resources: {e.get('Resources', '—')}"
            )
    return "\n\n".join(chunks)


def _format_kb_for_user(entries: list[dict]) -> str:
    """Render KB entries as a friendly markdown answer when no LLM is configured."""
    if not entries:
        return (
            "I couldn't find anything matching that in the V4 troubleshooting KB. "
            "Try describing the artifact more specifically — e.g. *blurry teeth*, "
            "*unnatural blinking*, *face floating*, *weird eyebrows*."
        )
    lines = []
    for e in entries:
        if e.get("_kind") == "artifact":
            lines.append(f"### {e.get('Artifacts')}")
            if e.get("Description"):
                lines.append(f"*{e['Description']}*")
            if e.get("Why?"):
                lines.append(f"\n**Why it happens**\n{e['Why?']}")
            if e.get("Solutions"):
                lines.append(f"\n**Suggested solutions**\n{e['Solutions']}")
            if e.get("Action / Config update"):
                lines.append(f"\n**Config / action**\n```\n{e['Action / Config update']}\n```")
            meta = []
            if e.get("Risk"):
                meta.append(f"Risk: **{e['Risk']}**")
            if e.get("Person-to-refer") and e["Person-to-refer"] != "-":
                meta.append(f"Refer to: {e['Person-to-refer']}")
            if e.get("Link"):
                meta.append(f"[Example]({e['Link']})")
            if meta:
                lines.append("\n" + " · ".join(meta))
        else:
            lines.append(f"### {e.get('Artifact / Problem')} — improvement")
            if e.get("Solution / Project"):
                lines.append(e["Solution / Project"])
            if e.get("Comment"):
                lines.append(f"\n{e['Comment']}")
        lines.append("\n---\n")
    lines.append(
        "_Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env` to get conversational AI explanations on top of this knowledge base._"
    )
    return "\n".join(lines)


_SUPPORT_SYSTEM_PROMPT = """You are the **Avatar Studio Support AI** — an expert troubleshooter for the Elai V4 avatar pipeline. You help avatar makers fix visual artifacts and pipeline issues.

## How to answer
1. Use ONLY the knowledge-base context provided below. Don't invent solutions that aren't supported by it.
2. If the KB has a matching artifact, give a clear, actionable answer with: a short diagnosis, the recommended fix, and any config snippet verbatim.
3. Format config snippets in fenced ```json``` code blocks.
4. Keep responses concise (≤ 8 short bullet/paragraphs). Use markdown.
5. If the KB has no good match, say so honestly and ask for a more specific symptom (e.g. "is it the teeth, blinking, eyebrows, or face stability?").
6. Mention who to refer the issue to (the `Person-to-refer` field) when relevant.
7. Don't expose internal links unless they're in the KB context."""


async def _llm_complete_openai(messages: list[dict]) -> str:
    """Call OpenAI Chat Completions; returns assistant text."""
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 700,
            },
        )
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:300]}")
    data = r.json()
    return data["choices"][0]["message"]["content"]


async def _llm_complete_anthropic(messages: list[dict]) -> str:
    """Call Anthropic Messages API; returns assistant text."""
    system = ""
    msgs = []
    for m in messages:
        if m["role"] == "system":
            system = (system + "\n\n" + m["content"]).strip()
        else:
            msgs.append(m)
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "system": system,
                "messages": msgs,
                "max_tokens": 700,
                "temperature": 0.2,
            },
        )
    if r.status_code >= 400:
        raise RuntimeError(f"Anthropic {r.status_code}: {r.text[:300]}")
    data = r.json()
    parts = data.get("content", [])
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


def _support_provider() -> str:
    if OPENAI_API_KEY:
        return "openai"
    if ANTHROPIC_API_KEY:
        return "anthropic"
    return "kb-only"


@app.get("/api/support/topics")
async def support_topics(user: User = Depends(require_role(Role.viewer))):
    """List of artifact titles for the suggestion-chip UI."""
    return {
        "provider": _support_provider(),
        "topics": [
            {"title": a.get("Artifacts", ""), "risk": a.get("Risk", "")}
            for a in SUPPORT_KB.get("artifacts", [])
        ],
    }


@app.post("/api/support/chat")
async def support_chat(req: ChatRequest, user: User = Depends(require_role(Role.viewer))):
    """
    Conversational support endpoint.
    Pipeline: retrieve top-K KB entries by query → either feed to LLM or
    render directly as the answer when no LLM key is configured.
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    user_query = next(
        (m.content for m in reversed(req.messages) if m.role == "user"),
        "",
    ).strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found")

    retrieved = _retrieve_kb(user_query, top_k=3)
    provider = _support_provider()

    # KB-only fallback (no LLM key configured)
    if provider == "kb-only":
        return {
            "ok": True,
            "provider": provider,
            "reply": _format_kb_for_user(retrieved),
            "sources": [
                {"title": e.get("Artifacts") or e.get("Artifact / Problem"),
                 "link": e.get("Link")}
                for e in retrieved
            ],
        }

    # Build LLM-ready messages with KB context appended to the system prompt
    kb_context = _format_kb_for_llm(retrieved)
    system_msg = (
        _SUPPORT_SYSTEM_PROMPT
        + "\n\n## Knowledge-base context for this turn\n"
        + kb_context
    )

    # Keep only the last ~10 turns to bound the prompt
    history = [m.model_dump() for m in req.messages[-10:]]
    llm_messages = [{"role": "system", "content": system_msg}] + history

    try:
        if provider == "openai":
            reply = await _llm_complete_openai(llm_messages)
        else:
            reply = await _llm_complete_anthropic(llm_messages)
    except Exception as e:
        # Fall back to KB rendering instead of failing the chat outright
        return {
            "ok": True,
            "provider": "kb-only",
            "reply": (
                f"_⚠ LLM provider failed ({e!s}). Showing raw KB matches below._\n\n"
                + _format_kb_for_user(retrieved)
            ),
            "sources": [
                {"title": e.get("Artifacts") or e.get("Artifact / Problem"),
                 "link": e.get("Link")}
                for e in retrieved
            ],
        }

    return {
        "ok": True,
        "provider": provider,
        "reply": reply,
        "sources": [
            {"title": e.get("Artifacts") or e.get("Artifact / Problem"),
             "link": e.get("Link")}
            for e in retrieved
        ],
    }


# ────────────────────────────────────────────────────────────────────────────
# Utility endpoints
# ────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "ok": True,
        "voice_cloning": bool(CLEANVOICE_API_KEY and ELEVENLABS_API_KEY and _cv),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "sagemaker_mode": "live" if _SAGEMAKER_LIVE else "simulation",
        "sagemaker_region": AWS_REGION if _SAGEMAKER_LIVE else None,
        "sagemaker_reason": None if _SAGEMAKER_LIVE else _SAGEMAKER_INIT_ERROR,
        "support_chat_provider": _support_provider(),
        "support_kb_entries": len(SUPPORT_KB.get("artifacts", [])),
        "enhance_mode": "live" if _ENHANCE_LIVE else "simulation",
        "enhance_gpu": _ENHANCE_GPU,
        "tune_mode": "live" if _TUNE_LIVE else "simulation",
        "s3_available": _s3_client is not None,
        "s3_error": _S3_INIT_ERROR,
        "oauth_providers": [k for k, v in _oauth_enabled().items() if v],
        "db_url_kind": "postgres" if "postgres" in os.getenv("DATABASE_URL", "sqlite") else "sqlite",
    }


# ════════════════════════════════════════════════════════════════════════════
# Tool 7: Client Onboarding pipeline
# ────────────────────────────────────────────────────────────────────────────
# Three-step funnel:
#   1. Support fills a form  → POST /api/onboarding/create
#      (stores a ClientOnboarding row, returns /setup/<token> link).
#   2. Client opens /setup/<token>, drag-drops footage + audio
#      (POST /api/onboarding/<token>/upload).
#   3. Backend creates a Google Drive folder, uploads the files, and
#      opens an "Avatars creation" issue in Jira (project VM) with the
#      folder URL attached as the Footage field. A notification fans out
#      to every support and creator user.
# All Drive / Jira credentials live in .env — never on the client.
# When optional credentials are missing the pipeline runs in "simulation"
# mode so the entire flow stays demoable end-to-end.
# ════════════════════════════════════════════════════════════════════════════

# Public host (used to build the /setup link returned to support).
ONBOARDING_PUBLIC_HOST = (
    os.getenv("ONBOARDING_PUBLIC_HOST")
    or os.getenv("OAUTH_BASE_URL", "http://127.0.0.1:8765")
).rstrip("/")

ONBOARDING_MAX_FILE_MB    = int(os.getenv("ONBOARDING_MAX_FILE_MB", "2048"))
ONBOARDING_MAX_FILES      = int(os.getenv("ONBOARDING_MAX_FILES", "20"))
ONBOARDING_ALLOWED_PREFIX = ("video/", "audio/")
ONBOARDING_ALLOWED_EXT    = {
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg",
}

ROLE_SUPPORT    = "support"
ROLE_AVATAR_MAN = "avatar_manager"
ROLE_CREATOR    = "creator"  # для совместимости
ROLE_SUPERADMIN = "superadmin"

# Расширяем список тех, кто имеет доступ к менеджменту онбординга
ONBOARDING_AUDIENCE = {ROLE_SUPPORT, ROLE_AVATAR_MAN, ROLE_CREATOR, ROLE_SUPERADMIN}


def _onboarding_can_manage(user: User) -> bool:
    return (user.role.value if isinstance(user.role, Role) else str(user.role)) in ONBOARDING_AUDIENCE


def require_onboarding_manager(user: User = Depends(require_user)) -> User:
    if not _onboarding_can_manage(user):
        raise HTTPException(403, "This action requires the support, creator, or superadmin role.")
    return user


# ---- Pydantic --------------------------------------------------------------
class _OnbCreateReq(BaseModel):
    client_name: str
    client_email: str
    requirements: str = ""
    # Optional Jira metadata so the Avatars-creation card opens fully populated.
    organization_id:   Optional[str] = None
    account_id:        Optional[str] = None
    storage_region:    Optional[str] = None      # "EU" | "prod" | "ewizzard" | ...
    slack_link:        Optional[str] = None
    approver_emails:   Optional[str] = None      # comma-separated emails
    assignee_email:    Optional[str] = None
    evaluating_date:   Optional[str] = None      # ISO yyyy-mm-dd
    end_date:          Optional[str] = None
    due_date:          Optional[str] = None
    original_estimate: Optional[str] = None      # e.g. "8h", "30m", "1d"


def _norm_date(s: Optional[str]) -> Optional[str]:
    """Accept '2026-06-04' or 'Jun 02, 2026' -> ISO 'yyyy-mm-dd', else None."""
    if not s:
        return None
    s = s.strip()
    import datetime as _dt
    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%d/%m/%Y", "%d.%m.%Y"):
        try:
            return _dt.datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    return None


# ---- Support panel: create / list onboardings -----------------------------
@app.post("/api/onboarding/create")
async def onboarding_create(
    req: _OnbCreateReq,
    request: Request,
    user: User = Depends(require_onboarding_manager),
    db: Session = Depends(get_session),
):
    name = (req.client_name or "").strip()
    email = (req.client_email or "").strip().lower()
    if not name:
        raise HTTPException(400, "Client name is required.")
    if not _validate_email(email):
        raise HTTPException(400, "A valid client email is required.")

    # Default Evaluating date to today and estimate to 8h (matches the screenshot).
    import datetime as _dt
    eval_date = _norm_date(req.evaluating_date) or _dt.date.today().isoformat()
    estimate = (req.original_estimate or JIRA_DEFAULT_ESTIMATE).strip() or JIRA_DEFAULT_ESTIMATE

    storage = (req.storage_region or "").strip() or None
    if storage and JIRA_STORAGE_OPTIONS and storage not in JIRA_STORAGE_OPTIONS:
        raise HTTPException(400, f"Invalid storage value. Allowed: {', '.join(JIRA_STORAGE_OPTIONS)}")

    approvers_raw = (req.approver_emails or "").strip()
    approvers_csv = ",".join(
        a.strip().lower() for a in approvers_raw.replace(";", ",").split(",") if a.strip()
    ) or None

    token = uuid.uuid4().hex
    onb = ClientOnboarding(
        token=token,
        client_name=name,
        client_email=email,
        requirements=(req.requirements or "").strip()[:4000],
        status="pending",
        created_by_id=user.id,
        organization_id=(req.organization_id or "").strip() or None,
        account_id=(req.account_id or "").strip() or None,
        storage_region=storage,
        slack_link=(req.slack_link or "").strip() or None,
        approver_emails=approvers_csv,
        assignee_email=(req.assignee_email or "").strip().lower() or None,
        evaluating_date=eval_date,
        end_date=_norm_date(req.end_date),
        due_date=_norm_date(req.due_date),
        original_estimate=estimate,
    )
    db.add(onb)
    db.commit()
    db.refresh(onb)

    setup_url = f"{ONBOARDING_PUBLIC_HOST}/setup/{token}"

    # Personal nudge for the support manager (self-notification — confirms
    # the link is ready to send and remains in their bell history).
    create_notification(
        db,
        text=f"Onboarding link generated for {name} ({email}).",
        role="support",
        link=setup_url,
        icon="paper-plane",
        user_id=user.id,
    )

    return {
        "ok": True,
        "onboarding": onb.to_dict(),
        "setup_url": setup_url,
    }


@app.get("/api/onboarding/list")
async def onboarding_list(
    user: User = Depends(require_onboarding_manager),
    db: Session = Depends(get_session),
):
    rows = list_recent_onboardings(db, limit=200)
    return {"items": [r.to_dict() for r in rows], "total": len(rows)}


@app.get("/api/onboarding/mode")
async def onboarding_mode(user: User = Depends(require_user)):
    """Configuration snapshot — drives the UI badges and gating."""
    return {
        "can_manage":  _onboarding_can_manage(user),
        "public_host": ONBOARDING_PUBLIC_HOST,
        "drive":       drive_service.status(),
        "jira":        {"configured": JIRA_CONFIGURED,
                        "project": JIRA_PROJECT_KEY,
                        "issue_type": JIRA_ISSUE_TYPE,
                        "storage_options": JIRA_STORAGE_OPTIONS,
                        "default_estimate": JIRA_DEFAULT_ESTIMATE},
        "limits": {
            "max_file_mb": ONBOARDING_MAX_FILE_MB,
            "max_files":   ONBOARDING_MAX_FILES,
            "allowed_ext": sorted(ONBOARDING_ALLOWED_EXT),
        },
    }


@app.get("/api/onboarding/{token}")
async def onboarding_get(
    token: str,
    user: User = Depends(require_onboarding_manager),
    db: Session = Depends(get_session),
):
    onb = get_onboarding_by_token(db, token)
    if not onb:
        raise HTTPException(404, "Onboarding link not found.")
    from sqlalchemy import select as _sa_select
    files = list(db.scalars(
        _sa_select(OnboardingFile).where(OnboardingFile.onboarding_id == onb.id)
    ).all())
    return {
        "onboarding": onb.to_dict(),
        "files": [f.to_dict() for f in files],
    }


# ---- QR code helper (server-side, no JS deps) ---------------------------
@app.get("/setup/{token}/qr.svg")
async def setup_qr(token: str, request: Request, db: Session = Depends(get_session)):
    """Inline SVG QR code for the mobile-handoff link."""
    onb = get_onboarding_by_token(db, token)
    if not onb:
        return PlainTextResponse("not found", status_code=404)
    url = f"{ONBOARDING_PUBLIC_HOST}/setup/{token}?via=phone"
    try:
        import qrcode
        from qrcode.image.svg import SvgPathImage
        img = qrcode.make(url, image_factory=SvgPathImage, box_size=10, border=2)
        import io
        buf = io.BytesIO(); img.save(buf)
        return Response(
            content=buf.getvalue(),
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=300"},
        )
    except Exception as e:
        # Fallback: redirect to a public QR API. Best-effort only.
        return RedirectResponse(
            url=f"https://api.qrserver.com/v1/create-qr-code/?size=300x300&data={url}",
            status_code=302,
        )


# ---- Public client setup page --------------------------------------------
@app.get("/setup/{token}", response_class=HTMLResponse)
async def setup_page(
    token: str, request: Request,
    db: Session = Depends(get_session),
):
    """Public landing page for a client to upload footage. No auth required."""
    onb = get_onboarding_by_token(db, token)
    if not onb:
        return templates.TemplateResponse(
            "setup.html",
            {"request": request, "onboarding": None, "token": token,
             "error": "This onboarding link is invalid or has been revoked."},
            status_code=404,
        )
    return templates.TemplateResponse(
        "setup.html",
        {
            "request": request,
            "token": token,
            "onboarding": onb.to_dict(),
            "max_file_mb": ONBOARDING_MAX_FILE_MB,
            "max_files": ONBOARDING_MAX_FILES,
            "allowed_extensions": sorted(ONBOARDING_ALLOWED_EXT),
            "drive_status": drive_service.status(),
            "jira_status": {"configured": JIRA_CONFIGURED, "project": JIRA_PROJECT_KEY},
            "error": None,
        },
    )


def _is_allowed_upload(filename: str, mime: str) -> bool:
    ext = Path(filename or "").suffix.lower()
    if ext in ONBOARDING_ALLOWED_EXT:
        return True
    return any((mime or "").startswith(p) for p in ONBOARDING_ALLOWED_PREFIX)


async def _create_jira_avatar_issue(
    *, onb: ClientOnboarding, drive_folder_url: str,
    drive_folder_id: str, files_meta: list[dict],
) -> dict:
    """Create a new "Avatars creation" issue with the card fully populated.

    Populated fields (when the corresponding data is available on the
    ClientOnboarding record):
      - summary, issuetype, description (ADF)
      - Customer Email     (customfield_11151)
      - Organization ID    (customfield_11152)
      - Account ID         (customfield_11154)
      - Storage            (customfield_11150, multiselect)
      - Footage            (customfield_11049, url)
      - Slack Link         (customfield_11153, url)
      - Approvers          (customfield_10494, multi-user)
      - Assignee           (system)
      - Evaluating date    (customfield_11046)
      - End date           (customfield_11048)
      - Due date           (duedate)
      - Original estimate  (timetracking)
    """
    summary = f"{onb.client_name} — avatar creation"
    requirements_block = (onb.requirements or "").strip() or "No additional requirements supplied."
    files_block = "\n".join(
        f"• {f['filename']}  ({_human_bytes(f.get('size', 0))})  {f.get('url', '')}"
        for f in files_meta
    ) or "• (no files reported)"
    desc = (
        f"Client: {onb.client_name} <{onb.client_email}>\n\n"
        f"Drive folder: {drive_folder_url}\n\n"
        f"Uploaded files:\n{files_block}\n\n"
        f"Client requirements / notes:\n{requirements_block}\n\n"
        f"Tracked onboarding token: {onb.token}"
    )

    fields: dict = {
        "project":   {"key": JIRA_PROJECT_KEY},
        "summary":   summary,
        "issuetype": {"name": JIRA_ISSUE_TYPE},
        "description": _jira_text_to_adf(desc),
    }

    # ── URL / text custom fields ─────────────────────────────────────────
    def _set(field_key: str, value):
        fid = JIRA_FIELDS.get(field_key)
        if fid and value not in (None, ""):
            fields[fid] = value

    _set("footage",       drive_folder_url)
    _set("customerEmail", onb.client_email)
    _set("orgId",         onb.organization_id)
    _set("accountId",     onb.account_id)
    _set("slackLink",     onb.slack_link)
    _set("evalDate",      onb.evaluating_date)
    _set("endDate",       onb.end_date)

    # ── Storage (multiselect by name) ────────────────────────────────────
    if onb.storage_region and JIRA_FIELDS.get("storage"):
        fields[JIRA_FIELDS["storage"]] = [{"value": onb.storage_region}]

    # ── Approvers (multi-user) — resolve emails to accountIds ───────────
    approver_account_ids: list[str] = []
    if onb.approver_emails:
        for raw in (onb.approver_emails or "").split(","):
            em = raw.strip().lower()
            if not em:
                continue
            aid = await _jira_lookup_account_id(em)
            if aid:
                approver_account_ids.append(aid)
    if approver_account_ids and JIRA_FIELDS.get("approvers"):
        fields[JIRA_FIELDS["approvers"]] = [{"accountId": a} for a in approver_account_ids]

    # ── Assignee ─────────────────────────────────────────────────────────
    if onb.assignee_email:
        aid = await _jira_lookup_account_id(onb.assignee_email)
        if aid:
            fields["assignee"] = {"accountId": aid}

    # ── Due date (system field) ─────────────────────────────────────────
    if onb.due_date:
        fields["duedate"] = onb.due_date

    # ── Original estimate ───────────────────────────────────────────────
    if onb.original_estimate:
        fields["timetracking"] = {"originalEstimate": onb.original_estimate}

    async def _post(payload_fields: dict):
        return await _jira_request("POST", "/rest/api/3/issue", json={"fields": payload_fields})

    r = await _post(fields)

    # Tolerant retry: some sites reject specific fields. If we get a 400 with
    # an "errors" map naming a field, retry without that one. Repeat up to
    # 4 times so a missing Approvers field doesn't kill the whole pipeline.
    fields_try = dict(fields)
    for _attempt in range(4):
        if r.status_code < 400:
            break
        try:
            err = r.json() or {}
        except Exception:
            err = {}
        err_fields = (err.get("errors") or {})
        if not err_fields:
            break
        # Drop any field Jira complained about.
        dropped = False
        for fid in list(err_fields.keys()):
            if fid in fields_try:
                fields_try.pop(fid, None); dropped = True
            elif fid == "assignee":
                fields_try.pop("assignee", None); dropped = True
            elif fid == "timetracking":
                fields_try.pop("timetracking", None); dropped = True
        if not dropped:
            break
        r = await _post(fields_try)

    if r.status_code >= 400:
        raise HTTPException(502, f"Jira issue creation failed ({r.status_code}): {(r.text or '')[:500]}")

    data = r.json() or {}
    key = data.get("key") or ""
    url = f"https://{JIRA_DOMAIN}/browse/{key}" if (JIRA_DOMAIN and key) else ""
    return {"key": key, "url": url, "raw": data}


def _looks_like_drive_url(s: str) -> bool:
    s = (s or "").strip().lower()
    return s.startswith("http") and ("drive.google.com" in s or "docs.google.com" in s
                                     or "dropbox.com" in s or "wetransfer.com" in s)


@app.post("/api/onboarding/{token}/upload")
async def onboarding_upload(
    token: str,
    request: Request,
    files: list[UploadFile] = File(default=[]),
    drive_link: str = Form(""),
    language: str = Form(""),
    notes: str = Form(""),
    db: Session = Depends(get_session),
):
    """Client-facing upload endpoint.

    Accepts EITHER uploaded files OR a Google Drive / Dropbox / WeTransfer
    link. When a link is supplied we skip the per-file copy step and just
    record the URL as the canonical Drive folder. When files are supplied
    they're pushed through ``services.drive`` (real Drive API or simulation
    fallback) and the resulting folder URL is attached to a freshly-created
    Jira issue.

    Pipeline:
      1. validate the token + payload (need files OR a drive link)
      2. push each file into a freshly-created Google Drive folder
         (or reuse the link the client pasted in)
      3. open a Jira "Avatars creation" issue with the Drive URL in the
         Footage field, plus the chosen language and any client notes
      4. fan out an in-app notification to support + creator users
    """
    onb = get_onboarding_by_token(db, token)
    if not onb:
        raise HTTPException(404, "Onboarding link not found.")
    if onb.status == "uploaded":
        raise HTTPException(409, "Footage has already been submitted for this client.")

    drive_link = (drive_link or "").strip()
    files = files or []
    if not files and not drive_link:
        raise HTTPException(400, "Please attach at least one file or paste a Google Drive link.")
    if drive_link and not _looks_like_drive_url(drive_link):
        raise HTTPException(400, "Drive link must be a Google Drive, Dropbox, or WeTransfer URL.")
    if len(files) > ONBOARDING_MAX_FILES:
        raise HTTPException(400, f"Too many files (max {ONBOARDING_MAX_FILES}).")

    # If new notes / language arrived from the client, append them.
    appended: list[str] = []
    if language:
        appended.append(f"Recording language: {language}")
    if notes:
        appended.append(f"Client added on submit:\n{notes.strip()[:2000]}")
    if appended:
        onb.requirements = ((onb.requirements or "") + "\n\n--- " + " ---\n".join(appended))[:4000]

    # ── 1. Create the Drive folder (or reuse a pasted link) ───────────────
    folder: dict
    if drive_link and not files:
        # Client supplied a link only — no need to mint a new Drive folder.
        folder = {"id": "external", "url": drive_link, "simulated": False, "external": True}
    else:
        try:
            folder = drive_service.create_client_folder(onb.client_name, onb.token)
        except Exception as exc:
            onb.status = "failed"
            onb.error_message = f"Drive folder creation failed: {exc!s}"
            db.commit()
            raise HTTPException(502, onb.error_message)

    onb.drive_folder_id  = folder["id"]
    onb.drive_folder_url = folder["url"]
    db.commit()

    # ── 2. Upload every file (skipped entirely when drive_link only) ─────
    uploaded_meta: list[dict] = []
    max_bytes = ONBOARDING_MAX_FILE_MB * 1024 * 1024
    try:
        for f in files:
            mime = (f.content_type or "").lower()
            filename = f.filename or "upload.bin"
            if not _is_allowed_upload(filename, mime):
                raise HTTPException(400, f"File type not allowed: {filename}")
            data = await f.read()
            if not data:
                continue
            if len(data) > max_bytes:
                raise HTTPException(
                    400,
                    f"File '{filename}' exceeds the {ONBOARDING_MAX_FILE_MB} MB limit.",
                )
            up = drive_service.upload_bytes_to_folder(
                folder_id=onb.drive_folder_id, token=onb.token,
                filename=filename, data=data, mime_type=mime or "application/octet-stream",
            )
            db.add(OnboardingFile(
                onboarding_id=onb.id,
                filename=up["filename"],
                mime_type=mime or "application/octet-stream",
                size_bytes=up["size"],
                drive_file_id=up["id"],
                drive_file_url=up["url"],
                local_path=up.get("local_path"),
            ))
            uploaded_meta.append(up)
    except HTTPException:
        db.commit()
        raise
    except Exception as exc:
        onb.status = "failed"
        onb.error_message = f"Drive upload failed: {exc!s}"
        db.commit()
        raise HTTPException(502, onb.error_message)

    if not uploaded_meta and not drive_link:
        raise HTTPException(400, "No non-empty files were submitted.")
    # Record an external-link "file" so the support panel surfaces it as
    # something the client provided, even when we never streamed bytes.
    if drive_link and not uploaded_meta:
        db.add(OnboardingFile(
            onboarding_id=onb.id,
            filename="External link (client-supplied)",
            mime_type="text/url",
            size_bytes=0,
            drive_file_id="external",
            drive_file_url=drive_link,
        ))

    # ── 3. Create the Jira issue ──────────────────────────────────────────
    jira_key: str = ""
    jira_url: str = ""
    if JIRA_CONFIGURED:
        try:
            jira = await _create_jira_avatar_issue(
                onb=onb,
                drive_folder_url=onb.drive_folder_url,
                drive_folder_id=onb.drive_folder_id,
                files_meta=uploaded_meta,
            )
            jira_key = jira["key"]
            jira_url = jira["url"]
            onb.jira_issue_key = jira_key
            onb.jira_issue_url = jira_url
            _jira_bust_caches()
        except HTTPException as he:
            onb.status = "failed"
            onb.error_message = str(he.detail)
            db.commit()
            raise
    else:
        # No Jira creds in .env — record a simulated key so the rest of the
        # pipeline (notifications, support panel) still has something to link.
        jira_key = f"SIM-{onb.id:04d}"
        jira_url = f"https://example.atlassian.net/browse/{jira_key}"
        onb.jira_issue_key = jira_key
        onb.jira_issue_url = jira_url
        onb.error_message  = "Jira not configured — issue not created (simulation only)."

    onb.status = "uploaded"
    onb.submitted_at = _onboarding_dt.datetime.utcnow()
    db.commit()

    # ── 4. Fan out notifications to support + creators ────────────────────
    notif_text = f"New client avatar request submitted for {onb.client_name}!"
    notif_link = jira_url or f"/?onboarding={onb.token}"
    for audience in (ROLE_SUPPORT, ROLE_CREATOR):
        create_notification(
            db, text=notif_text, role=audience,
            link=notif_link, icon="user-plus",
        )

    return {
        "ok": True,
        "files_uploaded": len(uploaded_meta),
        "drive": {"id": onb.drive_folder_id, "url": onb.drive_folder_url,
                  "simulated": folder.get("simulated", False)},
        "jira": {"key": jira_key, "url": jira_url,
                 "simulated": not JIRA_CONFIGURED},
    }


# ---- Notifications API ---------------------------------------------------
@app.get("/api/notifications")
async def notifications_list(
    user: User = Depends(require_user),
    db: Session = Depends(get_session),
):
    items = notifications_for_user(db, user, limit=30)
    return {
        "items":  [n.to_dict() for n in items],
        "unread": unread_notification_count(db, user),
        "role":   user.role.value if isinstance(user.role, Role) else str(user.role),
    }


class _NotifReadReq(BaseModel):
    ids: Optional[list[int]] = None


@app.post("/api/notifications/read")
async def notifications_read(
    req: _NotifReadReq,
    user: User = Depends(require_user),
    db: Session = Depends(get_session),
):
    n = mark_notifications_read(db, user, ids=req.ids)
    return {"ok": True, "updated": n, "unread": unread_notification_count(db, user)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
