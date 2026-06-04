"""
Avatar Studio — database layer.

Uses SQLAlchemy 2.0 ORM. Defaults to a local SQLite file
(`./avatars_studio.db`) for zero-config dev. Set DATABASE_URL in .env to
switch to PostgreSQL, e.g.:

    DATABASE_URL=postgresql+psycopg2://user:pwd@host:5432/avatars_studio

Roles model:
    viewer      — can browse, test TTS, view jobs
    developer   — can clone voices, submit training jobs, browse S3
    superadmin  — can deploy avatars to production, manage roles

The first registered user is automatically promoted to `superadmin`.
"""

from __future__ import annotations

import datetime as dt
import enum
import os
import secrets
from pathlib import Path
from typing import Optional

import bcrypt
from sqlalchemy import (
    String, Integer, DateTime, ForeignKey, Enum as SAEnum,
    create_engine, select, func, delete,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, sessionmaker, Session,
)

# ────────────────────────────────────────────────────────────────────────────
# Engine
# ────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SQLITE = f"sqlite:///{BASE_DIR / 'avatars_studio.db'}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE).strip()

# SQLite needs a special connect arg for multithreaded FastAPI workers
_engine_kw: dict = {"future": True, "echo": False}
if DATABASE_URL.startswith("sqlite"):
    _engine_kw["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **_engine_kw)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# ────────────────────────────────────────────────────────────────────────────
# Roles
# ────────────────────────────────────────────────────────────────────────────
class Role(str, enum.Enum):
    viewer = "viewer"
    developer = "developer"
    creator = "creator"      # avatar maker — full access to production tools
    support = "support"      # client onboarding manager
    manager = "manager"      # high-level stats observer (custom avatars/voices)
    superadmin = "superadmin"


# Privilege ordering. `creator`, `support`, `manager` and `developer` all sit at
# the same effective tier (each has its own scope of pages); `viewer` is below
# and `superadmin` is above all.
ROLE_LEVEL = {
    Role.viewer:     1,
    Role.developer:  2,
    Role.support:    2,
    Role.creator:    2,
    Role.manager:    2,
    Role.superadmin: 3,
}


def role_at_least(actual: Role | str, required: Role | str) -> bool:
    """Compare two roles by privilege level (viewer < developer < superadmin)."""
    actual_r = actual if isinstance(actual, Role) else Role(actual)
    required_r = required if isinstance(required, Role) else Role(required)
    return ROLE_LEVEL[actual_r] >= ROLE_LEVEL[required_r]


# ────────────────────────────────────────────────────────────────────────────
# Models
# ────────────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id:             Mapped[int]            = mapped_column(Integer, primary_key=True)
    email:          Mapped[str]            = mapped_column(String(255), unique=True, index=True)
    name:           Mapped[Optional[str]]  = mapped_column(String(255), nullable=True)
    avatar_url:     Mapped[Optional[str]]  = mapped_column(String(1024), nullable=True)
    password_hash:  Mapped[Optional[str]]  = mapped_column(String(255), nullable=True)
    role:           Mapped[Role]           = mapped_column(SAEnum(Role), default=Role.viewer)

    # OAuth provider info ("password", "google", "github")
    provider:       Mapped[str]            = mapped_column(String(32), default="password")
    provider_id:    Mapped[Optional[str]]  = mapped_column(String(255), nullable=True, index=True)

    created_at:     Mapped[dt.datetime]    = mapped_column(DateTime, default=dt.datetime.utcnow)
    last_login_at:  Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name or self.email.split("@")[0],
            "avatar_url": self.avatar_url,
            "role": self.role.value,
            "provider": self.provider,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
        }


class PasswordReset(Base):
    """One-time reset token. Created when user requests password recovery
    or when a SuperAdmin generates a reset link. Expires after 1 hour and
    can be used only once."""
    __tablename__ = "password_resets"

    id:         Mapped[int]            = mapped_column(Integer, primary_key=True)
    user_id:    Mapped[int]            = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token:      Mapped[str]            = mapped_column(String(128), unique=True, index=True)
    issued_by:  Mapped[str]            = mapped_column(String(32), default="self")   # "self" | "admin"
    created_at: Mapped[dt.datetime]    = mapped_column(DateTime, default=dt.datetime.utcnow)
    expires_at: Mapped[dt.datetime]    = mapped_column(DateTime)
    used_at:    Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)


class ClientOnboarding(Base):
    """A client invitation issued by a support manager.

    Lifecycle:
        pending  → support created the link, no upload yet
        uploaded → client opened /setup/<token> and uploaded files; Drive
                   folder created; Jira issue created
        failed   → upload completed but Drive/Jira pipeline raised an error
    """
    __tablename__ = "client_onboardings"

    id:                Mapped[int]   = mapped_column(Integer, primary_key=True)
    token:             Mapped[str]   = mapped_column(String(64), unique=True, index=True)
    client_name:       Mapped[str]   = mapped_column(String(255))
    client_email:      Mapped[str]   = mapped_column(String(255))
    requirements:      Mapped[str]   = mapped_column(String(4096), default="")
    status:            Mapped[str]   = mapped_column(String(32), default="pending", index=True)

    # Support manager who created the link
    created_by_id:     Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    # Extra metadata captured upfront so the Jira card is fully populated.
    # All optional — if missing, the corresponding Jira field is left blank.
    organization_id:   Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    account_id:        Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    storage_region:    Mapped[Optional[str]] = mapped_column(String(32),  nullable=True)
    slack_link:        Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    approver_emails:   Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)  # comma-separated
    assignee_email:    Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    evaluating_date:   Mapped[Optional[str]] = mapped_column(String(32),  nullable=True)   # ISO yyyy-mm-dd
    end_date:          Mapped[Optional[str]] = mapped_column(String(32),  nullable=True)
    due_date:          Mapped[Optional[str]] = mapped_column(String(32),  nullable=True)
    original_estimate: Mapped[Optional[str]] = mapped_column(String(32),  nullable=True)   # e.g. "8h"

    # Filled in after the client uploads files
    drive_folder_id:   Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    drive_folder_url:  Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    jira_issue_key:    Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    jira_issue_url:    Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    error_message:     Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)

    created_at:        Mapped[dt.datetime]   = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    submitted_at:      Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "token": self.token,
            "client_name": self.client_name,
            "client_email": self.client_email,
            "requirements": self.requirements or "",
            "status": self.status,
            "created_by_id": self.created_by_id,
            "organization_id": self.organization_id,
            "account_id": self.account_id,
            "storage_region": self.storage_region,
            "slack_link": self.slack_link,
            "approver_emails": self.approver_emails,
            "assignee_email": self.assignee_email,
            "evaluating_date": self.evaluating_date,
            "end_date": self.end_date,
            "due_date": self.due_date,
            "original_estimate": self.original_estimate,
            "drive_folder_id": self.drive_folder_id,
            "drive_folder_url": self.drive_folder_url,
            "jira_issue_key": self.jira_issue_key,
            "jira_issue_url": self.jira_issue_url,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
        }


class OnboardingFile(Base):
    """Single uploaded file inside an onboarding submission."""
    __tablename__ = "onboarding_files"

    id:               Mapped[int]   = mapped_column(Integer, primary_key=True)
    onboarding_id:    Mapped[int]   = mapped_column(Integer, ForeignKey("client_onboardings.id", ondelete="CASCADE"), index=True)
    filename:         Mapped[str]   = mapped_column(String(512))
    mime_type:        Mapped[str]   = mapped_column(String(128), default="application/octet-stream")
    size_bytes:       Mapped[int]   = mapped_column(Integer, default=0)
    drive_file_id:    Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    drive_file_url:   Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    local_path:       Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    created_at:       Mapped[dt.datetime]   = mapped_column(DateTime, default=dt.datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "drive_file_id": self.drive_file_id,
            "drive_file_url": self.drive_file_url,
        }


class Notification(Base):
    """In-app notification surfaced through the header bell.

    `role` is the audience: 'support' | 'creator' | 'superadmin' | 'developer'
    | 'viewer' | 'all'. Users see notifications matching their own role plus
    those addressed to 'all'.
    """
    __tablename__ = "notifications"

    id:         Mapped[int]            = mapped_column(Integer, primary_key=True)
    text:       Mapped[str]            = mapped_column(String(1024))
    role:       Mapped[str]            = mapped_column(String(32), default="all", index=True)
    link:       Mapped[Optional[str]]  = mapped_column(String(1024), nullable=True)
    icon:       Mapped[str]            = mapped_column(String(64), default="bell")
    is_read:    Mapped[bool]           = mapped_column(default=False, index=True)
    user_id:    Mapped[Optional[int]]  = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    created_at: Mapped[dt.datetime]    = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "role": self.role,
            "link": self.link,
            "icon": self.icon,
            "is_read": bool(self.is_read),
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# Create tables (works for both sqlite and postgres without alembic)
Base.metadata.create_all(engine)


# ── SQLite forward-compat: add columns that didn't exist in earlier
# revisions. SQLAlchemy's create_all only creates *missing tables*, not
# missing columns, so we patch on boot. Safe to call repeatedly.
def _sqlite_ensure_columns():
    if not DATABASE_URL.startswith("sqlite"):
        return
    from sqlalchemy import text as _sa_text
    patches = [
        ("notifications", "icon",    "VARCHAR(64) DEFAULT 'bell'"),
        ("notifications", "user_id", "INTEGER"),
        # ClientOnboarding extra Jira fields (added Jun 2026).
        ("client_onboardings", "organization_id",   "VARCHAR(128)"),
        ("client_onboardings", "account_id",        "VARCHAR(128)"),
        ("client_onboardings", "storage_region",    "VARCHAR(32)"),
        ("client_onboardings", "slack_link",        "VARCHAR(1024)"),
        ("client_onboardings", "approver_emails",   "VARCHAR(1024)"),
        ("client_onboardings", "assignee_email",    "VARCHAR(255)"),
        ("client_onboardings", "evaluating_date",   "VARCHAR(32)"),
        ("client_onboardings", "end_date",          "VARCHAR(32)"),
        ("client_onboardings", "due_date",          "VARCHAR(32)"),
        ("client_onboardings", "original_estimate", "VARCHAR(32)"),
    ]
    with engine.begin() as conn:
        for table, col, ddl in patches:
            try:
                rows = conn.execute(_sa_text(f"PRAGMA table_info({table})")).fetchall()
                cols = {r[1] for r in rows}
                if col not in cols:
                    conn.execute(_sa_text(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}"))
            except Exception:
                pass

_sqlite_ensure_columns()


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    """bcrypt hash. Truncates to 72 bytes (bcrypt's hard limit) silently."""
    pwd_bytes = plain.encode("utf-8")[:72]
    return bcrypt.hashpw(pwd_bytes, bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8")[:72], hashed.encode("utf-8"))
    except Exception:
        return False


def get_session() -> Session:
    """FastAPI dependency: yields a Session, ensures it's closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def is_first_user(db: Session) -> bool:
    """True if there are no users in the database yet."""
    return db.scalar(select(func.count()).select_from(User)) == 0


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.scalar(select(User).where(User.email == email.lower()))


def get_user_by_id(db: Session, uid: int) -> Optional[User]:
    return db.get(User, uid)


# ────────────────────────────────────────────────────────────────────────────
# Password reset
# ────────────────────────────────────────────────────────────────────────────
RESET_TOKEN_TTL_HOURS = 1


def create_password_reset(db: Session, user: User, issued_by: str = "self") -> PasswordReset:
    """Issue a new one-time reset token, invalidating any previous unused ones."""
    db.execute(
        delete(PasswordReset).where(
            PasswordReset.user_id == user.id,
            PasswordReset.used_at.is_(None),
        )
    )
    token = secrets.token_urlsafe(48)
    pr = PasswordReset(
        user_id=user.id,
        token=token,
        issued_by=issued_by,
        expires_at=dt.datetime.utcnow() + dt.timedelta(hours=RESET_TOKEN_TTL_HOURS),
    )
    db.add(pr)
    db.commit()
    db.refresh(pr)
    return pr


def consume_password_reset(db: Session, token: str) -> Optional[User]:
    """Validate + mark token as used. Returns the User if successful."""
    pr = db.scalar(select(PasswordReset).where(PasswordReset.token == token))
    if not pr:
        return None
    if pr.used_at is not None:
        return None
    if pr.expires_at < dt.datetime.utcnow():
        return None
    user = db.get(User, pr.user_id)
    if not user:
        return None
    pr.used_at = dt.datetime.utcnow()
    db.commit()
    return user


def peek_password_reset(db: Session, token: str) -> tuple[Optional[User], Optional[str]]:
    """Look up a reset token without consuming it. Returns (user, error_code).
    error_code: None | "not_found" | "expired" | "used"."""
    pr = db.scalar(select(PasswordReset).where(PasswordReset.token == token))
    if not pr:
        return None, "not_found"
    if pr.used_at is not None:
        return None, "used"
    if pr.expires_at < dt.datetime.utcnow():
        return None, "expired"
    user = db.get(User, pr.user_id)
    if not user:
        return None, "not_found"
    return user, None


# ────────────────────────────────────────────────────────────────────────────
# Notifications
# ────────────────────────────────────────────────────────────────────────────
def create_notification(
    db: Session, *, text: str, role: str = "all",
    link: Optional[str] = None, icon: str = "bell",
    user_id: Optional[int] = None,
) -> Notification:
    """Insert a notification visible to all users with the given role
    (or to a specific user when user_id is set)."""
    n = Notification(text=text, role=role, link=link, icon=icon, user_id=user_id)
    db.add(n)
    db.commit()
    db.refresh(n)
    return n


def notifications_for_user(db: Session, user: "User", limit: int = 30) -> list[Notification]:
    """Return notifications targeted at the user's role or directly to them,
    newest first."""
    role_val = user.role.value if isinstance(user.role, Role) else str(user.role)
    audience = {role_val, "all"}
    stmt = (
        select(Notification)
        .where(
            (Notification.role.in_(list(audience)))
            | (Notification.user_id == user.id)
        )
        .order_by(Notification.created_at.desc())
        .limit(limit)
    )
    return list(db.scalars(stmt).all())


def unread_notification_count(db: Session, user: "User") -> int:
    role_val = user.role.value if isinstance(user.role, Role) else str(user.role)
    audience = {role_val, "all"}
    stmt = (
        select(func.count(Notification.id))
        .where(
            (Notification.role.in_(list(audience)))
            | (Notification.user_id == user.id)
        )
        .where(Notification.is_read.is_(False))
    )
    return int(db.scalar(stmt) or 0)


def mark_notifications_read(
    db: Session, user: "User", ids: Optional[list[int]] = None,
) -> int:
    """Mark either the supplied notification ids or every notification the
    user is allowed to see as read. Returns the number of rows updated."""
    role_val = user.role.value if isinstance(user.role, Role) else str(user.role)
    audience = {role_val, "all"}
    q = select(Notification).where(
        (Notification.role.in_(list(audience)))
        | (Notification.user_id == user.id)
    ).where(Notification.is_read.is_(False))
    if ids:
        q = q.where(Notification.id.in_(ids))
    updated = 0
    for n in db.scalars(q).all():
        n.is_read = True
        updated += 1
    if updated:
        db.commit()
    return updated


# ────────────────────────────────────────────────────────────────────────────
# Client onboarding
# ────────────────────────────────────────────────────────────────────────────
def get_onboarding_by_token(db: Session, token: str) -> Optional[ClientOnboarding]:
    return db.scalar(select(ClientOnboarding).where(ClientOnboarding.token == token))


def list_recent_onboardings(db: Session, limit: int = 50) -> list[ClientOnboarding]:
    return list(
        db.scalars(
            select(ClientOnboarding)
            .order_by(ClientOnboarding.created_at.desc())
            .limit(limit)
        ).all()
    )


def get_or_create_oauth_user(
    db: Session, *, provider: str, provider_id: str,
    email: str, name: str | None, avatar_url: str | None,
) -> User:
    """Upsert an OAuth user. First user ever → superadmin."""
    email = email.lower().strip()
    user = db.scalar(
        select(User).where(User.provider == provider, User.provider_id == provider_id)
    ) or get_user_by_email(db, email)

    if user is None:
        role = Role.superadmin if is_first_user(db) else Role.viewer
        user = User(
            email=email, name=name, avatar_url=avatar_url,
            role=role, provider=provider, provider_id=provider_id,
        )
        db.add(user)
    else:
        # Refresh metadata on each login
        if name and not user.name:
            user.name = name
        if avatar_url:
            user.avatar_url = avatar_url
        if not user.provider_id:
            user.provider, user.provider_id = provider, provider_id

    user.last_login_at = dt.datetime.utcnow()
    db.commit()
    db.refresh(user)
    return user
