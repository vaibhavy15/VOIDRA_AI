"""
╔══════════════════════════════════════════════════════════════╗
║          VOIDRA DevAssist — Single-File Backend               ║
║          FastAPI + PostgreSQL + Gemini + Ollama              ║
╚══════════════════════════════════════════════════════════════╝

Run:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

API Endpoints:
    POST  /api/auth/signup
    POST  /api/auth/login
    GET   /api/auth/me
    POST  /api/debug
    POST  /api/upgrade
    POST  /api/generate
    GET   /api/chats
    GET   /api/chats/{chat_id}
    DELETE /api/chats/{chat_id}
    GET   /api/health
"""

# ══════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════
import os
import uuid
import logging
import traceback
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from typing import Optional, Literal
from pathlib import Path

import bcrypt
import httpx
from dotenv import load_dotenv
from jose import JWTError, jwt

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, field_validator

from sqlalchemy import (
    Column, String, Text, Boolean, DateTime,
    ForeignKey, Enum as SAEnum, select, desc
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.exc import IntegrityError                      # ← FIX: was missing
from sqlalchemy.ext.asyncio import (
    AsyncSession, create_async_engine, async_sessionmaker
)
from sqlalchemy.orm import DeclarativeBase, relationship
from google import genai
from dotenv import load_dotenv
from fastapi.responses import FileResponse
# ── Load .env ──────────────────────────────────────────────────
load_dotenv()

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════
class Config:
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost/voidra_devassist"
    )
    # Auth
    SECRET_KEY:   str = os.getenv("SECRET_KEY", "change-this-in-production-please")
    ALGORITHM:    str = "HS256"
    TOKEN_EXPIRE_MINUTES: int = int(os.getenv("TOKEN_EXPIRE_MINUTES", "10080"))  # 7 days
    # AI
    GEMINI_API_KEY:  str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL:    str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL:    str = os.getenv("OLLAMA_MODEL", "llama3")
    OLLAMA_THRESHOLD: int = int(os.getenv("OLLAMA_TOKEN_THRESHOLD", "400"))
    # CORS
    FRONTEND_ORIGIN: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5500")


cfg = Config()
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(name)s │ %(message)s")
log = logging.getLogger("voidra")

# ══════════════════════════════════════════════════════════════
#  DATABASE LAYER
# ══════════════════════════════════════════════════════════════
engine = create_async_engine(
    cfg.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False,
)
AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession,
    expire_on_commit=False, autoflush=False, autocommit=False,
)


class Base(DeclarativeBase):
    pass


# ── Models ─────────────────────────────────────────────────────
def _now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id              = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username        = Column(String(64),  unique=True,  nullable=False, index=True)
    email           = Column(String(255), unique=True,  nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active       = Column(Boolean,     default=True)
    created_at      = Column(DateTime(timezone=True), default=_now)
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")


class Chat(Base):
    __tablename__ = "chats"
    id         = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id    = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    module     = Column(SAEnum("debug", "upgrade", "generate", name="module_enum"), nullable=False)
    title      = Column(String(255), default="New Chat")
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)
    user     = relationship("User",    back_populates="chats")
    messages = relationship("Message", back_populates="chat",
                            cascade="all, delete-orphan", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"
    id         = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id    = Column(PGUUID(as_uuid=True), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    role       = Column(SAEnum("user", "assistant", name="role_enum"), nullable=False)
    content    = Column(Text, nullable=False)
    ai_backend = Column(String(32), nullable=True)   # "gemini" | "ollama"
    created_at = Column(DateTime(timezone=True), default=_now)
    chat = relationship("Chat", back_populates="messages")


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database tables ready ✓")


# ══════════════════════════════════════════════════════════════
#  AUTH LAYER
# ══════════════════════════════════════════════════════════════
# ── Use bcrypt directly — passlib is incompatible with bcrypt 4.x ──
bearer = HTTPBearer(auto_error=False)


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(pw: str, hashed: str) -> bool:
    return bcrypt.checkpw(pw.encode("utf-8"), hashed.encode("utf-8"))


def create_token(user_id: uuid.UUID) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=cfg.TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": str(user_id), "exp": exp}, cfg.SECRET_KEY, algorithm=cfg.ALGORITHM)


def decode_token(token: str) -> Optional[str]:
    try:
        return jwt.decode(token, cfg.SECRET_KEY, algorithms=[cfg.ALGORITHM]).get("sub")
    except JWTError:
        return None


async def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    db:    AsyncSession = Depends(get_db),
) -> User:
    exc = HTTPException(status_code=401, detail="Invalid or missing token",
                        headers={"WWW-Authenticate": "Bearer"})
    if not creds:
        raise exc
    uid = decode_token(creds.credentials)
    if not uid:
        raise exc
    result = await db.execute(select(User).where(User.id == uuid.UUID(uid)))
    user   = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise exc
    return user


# ══════════════════════════════════════════════════════════════
#  PYDANTIC SCHEMAS                     ← FIX: moved above routes
# ══════════════════════════════════════════════════════════════

class SignupRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

    @field_validator("username")
    @classmethod
    def check_username(cls, v):
        if not (3 <= len(v.strip()) <= 32):
            raise ValueError("Username must be 3–32 characters")
        return v.strip()

    @field_validator("password")
    @classmethod
    def check_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    created_at: datetime


# ── FIX: AIRequest / AIResponse were missing entirely ─────────
class AIRequest(BaseModel):
    code:     str = ""
    prompt:   Optional[str] = None
    language: Optional[str] = None
    chat_id:  Optional[str] = None


class AIResponse(BaseModel):
    reply:      str
    chat_id:    str
    message_id: str
    ai_backend: str


class MessageOut(BaseModel):
    id:         str
    role:       str
    content:    str
    ai_backend: Optional[str]
    created_at: datetime


class ChatSummary(BaseModel):
    id:            str
    module:        str
    title:         str
    updated_at:    datetime
    message_count: int


class ChatDetail(BaseModel):
    id:       str
    module:   str
    title:    str
    messages: list[MessageOut]


# ══════════════════════════════════════════════════════════════
#  AI SERVICE LAYER
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPTS = {
    "debug": """You are VOIDRA Debugger — a specialist in identifying and fixing code errors.
Your ONLY job is debugging. Never add features or refactor beyond fixing bugs.

For every request you MUST:
1. Identify ALL bugs with line references where possible.
2. Explain WHY each error occurs (root cause analysis).
3. Provide a corrected, complete version of the code with comments on each fix.
4. Summarise all changes in a numbered list.

Format your response with these exact sections:
## ERRORS FOUND
## ROOT CAUSE ANALYSIS
## FIXED CODE
## SUMMARY OF CHANGES""",

    "upgrade": """You are VOIDRA Upgrader — a specialist in code optimisation and refactoring.
Your ONLY job is improving code quality. Never debug bugs or add new features.

For every request you MUST:
1. Analyse performance bottlenecks, anti-patterns, and readability issues.
2. Refactor structure using language-specific best practices and idiomatic patterns.
3. Provide the complete upgraded code with inline comments explaining improvements.
4. Guarantee ALL existing functionality is preserved — no breaking changes.

Format your response with these exact sections:
## ANALYSIS
## OPTIMISATIONS APPLIED
## UPGRADED CODE
## IMPROVEMENTS SUMMARY""",

    "generate": """You are VOIDRA Generator — a specialist in writing new code from specifications.
Your ONLY job is generating clean, production-ready code. Never debug or refactor existing code.

For every request you MUST:
1. Understand the full specification provided.
2. Generate complete, working code with NO placeholders or TODOs (unless explicitly asked).
3. Follow best practices for the language/framework specified.
4. Include clear inline comments for non-obvious logic.

Format your response with these exact sections:
## GENERATED CODE
## HOW IT WORKS
## USAGE EXAMPLE""",
}


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


async def _call_ollama(prompt: str) -> str:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            res = await client.post(
                f"{cfg.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": cfg.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                }
            )

            res.raise_for_status()
            data = res.json()
            return data.get("response", "")

    except Exception as e:
        print("Ollama error:", e)
        raise


async def _call_gemini(system: str, user_msg: str) -> str:
    client = genai.Client(api_key=cfg.GEMINI_API_KEY)

    response = client.models.generate_content(
        model=cfg.GEMINI_MODEL,
        contents=system + "\n\n" + user_msg
    )

    return response.text
async def run_ai(module, user_message, history=None):
    system = SYSTEM_PROMPTS[module]

    ctx = ""
    if history:
        for msg in history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            ctx += f"{role}: {msg['content']}\n"
        ctx = f"Previous conversation:\n{ctx}\n---\nCurrent request:\n"

    full_msg = ctx + user_message

    try:
        text = await _call_gemini(system, full_msg)
        return text, "gemini"
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI error: {e}")

# ══════════════════════════════════════════════════════════════
#  FASTAPI APPLICATION
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(
    title="VOIDRA DevAssist API",
    version="1.0.0",
    description="Hybrid AI developer assistant — Gemini + Ollama + PostgreSQL",
    lifespan=lifespan,
)
from fastapi.staticfiles import StaticFiles
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@app.get("/")
async def root():
    return {"message": "Voidra AI Backend Running 🚀"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # TEMP for deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve index.html at root ───────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_PATH = BASE_DIR.parent / "frontend" / "index.html"

@app.get("/")
async def root():
    if FRONTEND_PATH.exists():
        return FileResponse(str(FRONTEND_PATH))
    return {"message": "Frontend not found"}


# ══════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════════════════════

@app.post("/api/auth/signup", response_model=TokenResponse, status_code=201)
async def signup(body: SignupRequest, db: AsyncSession = Depends(get_db)):
    try:
        existing = await db.execute(
            select(User).where(
                (User.email == body.email) | (User.username == body.username)
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Email or username already taken")

        user = User(
            username=body.username.strip(),
            email=body.email.strip(),
            hashed_password=hash_password(body.password)
        )
        db.add(user)
        await db.flush()

        return TokenResponse(
            access_token=create_token(user.id),
            user_id=str(user.id),
            username=user.username
        )

    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=400, detail="User already exists")

    except HTTPException:
        raise

    except Exception as e:
        await db.rollback()
        log.error("SIGNUP ERROR: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/api/auth/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(select(User).where(User.email == body.email))
        user = result.scalar_one_or_none()

        if not user or not verify_password(body.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        return TokenResponse(
            access_token=create_token(user.id),
            user_id=str(user.id),
            username=user.username
        )

    except HTTPException:
        raise

    except Exception as e:
        log.error("LOGIN ERROR: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/api/auth/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return UserResponse(
        id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        created_at=current_user.created_at
    )


# ── Alias routes for frontend compatibility ────────────────────
@app.post("/api/signup", response_model=TokenResponse)
async def signup_alias(body: SignupRequest, db: AsyncSession = Depends(get_db)):
    return await signup(body, db)


@app.post("/api/login", response_model=TokenResponse)
async def login_alias(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    return await login(body, db)


# ══════════════════════════════════════════════════════════════
#  SHARED AI HANDLER
# ══════════════════════════════════════════════════════════════

async def _ai_handler(
    module:       str,
    body:         AIRequest,
    current_user: User,
    db:           AsyncSession,
) -> AIResponse:
    # ── Resolve or create chat ────────────────────────────────
    chat: Chat | None = None
    if body.chat_id:
        res  = await db.execute(
            select(Chat).where(Chat.id == uuid.UUID(body.chat_id),
                               Chat.user_id == current_user.id)
        )
        chat = res.scalar_one_or_none()

    if not chat:
        snippet = (body.prompt or body.code or "")[:50].strip()
        chat = Chat(user_id=current_user.id, module=module,
                    title=snippet or "New Chat")
        db.add(chat)
        await db.flush()

    # ── Load history ──────────────────────────────────────────
    hist = await db.execute(
        select(Message).where(Message.chat_id == chat.id).order_by(Message.created_at)
    )
    raw_history = [{"role": m.role, "content": m.content}
                   for m in hist.scalars().all()]

    # ── Build user message text ───────────────────────────────
    if module == "generate":
        user_text = (
            f"Language/framework: {body.language or 'auto-detect'}\n\n"
            f"Specification:\n{body.prompt or body.code}"
        )
    elif module == "debug":
        user_text = f"```\n{body.code}\n```"
        if body.prompt:
            user_text += f"\n\nAdditional notes from developer: {body.prompt}"
    else:  # upgrade
        user_text = f"```\n{body.code}\n```"
        if body.prompt:
            user_text += f"\n\nUpgrade goals: {body.prompt}"

    # ── Call hybrid AI ────────────────────────────────────────
    reply, backend = await run_ai(module, user_text, raw_history)

    # ── Persist messages ──────────────────────────────────────
    db.add(Message(chat_id=chat.id, role="user",      content=user_text))
    ai_msg = Message(chat_id=chat.id, role="assistant", content=reply, ai_backend=backend)
    db.add(ai_msg)
    await db.flush()

    return AIResponse(reply=reply, chat_id=str(chat.id),
                      message_id=str(ai_msg.id), ai_backend=backend)


# ══════════════════════════════════════════════════════════════
#  AI MODULE ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.post("/api/debug", response_model=AIResponse,
          summary="Debug code — detect errors, explain root causes, provide fixes")
async def debug_code(
    body:         AIRequest,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    if not body.code.strip():
        raise HTTPException(400, "No code provided. Paste code in the 'code' field.")
    return await _ai_handler("debug", body, current_user, db)


@app.post("/api/upgrade", response_model=AIResponse,
          summary="Upgrade code — optimise, refactor, improve quality")
async def upgrade_code(
    body:         AIRequest,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    if not body.code.strip():
        raise HTTPException(400, "No code provided. Paste code in the 'code' field.")
    return await _ai_handler("upgrade", body, current_user, db)


@app.post("/api/generate", response_model=AIResponse,
          summary="Generate new code from a specification or prompt")
async def generate_code(
    body:         AIRequest,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    if not body.code.strip() and not body.prompt:
        raise HTTPException(400, "Provide a description in 'prompt' or reference code in 'code'.")
    return await _ai_handler("generate", body, current_user, db)


# ══════════════════════════════════════════════════════════════
#  CHAT HISTORY ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/api/chats", response_model=list[ChatSummary],
         summary="List all chats for current user (optional ?module= filter)")
async def list_chats(
    module:       str | None  = None,
    current_user: User        = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    q = select(Chat).where(Chat.user_id == current_user.id).order_by(desc(Chat.updated_at))
    if module:
        q = q.where(Chat.module == module)
    chats = (await db.execute(q)).scalars().all()

    result = []
    for c in chats:
        count = len((await db.execute(
            select(Message).where(Message.chat_id == c.id)
        )).scalars().all())
        result.append(ChatSummary(
            id=str(c.id), module=c.module, title=c.title,
            updated_at=c.updated_at, message_count=count,
        ))
    return result


@app.get("/api/chats/{chat_id}", response_model=ChatDetail,
         summary="Get a single chat with all messages")
async def get_chat(
    chat_id:      str,
    current_user: User        = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    res  = await db.execute(
        select(Chat).where(Chat.id == uuid.UUID(chat_id),
                           Chat.user_id == current_user.id)
    )
    chat = res.scalar_one_or_none()
    if not chat:
        raise HTTPException(404, "Chat not found")

    msgs = (await db.execute(
        select(Message).where(Message.chat_id == chat.id).order_by(Message.created_at)
    )).scalars().all()

    return ChatDetail(
        id=str(chat.id), module=chat.module, title=chat.title,
        messages=[
            MessageOut(id=str(m.id), role=m.role, content=m.content,
                       ai_backend=m.ai_backend, created_at=m.created_at)
            for m in msgs
        ],
    )


@app.delete("/api/chats/{chat_id}", status_code=204,
            summary="Delete a chat and all its messages")
async def delete_chat(
    chat_id:      str,
    current_user: User        = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    res  = await db.execute(
        select(Chat).where(Chat.id == uuid.UUID(chat_id),
                           Chat.user_id == current_user.id)
    )
    chat = res.scalar_one_or_none()
    if not chat:
        raise HTTPException(404, "Chat not found")
    await db.delete(chat)


# ══════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ══════════════════════════════════════════════════════════════

@app.get("/api/health", summary="Health check")
async def health(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(select(1))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"

    ollama_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{cfg.OLLAMA_BASE_URL}/api/tags")
            ollama_status = "online" if r.status_code == 200 else f"http {r.status_code}"
    except Exception:
        ollama_status = "offline"

    return {
        "status":   "online",
        "version":  "1.0.0",
        "database": db_status,
        "ollama":   ollama_status,
        "gemini":   "configured" if cfg.GEMINI_API_KEY else "not configured",
    }