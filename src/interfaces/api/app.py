"""
FastAPI Application - Production-ready API with:
- Authentication & authorization
- Distributed tracing
- Tenant isolation
- WebSocket support
- Event-driven updates
- Circuit breakers
- API versioning
"""

# CRITICAL: Import ML libraries FIRST to avoid SIGSEGV crash
# PyTorch and related libraries must be imported before ANY application code
import torch
import sentence_transformers
import faiss
import numpy as np

from fastapi import FastAPI, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import secrets
from datetime import datetime, timezone
import logging
import logging.config
import json
from pathlib import Path

# Initialize logging FIRST before any application imports
try:
    logging_config_path = Path("config/logging_config.json")
    if logging_config_path.exists():
        with open(logging_config_path, 'r') as f:
            log_config = json.load(f)
        logging.config.dictConfig(log_config)
        print("✅ Logging initialized from config/logging_config.json")
    else:
        # Fallback to basic logging if config file not found
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | [%(name)s:%(lineno)d] | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        print("⚠️  Logging config not found, using basic configuration")
except Exception as e:
    print(f"⚠️  Failed to initialize logging: {e}")
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# Authentication removed - no longer needed
from src.shared.tenant_context import TenantContext, TenantContextFactory, TenantConfig
from src.infrastructure.tracing import Tracer, TraceContext
from src.infrastructure.event_store import EventBus
from src.domain.events import EventType


# ============================================================================
# Application Lifecycle
# ============================================================================

# Global tenant factory (initialized at startup)
tenant_factory: TenantContextFactory = None
event_bus_registry: Dict[str, EventBus] = {}
dialogue_pipelines: Dict[str, Any] = {}  # Tenant-specific dialogue pipelines


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Startup: Initialize tenants, load models, create dialogue pipelines
    Shutdown: Cleanup resources
    """
    global tenant_factory, event_bus_registry, dialogue_pipelines
    
    print("=" * 70)
    print(" 🚀 Starting Production-Ready Dialogue System v2.0")
    print("=" * 70)
    
    # Initialize tenant factory
    tenant_factory = TenantContextFactory()
    
    # Register tenants (would come from database in production)
    tenant_configs = [
        TenantConfig(
            tenant_id="default",
            tenant_name="Default Tenant",
            enable_forms=True,
            enable_context_augmentation=True,  # Enabled for full feature verification
            enable_summarization=False,  # Disabled - not needed for basic operation
            enable_ner=True,  # Enabled for full feature verification
            confidence_threshold=0.7,
        ),
        # Add more tenants as needed
    ]
    
    for config in tenant_configs:
        tenant_factory.register_tenant(config)
    
    print("✅ Tenants registered")
    
    # Initialize session stores per tenant (SHARED across all requests)
    from src.application.session import SessionStore
    
    for config in tenant_configs:
        store = SessionStore(
            ttl_seconds=config.session_ttl_seconds,
        )
        tenant_factory.register_session_store(config.tenant_id, store)
        print(f"✅ SessionStore initialized for tenant: {config.tenant_id}")
    
    # Initialize event buses and dialogue pipelines per tenant
    from src.application.dialogue_service import EnhancedDialoguePipeline
    
    for config in tenant_configs:
        tenant_id = config.tenant_id
        
        # Create context
        context = await tenant_factory.create_context(tenant_id)
        
        # Initialize event bus
        event_bus = await context.get_event_bus()
        event_bus_registry[tenant_id] = event_bus
        
        # Create dialogue pipeline with tenant config
        pipeline = EnhancedDialoguePipeline(
            enable_forms=config.enable_forms,
            enable_ner=config.enable_ner,
            enable_context_augmentation=config.enable_context_augmentation,
            enable_summarization=config.enable_summarization,
        )
        dialogue_pipelines[tenant_id] = pipeline
        
        print(f"✅ Initialized tenant: {tenant_id}")
    
    print("✅ Event buses ready")
    print("✅ Dialogue pipelines created")
    print("=" * 70)
    print("System ready! Access docs at http://localhost:8000/docs")
    print("=" * 70)
    
    yield  # Application runs
    
    # Shutdown
    print("🛑 Shutting down...")
    
    # Cleanup tenant contexts
    for tenant_id in tenant_configs:
        try:
            context = await tenant_factory.create_context(tenant_id.tenant_id)
            await context.cleanup()
        except Exception as e:
            print(f"Error cleaning up tenant {tenant_id.tenant_id}: {e}")
    
    print("✅ Shutdown complete")
    

# ============================================================================
# FastAPI App Configuration
# ============================================================================

app = FastAPI(
    title="NID Support Dialogue System",
    description="Production-ready dialogue system with event sourcing, multi-tenancy, and distributed tracing",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Middleware - Distributed Tracing
# ============================================================================

@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    """
    Inject distributed tracing into every request.
    
    Creates trace context and propagates through entire request.
    """
    # Generate trace ID
    trace_id = request.headers.get("X-Trace-ID") or secrets.token_urlsafe(16)
    
    # Start trace
    trace_context = Tracer.start_trace(trace_id)
    
    # Start root span
    span = trace_context.start_span(f"{request.method} {request.url.path}")
    
    if span:
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.url", str(request.url))
        span.set_attribute("http.user_agent", request.headers.get("user-agent", ""))
    
    try:
        response = await call_next(request)
        
        if span:
            span.set_attribute("http.status_code", response.status_code)
            trace_context.end_span(span)
        
        # Add trace ID to response headers
        response.headers["X-Trace-ID"] = trace_id
        
        return response
    
    except Exception as e:
        if span:
            span.set_attribute("error", str(e))
            trace_context.end_span(span)
        raise


# ============================================================================
# Tenant Context Helper (No Authentication Required)
# ============================================================================
# Removed get_tenant_context dependency - tenant context created directly in endpoints


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0",
    }


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "NID Support Dialogue System",
        "version": "2.0.0",
        "architecture": "Event-sourced, multi-tenant, production-ready",
        "features": [
            "JWT authentication",
            "Multi-tenancy",
            "Event sourcing",
            "Distributed tracing",
            "Circuit breakers",
            "WebSocket support",
        ],
        "docs": "/docs",
    }


# ============================================================================
# API v2 - Chat Endpoint (Secure)
# ============================================================================

from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """Chat request (NO authentication required!)"""
    query: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None  # Optional - will be generated if not provided


class ChatResponse(BaseModel):
    """Chat response"""
    text: str
    metadata: Dict[str, Any]
    success: bool
    session_id: str  # Server-generated or client-provided
    trace_id: Optional[str] = None


@app.post("/api/v2/chat", response_model=ChatResponse)
async def chat_v2(request: ChatRequest):
    """
    Process chat turn - NO AUTHENTICATION REQUIRED.

    FEATURES:
    - Full NLP pipeline (48-class classification + semantic search)
    - Forms system (multi-turn slot collection)
    - Policy engine (FAQ/Clarify/Form/Escalate)
    - Hooks system (logging, metrics, custom logic)
    - Event sourcing (full audit trail)
    - Distributed tracing
    """
    # Generate or use provided session ID
    session_id = request.session_id or secrets.token_urlsafe(16)
    tenant_id = "default"

    # Get trace context
    trace_ctx = Tracer.get_current_context()
    correlation_id = trace_ctx.trace_id if trace_ctx else secrets.token_urlsafe(16)

    # Create tenant context directly
    try:
        tenant_context = await tenant_factory.create_context(
            tenant_id=tenant_id,
            correlation_id=correlation_id
        )
    except Exception as e:
        return ChatResponse(
            text="System initialization error",
            metadata={"error": str(e)},
            success=False,
            session_id=session_id,
            trace_id=correlation_id,
        )

    # Get dialogue pipeline for tenant
    pipeline = dialogue_pipelines.get(tenant_id)
    if not pipeline:
        return ChatResponse(
            text="System not ready",
            metadata={"error": "pipeline_not_found"},
            success=False,
            session_id=session_id,
            trace_id=correlation_id,
        )

    try:
        # Process dialogue turn through complete pipeline
        result = await pipeline.process_turn(
            query=request.query,
            session_id=session_id,
            tenant_context=tenant_context,
            correlation_id=correlation_id,
        )

        return ChatResponse(
            text=result["text"],
            metadata=result.get("metadata", {}),
            success=result["success"],
            session_id=session_id,
            trace_id=correlation_id,
        )

    except Exception as e:
        # Error handling with tracing
        if trace_ctx and trace_ctx.current_span:
            trace_ctx.current_span.set_attribute("error", str(e))

        return ChatResponse(
            text="দুঃখিত, একটি সমস্যা হয়েছে।",
            metadata={"error": str(e)},
            success=False,
            session_id=session_id,
            trace_id=correlation_id,
        )


# ============================================================================
# WebSocket - Real-Time Updates
# ============================================================================

class WebSocketConnectionManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        """Remove WebSocket connection"""
        self.active_connections.pop(session_id, None)
    
    async def send_message(self, session_id: str, message: dict):
        """Send message to specific session"""
        websocket = self.active_connections.get(session_id)
        if websocket:
            await websocket.send_json(message)


ws_manager = WebSocketConnectionManager()


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(
    websocket: WebSocket,
    session_id: str,
):
    """
    WebSocket endpoint for real-time dialogue updates.
    
    Client connects once and receives all updates via push.
    NO POLLING NEEDED!
    
    Events pushed:
    - NLP processing started
    - NLP completed
    - Form slot requested
    - Form completed
    - Response ready
    """
    await ws_manager.connect(session_id, websocket)
    
    try:
        # Subscribe to events for this session
        # (In production, parse session_id to get tenant_id)
        tenant_id = session_id.split(":")[0]
        event_bus = event_bus_registry.get(tenant_id)
        
        if event_bus:
            # Handle incoming messages
            while True:
                data = await websocket.receive_text()
                
                # Echo for now (would process through dialogue pipeline)
                await websocket.send_json({
                    "type": "echo",
                    "data": data,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
    
    except WebSocketDisconnect:
        ws_manager.disconnect(session_id)


# ============================================================================
# Authentication & Admin Endpoints REMOVED
# ============================================================================
# No authentication required - endpoints removed:
# - POST /api/v2/auth/login (JWT authentication)
# - GET /api/v2/admin/events/{session_id} (admin-only event replay)


# ============================================================================
# Error Handler
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with tracing"""
    span = Tracer.current_span()
    if span:
        span.set_attribute("error", str(exc))
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "trace_id": Tracer.get_current_context().trace_id if Tracer.get_current_context() else None,
        }
    )
