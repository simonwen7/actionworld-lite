"""Single FastAPI Vercel function: serves the website and actual physics API."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from . import engine, model_adapter, session

logger=logging.getLogger("actionworld.m8")
ROOT=Path(__file__).resolve().parents[1]
app=FastAPI(title="ActionWorld Lite — Real-Time Web Demo",docs_url=None,redoc_url=None,openapi_url=None)
MAX_BODY=64*1024


@app.middleware("http")
async def privacy_headers(request:Request,call_next):
    # No cookies. Do not cache signed tokens or image responses.
    result=await call_next(request)
    result.headers["Cache-Control"]="no-store" if request.url.path.startswith("/api/") else "no-cache"
    result.headers["X-Content-Type-Options"]="nosniff"
    result.headers["Referrer-Policy"]="no-referrer"
    result.headers["X-Frame-Options"]="DENY"
    result.headers["Content-Security-Policy"]=(
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; font-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )
    return result


@app.get("/",include_in_schema=False)
def home():
    return FileResponse(ROOT/"public/index.html",media_type="text/html")


@app.get("/m8-style.css",include_in_schema=False)
def style():
    return FileResponse(ROOT/"public/m8-style.css",media_type="text/css")


@app.get("/m8-app.js",include_in_schema=False)
def script():
    return FileResponse(ROOT/"public/m8-app.js",media_type="text/javascript")


@app.get("/api/health")
def health():
    # Health intentionally does not disclose local filesystem paths or load weights.
    return {"app":"ActionWorld Lite M8","mode":"real-inference-only","ready":bool(os.environ.get("SESSION_SECRET", ""))}


def secret():
    value=os.environ.get("SESSION_SECRET","")
    if len(value.encode("utf-8"))<32:
        raise HTTPException(503,"SESSION_SECRET not configured")
    return value


async def json_body(req:Request):
    # Read body incrementally to prevent unbounded buffering in app code.
    body=bytearray()
    async for part in req.stream():
        body.extend(part)
        if len(body)>MAX_BODY:
            raise HTTPException(413,"Request too large")
    try:
        data=json.loads(body)
    except (UnicodeDecodeError,json.JSONDecodeError) as exc:
        raise HTTPException(400,"Malformed JSON body") from exc
    if not isinstance(data,dict):
        raise HTTPException(400,"Expected JSON object")
    return data


def fail_closed(exc):
    if isinstance(exc,session.SessionError):
        raise HTTPException(401,str(exc)) from exc
    if isinstance(exc,ValueError):
        raise HTTPException(422,"Invalid physics or prediction input") from exc
    logger.exception("Frozen model or MuJoCo runtime failure")
    raise HTTPException(503,"Real model/physics runtime unavailable; no fallback") from exc


@app.post("/api/session")
async def reset(req:Request):
    await json_body(req)
    key=secret()
    try:
        output,state,previous,current=engine.reset_scene()
        token=session.new_payload(state,previous,current,model_adapter.FINGERPRINT)
        output["session_token"]=session.sign(token,key)
        return output
    except Exception as exc:
        fail_closed(exc)


@app.post("/api/action")
async def action(req:Request):
    body=await json_body(req)
    key=secret()
    if set(body)!={"session_token","step_size_mm","direction"}:
        raise HTTPException(400,"Unexpected action fields")
    mm=body.get("step_size_mm");direction=body.get("direction")
    if type(mm) is not int or mm not in (5,10,20) or direction not in ("up","down","left","right"):
        raise HTTPException(422,"Invalid direction or step size")
    try:
        previous=session.verify(body["session_token"],key)
        if previous["steps"]>=session.MAX_STEPS:
            raise HTTPException(409,"64-action session limit; Reset to continue")
        state,previous_frame,current_frame=session.extract(previous,model_adapter.FINGERPRINT,engine.nstate())
        output,new_state,p,c=engine.action_scene(state,previous_frame,current_frame,direction,mm)
        new=session.next_payload(previous,new_state,p,c)
        output["scored_actions"]=new["steps"]
        output["session_token"]=session.sign(new,key)
        return output
    except HTTPException:
        raise
    except Exception as exc:
        fail_closed(exc)
