"""
Second Chance — API server
Serves the static experience and exposes the simulation engine.
Run:  uvicorn app:app --reload
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import os

from simulate import SecondChance, OUTCOME_NAMES

app = FastAPI(title="Second Chance")
engine = SecondChance()

BASE = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")


class SimRequest(BaseModel):
    cpra: float = 97.0
    epts: float = 0.62
    seq: float = 180.0
    offer_kdpi: float = 0.43
    n: int = 8000


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(BASE, "templates", "index.html")) as f:
        return f.read()


@app.get("/api/params")
def params():
    return engine.p


@app.post("/api/simulate")
def simulate(req: SimRequest):
    patient = {"cpra": req.cpra, "epts": req.epts, "seq": req.seq}
    res = engine.run(patient, req.offer_kdpi, n=req.n)
    res["thompson"] = engine.thompson_recommendation(res)
    res["outcome_names"] = OUTCOME_NAMES
    return JSONResponse(res)


@app.post("/api/information")
def information(req: SimRequest):
    patient = {"cpra": req.cpra, "epts": req.epts, "seq": req.seq}
    return engine.information_gain(patient, req.offer_kdpi, n=req.n)
