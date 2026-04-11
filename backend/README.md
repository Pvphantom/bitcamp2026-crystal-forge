# Backend

This service will own:

- Hubbard-model physics
- exact diagonalization and observables
- ML inference
- stable export data for both the web frontend and Minecraft integration

## Immediate Scope

The current scaffold provides:

- FastAPI app entrypoint
- route registration
- Pydantic request/response models
- placeholder `HubbardGameStateService`

The next milestone is implementing the Hamiltonian and the 4 acceptance tests from the spec before adding richer endpoints.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload
```
