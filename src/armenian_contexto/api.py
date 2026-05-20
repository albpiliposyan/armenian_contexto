import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from typing import Callable

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from .contexto_engine import ArmenianContextoEngine


class BatchGuessRequest(BaseModel):
    """Payload for scoring several guesses against one target word."""

    target: str = Field(min_length=1)
    guesses: list[str] = Field(min_length=1, max_length=100)


def create_app(
    engine: ArmenianContextoEngine | None = None,
    engine_factory: Callable[[], ArmenianContextoEngine] = ArmenianContextoEngine,
    max_workers: int = 4,
) -> FastAPI:
    """Create the FastAPI application with one startup-loaded engine instance.

    The expensive data artifacts are NumPy matrices stored in `.npz` files. The
    API loads them once during startup and reuses that engine for every request
    instead of re-reading matrices per endpoint call.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine if engine is not None else engine_factory()
        app.state.executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            yield
        finally:
            app.state.executor.shutdown(wait=True)

    app = FastAPI(
        title="Armenian Contexto API",
        description="Hybrid semantic ranking API for Armenian Contexto.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/api/health")
    async def health(request: Request):
        """Return service status and loaded vocabulary size."""

        engine = request.app.state.engine
        return {
            "status": "ok",
            "vocabulary_size": len(engine.words),
            "engine_loaded_once": True,
        }

    @app.get("/api/guess")
    async def guess(
        request: Request,
        target: str = Query(..., min_length=1),
        guess_word: str = Query(..., alias="guess", min_length=1),
    ):
        """Score one guess against a target using the hybrid ranking engine."""

        engine = request.app.state.engine
        return await run_engine_call(request, engine.get_rank, guess_word, target)

    @app.get("/api/closest")
    async def closest(
        request: Request,
        target: str = Query(..., min_length=1),
        top_k: int = Query(20, ge=1, le=100),
    ):
        """Return the top-k closest vocabulary words for a target."""

        engine = request.app.state.engine
        words = await run_engine_call(request, engine.closest_words, target, top_k)
        return {
            "target": target,
            "top_k": top_k,
            "closest": words,
        }

    @app.post("/api/batch-guess")
    async def batch_guess(request: Request, payload: BatchGuessRequest):
        """Score multiple guesses concurrently while keeping the API loop free.

        This project is CPU-light per request and NumPy-heavy inside the engine.
        Running calls in a thread pool keeps FastAPI's event loop responsive,
        and NumPy can release the GIL during vector operations. Multiprocessing
        would duplicate the loaded matrices in each process, which is a poor
        tradeoff for this workload.
        """

        engine = request.app.state.engine
        tasks = [
            run_engine_call(request, engine.get_rank, guess_word, payload.target)
            for guess_word in payload.guesses
        ]
        results = await asyncio.gather(*tasks)
        return {
            "target": payload.target,
            "count": len(results),
            "results": results,
        }

    return app


async def run_engine_call(request: Request, function, *args):
    """Run a blocking engine call in the API thread pool."""

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            request.app.state.executor,
            partial(function, *args),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


app = create_app()
