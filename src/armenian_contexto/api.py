import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from typing import Callable

from fastapi import FastAPI, HTTPException, Query, Request

from .contexto_engine import ArmenianContextoEngine


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

    return app


async def run_engine_call(request: Request, function, *args):
    """Run a blocking NumPy-heavy engine call in the API thread pool."""

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            request.app.state.executor,
            partial(function, *args),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


app = create_app()
