"""
OpenSCAD MCP Server - stdio version
This module provides a stdio-based MCP server for OpenSCAD rendering.
"""
import contextlib
import contextvars
import datetime
import logging
import os
import re
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

import sentry_sdk
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Image as MCPImage
from PIL import Image as PILImage

# Import local utilities
from mcp_image_utils import to_mcp_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Initialize Sentry if DSN is provided
sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    logger.info(f"Initializing Sentry with DSN: {sentry_dsn[:20]}... (truncated for security)")
    sentry_sdk.init(
        dsn=sentry_dsn,
        enable_logs=True,
    )
    logger.info("Sentry initialized successfully")
else:
    logger.info("Sentry DSN not provided, running without Sentry")

MCP_TOKEN_CTX = contextvars.ContextVar("mcp_token", default=None)

# Initialize FastMCP for stdio
MCP_NAME = os.getenv("MCP_NAME", "openscad")
_safe_name = re.sub(r"[^a-z0-9_-]", "-", MCP_NAME.lower()).strip("-") or "service"
logger.info(f"Safe service name: {_safe_name}")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, value, default)
        return default


def _sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "-", name).strip("-.")
    return sanitized or "script"


# Storage layout and public asset configuration
DATA_DIR = Path(os.getenv("MCP_DATA_DIR", "./data")).resolve()
SCAD_DIR = DATA_DIR / "scad"
RENDER_DIR = DATA_DIR / "render"
STL_DIR = DATA_DIR / "stl"

for directory in (SCAD_DIR, RENDER_DIR, STL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Preview sizing (keep inline responses <~1MB)
PREVIEW_MAX_WIDTH = _env_int("MCP_PREVIEW_MAX_WIDTH", 800)
PREVIEW_MAX_HEIGHT = _env_int("MCP_PREVIEW_MAX_HEIGHT", 600)
PREVIEW_FORMAT = os.getenv("MCP_PREVIEW_FORMAT", "jpeg").lower()
if PREVIEW_FORMAT not in {"jpeg", "jpg", "png", "webp"}:
    logger.warning("Unsupported MCP_PREVIEW_FORMAT=%s; defaulting to jpeg", PREVIEW_FORMAT)
    PREVIEW_FORMAT = "jpeg"
PREVIEW_JPEG_QUALITY = _env_int("MCP_PREVIEW_JPEG_QUALITY", 85)

mcp = FastMCP(_safe_name, json_response=True)

# Concurrency guard to prevent CPU/memory overload on weak hosts
_max_concurrency = _env_int("RENDER_MAX_CONCURRENCY", _env_int("OPENSCAD_MAX_CONCURRENCY", 2))
_render_semaphore = threading.Semaphore(_max_concurrency)


@mcp.tool()
def render_scad_script(
    scad_code: str,
    view: str = "3d",
) -> list[Any]:
    """Render an OpenSCAD script and return a preview image with download link.

    Generates a full-resolution PNG via OpenSCAD and returns a JPEG/PNG preview
    with an HTTPS URL for downloading the full-resolution image. Always provide
    the Preview URL to users for downloading the rendered image.
    """

    try:
        with _render_semaphore:
            import shutil

            logger.info("render_scad_script invoked")

            # Auto-generate UID and filename
            render_uid = str(uuid.uuid4())
            filename = f"render_{render_uid[:8]}"

            # Log tool call details
            logger.info(f"render_scad_script: view={view}, render_uid={render_uid}, scad_length={len(scad_code)}")
            uid_scad_dir = SCAD_DIR / render_uid
            uid_render_dir = RENDER_DIR / render_uid

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".scad", delete=False
            ) as scad_file:
                temp_scad_path = Path(scad_file.name)
                scad_file.write(scad_code)
            temp_png_fd, temp_png_path_str = tempfile.mkstemp(suffix=".png")
            os.close(temp_png_fd)
            temp_png_path = Path(temp_png_path_str)

            safe_base = _sanitize_filename(filename)
            png_name = f"{safe_base}_{view}.png"
            permanent_scad_path = uid_scad_dir / f"{safe_base}.scad"
            permanent_png_path = uid_render_dir / png_name

            try:
                camera_settings = {
                    "top": ("0,0,100,0,0,0", "ortho"),
                    "front": ("0,-100,0,0,0,0", "ortho"),
                    "left": ("-100,0,0,0,0,0", "ortho"),
                    "3d": ("70,70,50,0,0,0", "perspective"),
                }

                if view not in camera_settings:
                    raise ValueError(
                        f"Invalid view '{view}'. Valid options: {list(camera_settings.keys())}"
                    )

                camera, projection = camera_settings[view]

                cmd = [
                    "openscad",
                    "-o",
                    str(temp_png_path),
                    "--autocenter",
                    "--viewall",
                    "--imgsize=800,600",
                    "--camera",
                    camera,
                    "--projection",
                    projection,
                    str(temp_scad_path),
                ]

                logger.info("Running OpenSCAD command: %s", " ".join(cmd))
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30
                )

                if result.returncode != 0:
                    raise RuntimeError(
                        f"OpenSCAD rendering failed: {result.stderr.strip()}"
                    )

                if not temp_png_path.exists():
                    raise RuntimeError(
                        "OpenSCAD rendering succeeded but no output file was created"
                    )

                # Always persist files
                uid_scad_dir.mkdir(parents=True, exist_ok=True)
                uid_render_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(temp_scad_path, permanent_scad_path)
                shutil.copy2(temp_png_path, permanent_png_path)
                logger.info(
                    "Persisted render UID=%s files at %s and %s",
                    render_uid,
                    permanent_scad_path,
                    permanent_png_path,
                )

                with PILImage.open(temp_png_path) as full_image:
                    preview_image = full_image.copy()
                    try:
                        resample = PILImage.Resampling.LANCZOS
                    except AttributeError:  # Pillow < 10 fallback
                        resample = PILImage.LANCZOS
                    preview_image.thumbnail(
                        (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), resample=resample
                    )

                preview_kwargs: dict[str, Any] = {}
                if PREVIEW_FORMAT in {"jpeg", "jpg"}:
                    preview_kwargs["quality"] = PREVIEW_JPEG_QUALITY
                    preview_kwargs["optimize"] = True
                elif PREVIEW_FORMAT in {"png", "webp"}:
                    preview_kwargs["optimize"] = True

                preview_block: MCPImage = to_mcp_image(
                    preview_image,
                    format=PREVIEW_FORMAT,
                    **preview_kwargs,
                )
                preview_content = preview_block.to_image_content()

                info_lines = [f"Render saved to {png_name}"]

                # Log successful render
                logger.info(f"render_scad_script successful: {view} view rendered, render_uid={render_uid}")

                return [preview_content, "\n".join(info_lines)]

            finally:
                if temp_scad_path.exists():
                    temp_scad_path.unlink()
                if temp_png_path.exists():
                    temp_png_path.unlink()

    except subprocess.TimeoutExpired as exc:
        logger.error(f"OpenSCAD rendering timed out: tool=render_scad_script, view={view}")
        raise RuntimeError("OpenSCAD rendering timed out (30 seconds)") from exc
    except Exception as exc:
        logger.error(f"Exception in render_scad_script: tool=render_scad_script, view={view}, error={exc}")
        raise RuntimeError(
            f"Exception occurred while rendering OpenSCAD script: {exc}"
        ) from exc


@mcp.tool()
def generate_stl(
    scad_code: str,
) -> list[Any]:
    """Generate an STL file from OpenSCAD code and provide download links.

    Takes OpenSCAD code and generates downloadable STL and SCAD files for 3D printing or CAD import.
    Returns HTTPS download URLs for both files. IMPORTANT: Always provide both STL and SCAD URLs to users so they
    can download the generated 3D model file for 3D printing or CAD software and the source code.
    """

    try:
        with _render_semaphore:
            import shutil

            logger.info("generate_stl invoked")

            # Auto-generate UID and filename
            stl_uid = str(uuid.uuid4())
            output_filename = f"model_{stl_uid[:8]}"
            uid_stl_dir = STL_DIR / stl_uid

            # Log tool call details
            logger.info(f"generate_stl: stl_uid={stl_uid}, output_filename={output_filename}, scad_length={len(scad_code)}")

            # Use provided scad_code directly
            scad_content = scad_code

            # Create temporary files
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".scad", delete=False
            ) as scad_file:
                temp_scad_path = Path(scad_file.name)
                scad_file.write(scad_content)

            temp_stl_fd, temp_stl_path_str = tempfile.mkstemp(suffix=".stl")
            os.close(temp_stl_fd)
            temp_stl_path = Path(temp_stl_path_str)

            safe_output_name = _sanitize_filename(output_filename)
            stl_filename = f"{safe_output_name}.stl"
            permanent_stl_path = uid_stl_dir / stl_filename

            try:
                # Generate STL using OpenSCAD
                cmd = [
                    "openscad",
                    "-o",
                    str(temp_stl_path),
                    str(temp_scad_path),
                ]

                logger.info("Running OpenSCAD STL command: %s", " ".join(cmd))
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=60
                )

                if result.returncode != 0:
                    raise RuntimeError(
                        f"OpenSCAD STL generation failed: {result.stderr.strip()}"
                    )

                if not temp_stl_path.exists():
                    raise RuntimeError(
                        "OpenSCAD STL generation succeeded but no output file was created"
                    )

                # Always persist STL and SCAD files
                uid_stl_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(temp_stl_path, permanent_stl_path)

                # Also save the SCAD file alongside the STL
                scad_filename = f"{safe_output_name}.scad"
                permanent_scad_path = uid_stl_dir / scad_filename
                shutil.copy2(temp_scad_path, permanent_scad_path)

                logger.info(
                    "Persisted STL UID=%s files at %s and %s",
                    stl_uid,
                    permanent_stl_path,
                    permanent_scad_path,
                )

                # Get file size for info
                stl_size = temp_stl_path.stat().st_size

                info_lines = [f"STL generated: {stl_filename} ({stl_size} bytes)", f"SCAD saved: {scad_filename}"]

                # Log successful STL generation
                logger.info(f"generate_stl successful: {stl_filename} and {scad_filename} generated, stl_uid={stl_uid}, size={stl_size} bytes")

                return ["\n".join(info_lines)]

            finally:
                if temp_scad_path.exists():
                    temp_scad_path.unlink()
                if temp_stl_path.exists():
                    temp_stl_path.unlink()

    except subprocess.TimeoutExpired as exc:
        logger.error(f"OpenSCAD STL generation timed out: tool=generate_stl, stl_uid={stl_uid}")
        raise RuntimeError("OpenSCAD STL generation timed out (60 seconds)") from exc
    except Exception as exc:
        logger.error(f"Exception in generate_stl: tool=generate_stl, stl_uid={stl_uid}, error={exc}")
        raise RuntimeError(
            f"Exception occurred while generating STL: {exc}"
        ) from exc


@mcp.resource(f"{_safe_name}://render/{{uid}}/{{name}}_{{view}}.png", mime_type="image/png")
def get_render_resource(uid: str, name: str, view: str) -> bytes:
    """Expose persisted render PNGs as MCP resources."""

    safe_name = _sanitize_filename(name)
    path = RENDER_DIR / uid / f"{safe_name}_{view}.png"
    if not path.exists():
        raise FileNotFoundError(
            f"Render {uid}/{safe_name}_{view}.png not found on server"
        )
    return path.read_bytes()


@mcp.resource(f"{_safe_name}://source/{{uid}}/{{name}}.scad", mime_type="text/plain")
def get_scad_resource(uid: str, name: str) -> str:
    """Expose persisted SCAD sources as MCP resources."""

    safe_name = _sanitize_filename(name)
    path = SCAD_DIR / uid / f"{safe_name}.scad"
    if not path.exists():
        raise FileNotFoundError(
            f"Source {uid}/{safe_name}.scad not found on server"
        )
    return path.read_text()


@mcp.resource(f"{_safe_name}://stl/{{uid}}/{{name}}.stl", mime_type="model/stl")
def get_stl_resource(uid: str, name: str) -> bytes:
    """Expose persisted STL files as MCP resources."""

    safe_name = _sanitize_filename(name)
    path = STL_DIR / uid / f"{safe_name}.stl"
    if not path.exists():
        raise FileNotFoundError(
            f"STL {uid}/{safe_name}.stl not found on server"
        )
    return path.read_bytes()


@mcp.resource(
    f"{_safe_name}://documentation",
    name="OpenSCAD Documentation",
    description="Documentation and guidance for using OpenSCAD scripting language",
    mime_type="text/markdown"
)
def get_documentation_resource() -> str:
    """Expose OpenSCAD documentation as an MCP resource."""

    doc_path = Path(__file__).parent / "openscad_documentation.md"
    if not doc_path.exists():
        raise FileNotFoundError("OpenSCAD documentation not found on server")
    return doc_path.read_text()


def main():
    """Main entry point for stdio MCP server."""
    logger.info(f"Starting {MCP_NAME} MCP server (stdio)")
    mcp.run()


if __name__ == "__main__":
    main()