from __future__ import annotations

import http.server
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.write_disk,
    pytest.mark.skipif(
        sys.platform != "win32",
        reason="temporary diagnostic for Windows access violations",
    ),
]

_CASES = (
    "auto-closed",
    "storage-closed",
    "auto-404",
    "storage-404",
)

_RETRY_PROFILES = {
    "old": {
        "POLARS_CLOUD_MAX_RETRIES": "2",
        "POLARS_CLOUD_RETRY_INIT_BACKOFF_MS": "100",
        "POLARS_CLOUD_RETRY_MAX_BACKOFF_MS": "15000",
        "POLARS_CLOUD_RETRY_TIMEOUT_MS": "10000",
    },
    "current": {
        "POLARS_CLOUD_MAX_RETRIES": "8",
        "POLARS_CLOUD_RETRY_INIT_BACKOFF_MS": "250",
        "POLARS_CLOUD_RETRY_MAX_BACKOFF_MS": "5000",
        "POLARS_CLOUD_RETRY_TIMEOUT_MS": "30000",
    },
}


class _NotFoundHandler(http.server.BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:
        self.server.request_count += 1  # type: ignore[attr-defined]
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


def _closed_endpoint() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def _start_server() -> tuple[http.server.ThreadingHTTPServer, threading.Thread, str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _NotFoundHandler)
    server.request_count = 0  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def _write_aws_config(root: Path, endpoint: str) -> None:
    config = root / "config"
    credentials = root / "credentials"
    config.write_text(f"[default]\nendpoint_url = {endpoint}\n", encoding="utf-8")
    credentials.write_text(
        "[default]\naws_access_key_id=Z\naws_secret_access_key=Z\n",
        encoding="utf-8",
    )
    os.environ["AWS_CONFIG_FILE"] = str(config)
    os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(credentials)
    os.environ["AWS_REGION"] = "us-east-1"
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"


def _run_worker(case: str) -> int:
    import polars as pl

    server: http.server.ThreadingHTTPServer | None = None
    server_thread: threading.Thread | None = None
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="polars-cloud-diagnostic-") as tmp:
        if case.endswith("-404"):
            server, server_thread, endpoint = _start_server()
        else:
            endpoint = _closed_endpoint()

        try:
            if case.startswith("auto-"):
                _write_aws_config(Path(tmp), endpoint)
                query = pl.scan_parquet("s3://bucket/path")
            else:
                query = pl.scan_parquet(
                    "s3://bucket/path",
                    storage_options={
                        "aws_access_key_id": "Z",
                        "aws_secret_access_key": "Z",
                        "aws_endpoint_url": endpoint,
                        "aws_region": "us-east-1",
                    },
                    credential_provider=None,
                )

            try:
                query.collect()
            except Exception as exc:
                result = {
                    "case": case,
                    "endpoint": endpoint,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "exception": type(exc).__name__,
                    "message": str(exc),
                    "server_request_count": (
                        0 if server is None else server.request_count  # type: ignore[attr-defined]
                    ),
                }
                print(json.dumps(result), flush=True)
                if case.endswith("-404") and result["server_request_count"] == 0:
                    return 3
                return 0

            print(json.dumps({"case": case, "error": "collect unexpectedly succeeded"}))
            return 2
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()
            if server_thread is not None:
                server_thread.join(timeout=5)


def _invoke_worker(case: str, retry_profile: str) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONFAULTHANDLER": "1",
            "RUST_BACKTRACE": "full",
            "POLARS_VERBOSE": "1",
            "POLARS_TIMEOUT_MS": "60000",
        }
    )
    env.update(_RETRY_PROFILES[retry_profile])
    command = [sys.executable, "-X", "faulthandler", __file__, "--worker", case]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=75,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "case": case,
            "retry_profile": retry_profile,
            "timeout": True,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "stdout": (exc.stdout or "").strip(),
            "stderr": (exc.stderr or "").strip(),
        }

    returncode = completed.returncode
    return {
        "case": case,
        "retry_profile": retry_profile,
        "auto_streaming": os.environ.get("POLARS_AUTO_STREAMING") == "1",
        "returncode": returncode,
        "returncode_hex": f"0x{returncode & 0xFFFFFFFF:08X}",
        "access_violation": (returncode & 0xFFFFFFFF) == 0xC0000005,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _record_github_summary(result: dict[str, Any]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path is None:
        return

    row = (
        f"- case=`{result['case']}`; profile=`{result['retry_profile']}`; "
        f"repeat={result['repeat']}; auto-streaming="
        f"{result.get('auto_streaming')}; return="
        f"`{result.get('returncode_hex', 'timeout')}`; access-violation="
        f"{result.get('access_violation', False)}; "
        f"elapsed={result['elapsed_seconds']}s\n"
    )
    with Path(summary_path).open("a", encoding="utf-8") as summary:
        summary.write(row)


@pytest.mark.parametrize("repeat", range(4))
@pytest.mark.parametrize("retry_profile", _RETRY_PROFILES)
@pytest.mark.parametrize("case", _CASES)
def test_windows_cloud_access_violation_diagnostic(
    case: str,
    retry_profile: str,
    repeat: int,
) -> None:
    result = _invoke_worker(case, retry_profile)
    result["repeat"] = repeat
    print(json.dumps(result, indent=2), flush=True)
    _record_github_summary(result)

    assert not result.get("timeout"), result
    assert not result.get("access_violation"), result
    assert result.get("returncode") == 0, result


if __name__ == "__main__":
    worker_index = sys.argv.index("--worker")
    raise SystemExit(_run_worker(sys.argv[worker_index + 1]))
