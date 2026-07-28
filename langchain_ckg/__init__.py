from langchain_ckg.retriever import CKGRetriever, CKGHostedRetriever, PolarUsageCallback

__all__ = ["CKGRetriever", "CKGHostedRetriever", "PolarUsageCallback"]
__version__ = "0.3.1"

import os as _os, sys as _sys, threading as _threading


def _beacon() -> None:
    if _os.environ.get("CKG_NO_TELEMETRY"):
        return
    def _send():
        try:
            import urllib.request as _req, json as _json
            payload = _json.dumps({
                "event": "ckg_import",
                "package": "langchain-ckg",
                "version": __version__,
                "python": f"{_sys.version_info.major}.{_sys.version_info.minor}",
                "platform": _sys.platform,
            }).encode()
            _req.urlopen(
                _req.Request(
                    "https://ckg-mcp.onrender.com/beacon",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=2,
            )
        except Exception:
            pass
    _threading.Thread(target=_send, daemon=True).start()


_beacon()
