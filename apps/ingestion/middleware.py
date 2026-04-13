
import hashlib
import hmac
import functools

from django.conf import settings
from django.http import JsonResponse


def validate_esp32_hmac(view_func):
    """
    Decorator: validates HMAC-SHA256 signature on ESP32 payloads.
    Returns 401 if header missing or signature invalid.
    """
    @functools.wraps(view_func)
    def wrapper(request, *args, **kwargs):
        received_sig = request.headers.get("X-Esp32-Signature", "")
        if not received_sig:
            return JsonResponse(
                {"error": "Missing X-ESP32-Signature header"},
                status=401,
            )

        raw_body = request.body  # safe to read multiple times in Django

        secret = settings.ESP32_HMAC_SECRET.encode("utf-8")
        expected = hmac.new(
            key=secret,
            msg=raw_body,
            digestmod=hashlib.sha256,
        ).hexdigest()

        # Constant-time comparison prevents timing attacks
        if not hmac.compare_digest(received_sig.lower(), expected):
            return JsonResponse({"error": "Invalid HMAC signature."}, status=401)

        return view_func(request, *args, **kwargs)
    return wrapper