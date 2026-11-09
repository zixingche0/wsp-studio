"""API Key 加密存储。

你要求在页面直接输入并保存，所以不能只依赖环境变量。按平台选择最强方案：

1. **Windows**：DPAPI（``CryptProtectData``，绑定当前用户账户）
2. **macOS**：Keychain（通过 ``security`` 命令）
3. **Linux**：Secret Service（通过 ``secret-tool``，即 GNOME Keyring / KWallet）
4. **回退**：本机随机密钥 + HMAC-SHA256 流式加密

任何情况下：**明文只在内存里存在**，UI 只显示掩码。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets as _secrets
import shutil
import subprocess
import sys
from pathlib import Path


class SecretsError(Exception):
    """加解密失败。"""


SECRET_SERVICE = "wsp-studio"
SECRET_ACCOUNT = "llm-api-key"


# ---------------------------------------------------------------------------
# Windows DPAPI
# ---------------------------------------------------------------------------

def _dpapi_available() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes  # noqa: F401
        import ctypes.wintypes  # noqa: F401
        return hasattr(ctypes.windll, "crypt32")
    except Exception:
        return False


def _dpapi_encrypt(plaintext: bytes) -> bytes:
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    def to_blob(data: bytes) -> DATA_BLOB:
        buf = ctypes.create_string_buffer(data, len(data))
        return DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))

    blob_in = to_blob(plaintext)
    blob_out = DATA_BLOB()
    ok = ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    )
    if not ok:
        raise SecretsError("CryptProtectData 失败")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def _dpapi_decrypt(ciphertext: bytes) -> bytes:
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    def to_blob(data: bytes) -> DATA_BLOB:
        buf = ctypes.create_string_buffer(data, len(data))
        return DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))

    blob_in = to_blob(ciphertext)
    blob_out = DATA_BLOB()
    ok = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    )
    if not ok:
        raise SecretsError("CryptUnprotectData 失败（密钥不属于当前用户或系统）")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


# ---------------------------------------------------------------------------
# macOS Keychain
# ---------------------------------------------------------------------------

def _os_name() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _keychain_available() -> bool:
    return sys.platform == "darwin" and bool(shutil.which("security"))


def _keychain_store(plaintext: str) -> None:
    subprocess.run(
        [
            "security", "add-generic-password",
            "-s", SECRET_SERVICE, "-a", SECRET_ACCOUNT,
            "-w", plaintext, "-U",
        ],
        check=True, capture_output=True, timeout=15,
    )


def _keychain_load() -> str:
    out = subprocess.run(
        [
            "security", "find-generic-password",
            "-s", SECRET_SERVICE, "-a", SECRET_ACCOUNT, "-w",
        ],
        capture_output=True, text=True, timeout=15,
    )
    if out.returncode != 0:
        raise SecretsError("Keychain 里没有找到保存的密钥")
    return out.stdout.strip()


# ---------------------------------------------------------------------------
# Linux Secret Service（GNOME Keyring / KWallet）
# ---------------------------------------------------------------------------

def _secret_service_available() -> bool:
    if _os_name() in ("windows", "macos"):
        return False
    return bool(shutil.which("secret-tool"))


def _secret_service_store(plaintext: str) -> None:
    subprocess.run(
        [
            "secret-tool", "store",
            "--label", "WSP Studio LLM API Key",
            "service", SECRET_SERVICE,
            "account", SECRET_ACCOUNT,
        ],
        input=plaintext.encode("utf-8"),
        check=True, capture_output=True, timeout=15,
    )


def _secret_service_load() -> str:
    out = subprocess.run(
        [
            "secret-tool", "lookup",
            "service", SECRET_SERVICE,
            "account", SECRET_ACCOUNT,
        ],
        capture_output=True, text=True, timeout=15,
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise SecretsError("Secret Service 里没有找到保存的密钥")
    return out.stdout.strip()


# ---------------------------------------------------------------------------
# 回退：本机密钥 + HMAC-SHA256 流式异或
# ---------------------------------------------------------------------------

def _fallback_key_path(home: Path) -> Path:
    return home / ".keyfile"


def _load_or_create_key(home: Path) -> bytes:
    path = _fallback_key_path(home)
    if path.exists():
        data = path.read_bytes()
        if len(data) >= 32:
            return data[:32]
    key = _secrets.token_bytes(32)
    path.write_bytes(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        block = hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:length])


def _fallback_encrypt(plaintext: bytes, key: bytes) -> bytes:
    nonce = _secrets.token_bytes(16)
    stream = _keystream(key, nonce, len(plaintext))
    body = bytes(a ^ b for a, b in zip(plaintext, stream))
    tag = hmac.new(key, nonce + body, hashlib.sha256).digest()[:16]
    return nonce + tag + body


def _fallback_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    if len(ciphertext) < 32:
        raise SecretsError("密文太短")
    nonce, tag, body = ciphertext[:16], ciphertext[16:32], ciphertext[32:]
    expect = hmac.new(key, nonce + body, hashlib.sha256).digest()[:16]
    if not hmac.compare_digest(tag, expect):
        raise SecretsError("校验失败：密文被篡改或密钥不匹配")
    stream = _keystream(key, nonce, len(body))
    return bytes(a ^ b for a, b in zip(body, stream))


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------

def encrypt_secret(plaintext: str, home: Path) -> str:
    """加密一个字符串，返回可直接存库的带方案前缀的字符串。"""
    if not plaintext:
        return ""

    # 1. Windows DPAPI
    if _dpapi_available():
        blob = _dpapi_encrypt(plaintext.encode("utf-8"))
        return "dpapi:" + base64.b64encode(blob).decode("ascii")

    # 2. macOS Keychain —— 明文交给系统保管，库里只存一个引用标记
    if _keychain_available():
        try:
            _keychain_store(plaintext)
            return "keychain:"
        except (subprocess.SubprocessError, OSError):
            pass

    # 3. Linux Secret Service
    if _secret_service_available():
        try:
            _secret_service_store(plaintext)
            return "secretservice:"
        except (subprocess.SubprocessError, OSError):
            pass

    # 4. 回退：本机密钥
    key = _load_or_create_key(home)
    blob = _fallback_encrypt(plaintext.encode("utf-8"), key)
    return "localkey:" + base64.b64encode(blob).decode("ascii")


def decrypt_secret(stored: str, home: Path) -> str:
    """解密。兼容环境变量兜底。"""
    if not stored:
        return ""
    if ":" not in stored:
        # 历史数据或明文（不应出现），按原样返回
        return stored
    scheme, _, payload = stored.partition(":")

    if scheme == "dpapi":
        if not _dpapi_available():
            raise SecretsError("此密文由 Windows DPAPI 加密，当前平台无法解密")
        return _dpapi_decrypt(base64.b64decode(payload)).decode("utf-8")

    if scheme == "keychain":
        return _keychain_load()

    if scheme == "secretservice":
        return _secret_service_load()

    if scheme == "localkey":
        try:
            blob = base64.b64decode(payload)
        except Exception as exc:
            raise SecretsError("密文不是合法 base64") from exc
        key = _load_or_create_key(home)
        return _fallback_decrypt(blob, key).decode("utf-8")

    raise SecretsError(f"未知的加密方案: {scheme}")


def mask_secret(value: str, visible: int = 4) -> str:
    """把密钥变成 ``sk-****abcd`` 形式，用于 UI 回显。"""
    if not value:
        return ""
    if len(value) <= visible:
        return "*" * len(value)
    return "*" * max(4, len(value) - visible) + value[-visible:]


def scheme_name() -> str:
    """当前平台会使用的加密方案名。"""
    if _dpapi_available():
        return "dpapi"
    if _keychain_available():
        return "keychain"
    if _secret_service_available():
        return "secretservice"
    return "localkey"


def scheme_description() -> str:
    """给 UI 的人话说明。"""
    mapping = {
        "dpapi": "Windows DPAPI（绑定当前用户，密文不可跨机器解密）",
        "keychain": "macOS Keychain（系统密钥链）",
        "secretservice": "Linux Secret Service（GNOME Keyring / KWallet）",
        "localkey": "本机密钥文件（较弱：密钥与密文在同一台机器上）",
    }
    return mapping.get(scheme_name(), scheme_name())
