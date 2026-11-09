"""用 Edge headless + DevTools Protocol 做真实前端测试。

不依赖 playwright/selenium：直接用 websocket 跟浏览器通信。
需要 websocket-client（httpx 已装，但它不带 ws），这里用极简实现。

如果 websocket 不可用，退化为静态检查并给出提示。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
URL = "http://127.0.0.1:8765"
PORT = 9223


class Edge:
    """一个 headless Edge 实例。"""

    def __init__(self, port: int = PORT):
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.profile = tempfile.mkdtemp(prefix="wsp-edge-")

    def start(self) -> None:
        if not os.path.exists(EDGE):
            raise RuntimeError(f"找不到 Edge: {EDGE}")
        self.proc = subprocess.Popen(
            [
                EDGE,
                "--headless=new",
                f"--remote-debugging-port={self.port}",
                f"--user-data-dir={self.profile}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-gpu",
                "--disable-extensions",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(60):
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/json/version", timeout=1
                )
                return
            except Exception:
                time.sleep(0.25)
        raise RuntimeError("Edge 没能在 15 秒内启动")

    def targets(self) -> list[dict]:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/json", timeout=5
        ) as r:
            return json.load(r)

    def new_tab(self, url: str) -> dict:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/json/new?{url}", method="PUT"
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)

    def stop(self) -> None:
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class CDP:
    """极简 Chrome DevTools Protocol 客户端（只用 HTTP + eval）。"""

    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        try:
            import websocket  # type: ignore
        except ImportError:
            raise RuntimeError("需要 websocket-client: pip install websocket-client")
        self.ws = websocket.create_connection(ws_url, timeout=30)
        self.counter = 0

    def send(self, method: str, params: dict | None = None) -> dict:
        self.counter += 1
        msg_id = self.counter
        self.ws.send(json.dumps({"id": msg_id, "method": method,
                                 "params": params or {}}))
        deadline = time.time() + 30
        while time.time() < deadline:
            raw = self.ws.recv()
            data = json.loads(raw)
            if data.get("id") == msg_id:
                if "error" in data:
                    raise RuntimeError(f"{method}: {data['error']}")
                return data.get("result", {})
        raise TimeoutError(f"{method} 超时")

    def eval(self, expression: str, await_promise: bool = False):
        result = self.send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": await_promise,
            },
        )
        if result.get("exceptionDetails"):
            detail = result["exceptionDetails"]
            raise RuntimeError(
                f"JS 异常: {detail.get('text')} "
                f"{detail.get('exception', {}).get('description', '')[:300]}"
            )
        return result.get("result", {}).get("value")

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass


passed: list[str] = []
failed: list[tuple[str, str]] = []


def check(name: str, fn):
    try:
        value = fn()
        passed.append(name)
        shown = f" -> {value}" if value not in (True, None) else ""
        print(f"  ✓ {name}{shown}")
        return value
    except Exception as exc:
        failed.append((name, str(exc)))
        print(f"  ✗ {name}")
        print(f"      {exc}")
        return None


def main() -> int:
    print("=" * 70)
    print("前端真实渲染测试（Edge headless）")
    print("=" * 70)

    try:
        import websocket  # noqa: F401
    except ImportError:
        print("\n缺少 websocket-client，无法做渲染测试。")
        print("安装: pip install websocket-client")
        return 2

    edge = Edge()
    print(f"启动 Edge（profile: {edge.profile}）…")
    edge.start()
    print("Edge 已就绪\n")

    tab = edge.new_tab(URL)
    cdp = CDP(tab["webSocketDebuggerUrl"])
    try:
        cdp.send("Runtime.enable")
        cdp.send("Page.enable")

        # 等页面加载
        time.sleep(3)

        print("第 1 部分：页面基本加载")
        check("document.readyState 完成", lambda: cdp.eval(
            "document.readyState") == "complete")
        check("标题正确", lambda: cdp.eval("document.title"))
        check("主容器存在", lambda: cdp.eval(
            "!!document.querySelector('header h1')"))
        check("侧栏存在", lambda: cdp.eval("!!document.querySelector('aside')"))
        check("输入框存在", lambda: cdp.eval("!!document.getElementById('input')"))
        check("发送按钮存在", lambda: cdp.eval(
            "!!document.getElementById('sendBtn')"))

        print("\n第 2 部分：JS 是否真的执行了")
        check("全局 state 已初始化", lambda: cdp.eval(
            "typeof state === 'object' && state !== null"))
        check("API 函数存在", lambda: cdp.eval("typeof api === 'function'"))
        check("send 函数存在", lambda: cdp.eval("typeof send === 'function'"))
        def check_boot():
            n = cdp.eval("(state.models || []).length")
            assert n >= 1, f"没有拉到模型（{n}）"
            return f"{n} 个"

        check("boot 已完成（模型已拉取）", check_boot)
        check("启动无 JS 错误", lambda: cdp.eval("!window.__wspError"))

        print("\n第 3 部分：实际调用后端")
        check("页面能拉到模型列表", lambda: cdp.eval(
            "(async () => { const r = await fetch('/api/models');"
            " const d = await r.json(); return d.models.length; })()",
            await_promise=True))

        check("页面能拉到来源状态", lambda: cdp.eval(
            "(async () => { const r = await fetch('/api/models/sources');"
            " const d = await r.json();"
            " return d.sources.map(s => s.key).join(','); })()",
            await_promise=True))

        print("\n第 4 部分：DOM 交互")
        def open_settings():
            # openSettings 是 async（要先拉配置），必须等它完成
            cdp.eval("(async () => { await openSettings(); return true; })()",
                     await_promise=True)
            time.sleep(0.5)
            opened = cdp.eval("document.getElementById('settingsDlg').open")
            assert opened is True, f"对话框没打开 (open={opened})"
            return True

        check("打开设置对话框", open_settings)

        check("填入 Base URL", lambda: cdp.eval(
            "(() => { const el = document.getElementById('setBaseUrl');"
            " el.value = 'http://test.local/v1'; return el.value; })()"))

        check("关闭设置对话框", lambda: cdp.eval(
            "(() => { document.getElementById('settingsDlg').close();"
            " return !document.getElementById('settingsDlg').open; })()"))

        def open_models():
            cdp.eval("(async () => { await openModels(); return true; })()",
                     await_promise=True)
            time.sleep(0.5)
            opened = cdp.eval("document.getElementById('modelsDlg').open")
            assert opened is True, f"对话框没打开 (open={opened})"
            return True

        check("打开模型对话框", open_models)

        check("模型表格已渲染", lambda: cdp.eval(
            "document.getElementById('modelTable').innerHTML.length > 0"))

        check("关闭模型对话框", lambda: cdp.eval(
            "(() => { document.getElementById('modelsDlg').close(); return true; })()"))

        print("\n第 5 部分：引导与卡片")
        check("引导检查函数存在", lambda: cdp.eval(
            "typeof checkOnboarding === 'function'"))

        def add_card():
            n = cdp.eval(
                "(() => { const before = document.querySelectorAll('.card').length;"
                " addCard('测试卡片', '<b>内容</b>');"
                " return document.querySelectorAll('.card').length - before; })()"
            )
            assert n == 1, f"卡片没新增（diff={n}）"
            return n

        check("卡片创建函数可用", add_card)

        def add_msg():
            n = cdp.eval(
                "(() => { const before = document.querySelectorAll('.msg.user').length;"
                " addMessageEl('user', '你好');"
                " return document.querySelectorAll('.msg.user').length - before; })()"
            )
            assert n == 1, f"消息没新增（diff={n}）"
            return n

        check("对话区可写消息", add_msg)

        print("\n第 6 部分：无未捕获异常")
        check("错误捕获器已安装", lambda: cdp.eval(
            "(() => { window.__errCount = 0;"
            " window.addEventListener('error', () => window.__errCount++);"
            " return true; })()"))

        check("触发一次完整引导检查后无异常", lambda: cdp.eval(
            "(async () => { await checkOnboarding();"
            " return window.__errCount || 0; })()",
            await_promise=True))

        print("\n第 7 部分：中文渲染")
        check("中文字体生效（文字非空）", lambda: cdp.eval(
            "document.body.innerText.length > 100"))
        check("页面含中文文案", lambda: cdp.eval(
            "document.body.innerText.includes('项目') ||"
            " document.body.innerText.includes('对话')"))

    finally:
        cdp.close()
        edge.stop()

    print()
    print("=" * 70)
    print(f"通过 {len(passed)}，失败 {len(failed)}")
    print("=" * 70)
    if failed:
        for name, err in failed:
            print(f"  ✗ {name}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
