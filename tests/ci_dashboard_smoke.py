"""Smoke-проверка витрины для CI: сервер жив, а без данных страница показывает
понятное сообщение, а не трейсбек (CLAUDE.md, 0b, пункт 4б; 5p, «Проверка на GitHub»).

    python tests/ci_dashboard_smoke.py http://localhost:8501

Код 0 — всё в порядке, 1 — нет; причина печатается. Данных и секретов не требует.
"""
import asyncio
import sys
import time
import urllib.request

import choreographer as choreo

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
EXPECT = "Нет данных для витрины"
FORBID = ("Traceback", "Error:", "Exception")


def wait_health(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(URL + "/_stcore/health", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit(f"ПРОВАЛ: /_stcore/health не ответил за {timeout} с")


async def page_text(timeout=120):
    async with choreo.Browser(headless=True) as b:
        tab = await b.create_tab(URL)
        t0, text, exc = time.time(), "", ""
        while time.time() - t0 < timeout:
            await asyncio.sleep(3)
            r = await tab.send_command("Runtime.evaluate", params={"expression": "document.body.innerText"})
            text = r["result"]["result"].get("value", "")
            r = await tab.send_command("Runtime.evaluate", params={
                "expression": "(document.querySelector('[data-testid=stException]')||{}).innerText||''"})
            exc = r["result"]["result"].get("value", "")
            if EXPECT in text or exc:
                break
        return text, exc


def main():
    print(f"health: ответ через {wait_health():.0f} с")
    text, exc = asyncio.run(page_text())
    bad = [w for w in FORBID if w in text]
    if exc or bad:
        print("ПРОВАЛ: на странице ошибка вместо сообщения:", (exc or text)[:500])
        return 1
    if EXPECT not in text:
        print("ПРОВАЛ: нет сообщения «" + EXPECT + "». Текст страницы:", text[:500])
        return 1
    print("ОК: без данных витрина показывает сообщение, трейсбека нет")
    return 0


if __name__ == "__main__":
    sys.exit(main())
