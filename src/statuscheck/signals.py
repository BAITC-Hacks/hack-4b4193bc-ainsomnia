"""Триангуляция раздела 5o: сигналы по правилу, итоговая карта, сверка с прогнозом.

    .venv/bin/python -m src.statuscheck.signals

Правило (CLAUDE.md, 5o, до кода): сигнал — служба в худшей четверти не меньше
чем по 2 засчитанным каналам, и один из них — канал 3; сильный — 3 из 3; служба
с одним засчитанным каналом сигнала не получает. Рядом — ожидаемое число
случайных совпадений: шанс попасть в верхнюю четверть двух независимых каналов —
1 из 16. Если сигналов ноль, главный выход — итоговая карта: 21 клетка с тем,
что засчитано после проверок на шум, и причиной, если нет.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.statuscheck.frame import FRAME
from src.statuscheck.load import SPEC
from src.statuscheck.observability import SHORT

OUT = Path("reports/5o")
CH = ("1. ритм закрытий", "2. зависшие", "3. поток после закрытия")


def _read(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _almaty_open(fmap) -> str:
    obs = _read("observability.json")
    x = next(o for o in obs if o["region"] == "Алматинская область" and o["channel"] == CH[1])
    m = re.search(r"незакрытых старше 60 дней на [\d.]+: (\d+)", x["numbers"])
    return f"{m.group(1)} незакрытых старше 60 дней" if m else "—"


def final_map(obs, ch1, ch3) -> list[dict]:
    c1 = {r["region"]: r for r in ch1}
    c3 = {r["region"]: r for r in ch3}
    out = []
    for x in obs:
        region, ch, state, reason = x["region"], x["channel"], x["state"], x["reason"]
        if ch == CH[0] and region in c1 and state.startswith("по службам"):
            r = c1[region]
            sh, pl = r["check_b_split_half"], r["check_c_placebo"] or {}
            fails = []
            if r["check_a_noise"]:
                fails.append(f"синхронных пачек {r['check_a_sync_bursts']:.0%} — больше трети")
            if sh["noise"]:
                fails.append("ранги первой и второй половины периода не держатся: ρ "
                             + (f"{sh['rho']:.2f}, интервал {sh['ci'][0]:.2f}…{sh['ci'][1]:.2f}"
                                if sh["rho"] is not None else "не посчитана")
                             + f" на {sh['pairs']} парах служба × тема")
            if pl.get("noise", True):
                fails.append(f"при перемешивании меток служб разброс не уже фактического в "
                             f"{pl.get('not_narrower', '—')} из {pl.get('reps', '—')} случаев")
            state = "засчитан по службам" if r["counted"] else "не засчитан"
            reason = (f"прошёл проверки; служб в ранжировании {r['ranked_services']}, пар "
                      f"служба × тема {r['ranked_pairs']}, в худшей четверти {r['worst_pairs']}"
                      if r["counted"] else "; ".join(fails))
        if ch == CH[2] and region in c3:
            r = c3[region]
            if r["counted_services"]:
                state = "засчитан по службам"
            elif r["counted_region"]:
                state = "засчитан по теме региона, не по службам"
                reason = (f"нет темы с 5+ службами по 3+ «своих» всплеска (служб с 3+ — "
                          f"{r['services_3plus']})")
            else:
                ly = r["last_year"]
                why = []
                if r["shift"]["noise"]:
                    why.append("порядок всплесков не зависит от дат закрытия")
                if ly["noise"]:
                    why.append(f"проверка «те же даты год назад» невозможна — всплесков с прошлым "
                               f"годом {ly['spikes']}, нужно 10+" if not ly["checked"] else
                               f"хвосты не длиннее прошлогодних, p = {ly['p']:.3f}")
                state, reason = "не засчитан", "; ".join(why)
        out.append({"region": region, "channel": ch, "state": state, "reason": reason})
    return out


def signals(ch1, ch3) -> tuple[list[dict], int]:
    """Службы, где засчитаны 2+ канала, один из них — канал 3."""
    per_service = {}
    for r in ch1:
        if r["counted"]:
            for row in r["table"]:
                per_service.setdefault((r["region"], row["topic"], row["service"]), {})["1"] = row["worst"]
    if any(r["counted_services"] for r in ch3):
        # Отметка худшей четверти по каналу 3 для служб не реализована: на данных
        # 2026-09-27 канал 3 по службам не засчитан нигде. Молча выдать ноль
        # сигналов здесь было бы ошибкой.
        raise NotImplementedError("канал 3 засчитан по службам — нужна отметка худшей "
                                  "четверти по каналу 3 в signals.py")
    eligible = [k for k, v in per_service.items() if len(v) >= 2 and "3" in v]
    found = [{"region": k[0], "topic": k[1], "service": k[2], **per_service[k]}
             for k in eligible if sum(per_service[k].values()) >= 2]
    return found, len(eligible)


WHY_NOT = (("нет даты закрытия", ("нет факта закрытия", "нет даты закрытия")),
           ("нет поля исполнителя", ("нет поля исполнителя",)),
           ("нет статуса", ("нет колонки статуса",)),
           ("в выгрузке только закрытые заявки", ("только закрытыми",)),
           ("данных слишком мало", ("слишком мало",)))


def _cells(fmap, state) -> list[str]:
    return [f"{x['channel'].split('. ')[1]} — {SHORT[x['region']]}" for x in fmap
            if x["state"] == state]


def lead(fmap, sig) -> str:
    """Первая строка отчёта: что узнали. Главный выход — карта, сигналы — следствие."""
    by = lambda st: [x for x in fmap if x["state"] == st]
    services, topic = by("засчитан по службам"), by("засчитан по теме региона, не по службам")
    region, noise, absent = by("по региону"), by("не засчитан"), by("не считается")
    why = {}
    for x in absent:
        k = next((name for name, keys in WHY_NOT if any(t in x["reason"] for t in keys)), "другое")
        why[k] = why.get(k, 0) + 1
    return (f"**Что мы узнали: работу служб по этой выгрузке можно проверить в "
            f"{len(services)} {'клетке' if len(services) == 1 else 'клетках'} из {len(fmap)}** (регион × канал) — "
            + "; ".join(_cells(fmap, "засчитан по службам")) + ". "
            f"Ещё {len(region) + len(topic)} дают картину только по региону или теме, без "
            f"служб ({'; '.join(_cells(fmap, 'по региону') + _cells(fmap, 'засчитан по теме региона, не по службам'))}); "
            f"{len(noise)} не выдержали проверок на шум; {len(absent)} не считаются вовсе — "
            + ", ".join(f"{k}: {v}" for k, v in sorted(why.items(), key=lambda kv: -kv[1]))
            + f". **Поэтому сигналов по правилу {len(sig)}:** правило требует, чтобы у службы "
            "сошлись два канала, один из них — поток после закрытия, а по службам он не "
            "засчитан нигде. Ноль сигналов — следствие карты, а не вывод о службах.")


def render(sig, eligible, fmap, ch1, ch3) -> str:
    L = [lead(fmap, sig), "", FRAME, "", "# 5o: карта наблюдаемости и сигналы", "",
         f"Сигналов по правилу: {len(sig)}. Служб, у которых засчитаны хотя бы два канала "
         f"и среди них канал 3, — {eligible}; ожидаемое число случайных совпадений при "
         f"независимых каналах — {eligible / 16:.2f}.", ""]
    L += ["## Итоговая карта — после проверок на шум", "",
          "| Регион | " + " | ".join(CH) + " |", "|---|---|---|---|"]
    for region in SPEC:
        cells = {x["channel"]: x["state"] for x in fmap if x["region"] == region}
        L.append(f"| {SHORT[region]} | " + " | ".join(cells[c] for c in CH) + " |")
    L += ["", "| Регион | Канал | Итог | Причина |", "|---|---|---|---|"]
    for x in fmap:
        L.append(f"| {SHORT[x['region']]} | {x['channel']} | {x['state']} | {x['reason']} |")

    c1 = {r["region"]: r for r in ch1}
    c3 = {r["region"]: r for r in ch3}
    tur = c1.get("Туркестанская область", {})
    kos = c1.get("Костанайская область", {})
    L += ["", "## Сверка с прогнозом, записанным до прогона", "",
          "| Прогноз (CLAUDE.md, 5o) | Факт |", "|---|---|",
          "| Канал 1 засчитан в Костанае; Туркестан под вопросом | Костанай — "
          + ("засчитан" if kos.get("counted") else "не засчитан") + "; Туркестан — "
          + ("засчитан" if tur.get("counted") else "не засчитан") + " |",
          "| Ранговая корреляция между половинами периода около 0.3–0.5 | "
          + "; ".join(f"{SHORT[r['region']]} "
                      + (f"{r['check_b_split_half']['rho']:.2f}" if r["check_b_split_half"]["rho"]
                         is not None else "—") for r in ch1) + " |",
          "| Канал 2 по службам не считается нигде; Алматы по региону 726 из 19 334 | "
          + ("по службам — нигде" if not any(x["channel"] == CH[1] and x["state"].startswith(
              "по службам") for x in fmap) else "по службам — есть")
          + "; Алматы по региону — " + _almaty_open(fmap) + " |",
          "| Канал 3 по службам — единицы; у большинства всплесков хвост короче двух недель | "
          + "; ".join(f"{SHORT[r['region']]}: служб {r['services_3plus']}, медиана хвоста "
                      f"{r['tail_median']:.1f} дн." for r in ch3) + " |",
          f"| Сигналов — 0; больше пяти — искать ошибку | {len(sig)} |", "",
          "**Итог сверки:** ноль сигналов "
          + ("угадан" if not sig else "не угадан") + "; канал 1 предсказан "
          + ("неверно в обе стороны — ждали Костанай, он не засчитан; Туркестан считали "
             "сомнительным, он засчитан" if tur.get("counted") and not kos.get("counted")
             else "верно" if kos.get("counted") and not tur.get("counted")
             else "неверно наполовину") + ".", ""]
    return "\n".join(L)


def main() -> None:
    obs, ch1, ch3 = _read("observability.json"), _read("channel1.json"), _read("channel3.json")
    fmap = final_map(obs, ch1, ch3)
    sig, eligible = signals(ch1, ch3)
    (OUT / "signals.json").write_text(json.dumps({"signals": sig, "eligible": eligible,
                                                  "final_map": fmap}, ensure_ascii=False,
                                                 indent=2), encoding="utf-8")
    (OUT / "signals.md").write_text(render(sig, eligible, fmap, ch1, ch3), encoding="utf-8")
    print(f"сигналов {len(sig)}; служб с 2+ каналами, включая канал 3: {eligible}")


if __name__ == "__main__":
    main()
