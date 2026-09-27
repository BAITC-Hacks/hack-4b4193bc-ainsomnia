"""Поддельная выгрузка 109 для tests/ — генератор (CLAUDE.md, 5p).

    .venv/bin/python -m src.fake.generate                        # -> tests/fixtures/fake_export/
    .venv/bin/python -m src.fake.generate --out DIR              # в другой каталог (тест сверки сумм)
    .venv/bin/python -m src.fake.generate --out DIR --variant new-category

Seed зафиксирован: повторный запуск даёт тот же набор до байта, суммы — в manifest.json.
Персональных данных нет по построению: все ФИО, телефоны, номера и адреса — из списка
фальшивых значений 5p, и стоят только там, где их обязана найти проверка ПДн.
Названия категорий — настоящие значения справочника (они в tests/topic_map_baseline.json),
организации-исполнители — поддельные. Настоящие данные генератор не читает.
Строки со сдвигом полей записаны сломанными кавычками в байтах csv: сдвиг делает сам
разбор pandas, как в настоящей выгрузке, а генератор после записи это сверяет.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from src import paths
from src.adapters.schemas import HEAD, FILES
from src.topic_mapping import classify_appeal, map_topic

SEED = 20260927
D = lambda s: date.fromisoformat(s)

# ---------------------------------------------------------------- фальшивые значения (5p)
FIO = ("Тестов Тест Тестович", "Тестова Теста Тестовна")
PHONES = ("8 700 000 00 00", "8-700-000-00-00", "8(700)0000000", "+7 700 000 0000")
DIGITS12 = ("000000000000", "999999999999")
ADDR_FULL, ADDR_SHORT = "ул. Тестовая, дом 0, кв 0", "ул. Тестовая, дом 0"
IP_KOS, IP_KAR = "ИП Тестов Т.Т.", "ИП «Тестова»"
ORGS = ("ГУ «Отдел тестового хозяйства»", "ТОО «Тест-Сервис»", "ГКП «Тестводоканал»",
        "АО «Тестэнерго»", "ГУ «Отдел тестов»")
NEW_INFO_CATEGORY = "Справочные вопросы по тарифам (тестовая категория)"

# ---------------------------------------------------------------- справочники семейств
CAT = {
    "kar_p": ["Дворовое/уличное освещение", "Отсутствие электроэнергии", "Открытие/закрытие дорог",
              "Благоустройство города", "Благоустройство дворовых территорий", "Дороги",
              "Отсутствие воды", "Ликвидация аварий на водопроводно-канализационных сетях",
              "Отлов и уничтожение бродячих собак и кошек", "Вывоз ТБО", "Иное",
              "Организация движения автобусов", "Снабжение потребителей тепловой энергией и ГВС",
              "Передача и распределение тепловой энергии", "Газификация", "Телекоммуникация",
              "Действие/бездействие кондоминиума", "Среднее образование"],
    "kar_i": ["Информирование населения", "Контактные данные, адрес"],
    "kar_s": ["Зачитывание ответа на свой запрос", "Cброс звонка"],
    "kos_p": ["ДОРОЖНАЯ ИНФРАСТРУКТУРА", "УЛИЧНОЕ ОСВЕЩЕНИЕ", "ТВЕРДЫЕ БЫТОВЫЕ ОТХОДЫ", "КОЛОДЕЦ, ЛЮК",
              "ВОДОСНАБЖЕНИЕ ГОРОДА", "ОБЩЕСТВЕННЫЙ ТРАНСПОРТ", "ГАЗОСНАБЖЕНИЕ", "ЗДРАВООХРАНЕНИЕ"],
    "kos_spike": "ЗЕЛЕНЫЕ НАСАЖДЕНИЯ",
    "kos_i": ["СПРАВОЧНАЯ ИНФОРМАЦИЯ"], "kos_s": ["КАЧЕСТВО РАБОТЫ ЕКЦ", "СРЫВ ЗВОНКА"],
    # без «ЭЛЕКТРОСНАБЖЕНИЕ ГОРОДА»: та же тема, что у пачки 02.06, — пачка была бы не ровно 15
    "tur_p": ["ВЕТСЕРВИС", "ТВЕРДЫЕ БЫТОВЫЕ ОТХОДЫ", "ВОДОСНАБЖЕНИЕ ГОРОДА",
              "ЗЕЛЕНЫЕ НАСАЖДЕНИЯ", "ДОРОЖНАЯ ИНФРАСТРУКТУРА", "ГАЗОСНАБЖЕНИЕ"],
    "tur_burst": "УЛИЧНОЕ ОСВЕЩЕНИЕ",
    "tur_i": ["СПРАВОЧНАЯ ИНФОРМАЦИЯ"], "tur_s": ["КАЧЕСТВО РАБОТЫ ЕКЦ"],
    "vko_p": ["ЖКХ, бытовое обслуживание населения", "Связь и информация", "Дороги",
              "Использование природно-сырьевых ресурсов, экология", "Транспорт, коммуникации"],
    "vko_i": ["Справка", "Государственное устройство"],
    "alm_p": ["Электроснабжение города", "Ветеринария", "Водоснабжение в частном секторе",
              "Твердые бытовые отходы", "Дорожная инфраструктура", "Уличное освещение", "Зеленные насаждения"],
    "alm_i": ["Справочная информация"],
    "akm_p": ["Канализационный колодец на подпоре", "Отсутствие электроэнергии",
              "Отсутствие освещения в ночное время", "Ремонт существующих дорог, тротуаров",
              "Уборка переполненных контейнеров, урн на площадке", "Отсутствие отопления", "Скос травы"],
    "akm_i": ["Справка"], "akm_s": ["Благодарность, положительные отзывы"],
    "pav_p": ["Электроснабжение города", "Водоснабжение МЖД", "Здравоохранение", "Теплоснабжение города",
              "Зеленные насаждения", "Сан чистка", "Дорожная инфраструктура", "Уличное освещение"],
    "pav_i": ["Справочная информация", "Вопросы карантинного ограничения"],
}



def days(a, b):
    return [D(a) + timedelta(i) for i in range((D(b) - D(a)).days + 1)]


def broken(a, b, c):
    """Сломанные кавычки, как в настоящих битых строках (5p): разбор даёт ДВА поля —
    «a "b» и « c""» — вместо одного. Проверено на pandas 3.0.5."""
    return f'"{a} ""{b}"," {c}"""""'


class Gen:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.used = set()         # (регион, день, минута или секунда) — уникально на весь регион

    def pick(self, seq):
        return seq[int(self.rng.integers(len(seq)))]

    def spread(self, n, dd):
        """n дат по дням dd: каждый день хотя бы раз (если n >= len(dd)), остальное — случайно."""
        base = list(dd) if n >= len(dd) else []
        rest = [dd[i] for i in self.rng.integers(len(dd), size=n - len(base))]
        out = base + rest
        self.rng.shuffle(out)
        return out

    def stamps(self, dates, unit, tag):
        """Уникальное время внутри региона (tag) по всем вызовам: минута (Караганда) или секунда."""
        per, out = (1440 if unit == "m" else 86400), []
        for d in dates:
            while True:
                k = int(self.rng.integers(per))
                if (tag, d, k) not in self.used:
                    self.used.add((tag, d, k)); break
            base = datetime(d.year, d.month, d.day)
            out.append(base + (timedelta(minutes=k) if unit == "m" else timedelta(seconds=k)))
        return out

    def classes(self, spec):
        """spec: [(категории, число)] -> перемешанный список категорий ровно этих объёмов."""
        out = [self.pick(cats) for cats, n in spec for _ in range(n)]
        self.rng.shuffle(out)
        return out


def iso_us(t):
    return t.strftime("%Y-%m-%d %H:%M:%S.%f")


def kar_date(t):
    return f"{t.month}/{t.day}/{t.strftime('%y')} {t.hour}:{t.minute:02d}"


# ---------------------------------------------------------------- семейства
def karaganda(g):
    early, late = days("2022-01-01", "2023-06-30"), days("2023-07-01", "2023-12-31")
    rows = []
    rate = {c: 0.15 + 0.6 * i / (len(CAT["kar_p"]) - 1) for i, c in enumerate(CAT["kar_p"])}

    def row(t, atype, sub, org, overdue_rate):
        late_ = g.rng.random() < overdue_rate
        dd = g.rng.uniform(16, 120) if late_ else g.rng.uniform(0.05, 14)
        upd = t + timedelta(days=float(dd))
        ans = g.pick(["Быстрый ответ", "Быстрый ответ", "Письменное обращение", ""])
        return [kar_date(t), kar_date(upd), kar_date(t), atype, g.pick(["Call-центр", "Call-центр", "Соц. сети (вручную)"]),
                g.pick(["город Караганда", "город Караганда", "город Темиртау", "город Балхаш"]),
                g.pick(["Район имени Казыбек би", "Район имени Алихана Бокейхана"]), "",
                org, sub, org, ans, "Физ. лицо"]

    # Обращение + Инцидент: 3 000 до 2023-07-01 и 2 000 после — train и test train.py (5p)
    t_early = g.stamps(g.spread(3000, early), "m", "kar")
    t_late = g.stamps(g.spread(2000, late), "m", "kar")
    ip_left = 25                                      # «ИП «Тестова»» — в тестовом периоде, чтобы попасть в блок риска
    for i, t in enumerate(t_early + t_late):
        sub = g.pick(CAT["kar_p"])
        org = IP_KAR if (i >= 3000 and ip_left > 0 and i % 40 == 0) else g.pick(ORGS)
        ip_left -= org == IP_KAR
        rows.append(row(t, "Обращение" if i % 12 else "Инцидент", sub, org, rate[sub]))
    assert ip_left == 0, ip_left
    # остальное — «Запрос информации»: 1 000 problem, 2 970 info, 978 system
    alld = early + late
    rest = g.classes([(CAT["kar_p"], 1000), (CAT["kar_i"], 2970), (CAT["kar_s"], 978)])
    info_rows = []
    for sub, t in zip(rest, g.stamps(g.spread(len(rest), alld), "m", "kar")):
        r = row(t, "Запрос информации", sub, "Справочная" if sub not in CAT["kar_p"] else g.pick(ORGS), 0.0)
        rows.append(r)
        if sub in CAT["kar_i"]:
            info_rows.append(r)
    dups = [list(info_rows[i]) for i in range(30)]     # 30 лишних копий — одинаковые строки в сводной
    raw = [list(r) for r in rows] + dups
    # 20 строк со сдвигом: сломанное название в executor_gov_org, последнего поля нет
    shifted = []
    for k, t in enumerate(g.stamps([early[int(i)] for i in g.rng.integers(len(early), size=20)], "m", "kar")):
        r = row(t, "Обращение", g.pick(CAT["kar_p"]), "", 0.3)
        head = r[:10]
        line = (_line(head) + "," + broken("ГУ", f"Отдел тестов №{k + 1}", "Тестового района") + ","
                + r[11])
        shifted.append(line)
    # 2 строки целиком в одной ячейке created_date
    whole = []
    for t in g.stamps([early[5], late[5]], "m", "kar"):
        r = row(t, "Обращение", g.pick(CAT["kar_p"]), g.pick(ORGS), 0.3)
        whole.append('"' + ",".join(r) + '"')
    return HEAD["kar"], raw, shifted + whole


def _line(fields):
    b = io.StringIO()
    csv.writer(b, lineterminator="").writerow(fields)
    return b.getvalue()


def kos_tur(g, kind):
    if kind == "kos":
        dd = days("2025-03-01", "2025-08-31")
    else:
        dd = days("2025-01-01", "2025-02-28") + days("2025-06-01", "2025-08-24")
    cats = []
    if kind == "kos":
        # фон «ЗЕЛЕНЫЕ НАСАЖДЕНИЯ»: 0, 1, 2 в день по кругу, 27.08.2025 — 40 (всплеск, 5p)
        for i, d in enumerate(dd):
            n = 40 if d == D("2025-08-27") else i % 3
            cats += [(d, CAT["kos_spike"])] * n
        n_other = 495 - len(cats)
        spec = [(CAT["kos_p"], n_other), (CAT["kos_i"], 3), (CAT["kos_s"], 2)]
        region = ["КОСТАНАЙ"]
    else:
        cats += [(D("2025-06-02"), CAT["tur_burst"])] * 15          # пачка 15 после провала
        spec = [(CAT["tur_p"], 446 - 15), (CAT["tur_i"], 3), (CAT["tur_s"], 1)]
        region = ["ТУРКЕСТАН", "САЙРАМСКИЙ", "КЕНТАУ"]
    rest = g.classes(spec)
    # каждый день хотя бы одно обращение: дни без фоновых строк закрываются первыми
    busy = {d for d, _ in cats}
    empty = [d for d in dd if d not in busy]
    fill = empty + [dd[i] for i in g.rng.integers(len(dd), size=len(rest) - len(empty))]
    g.rng.shuffle(fill)
    cats += list(zip(fill, rest))
    cats.sort(key=lambda x: x[0])
    stamps = g.stamps([d for d, _ in cats], "s", kind)
    rows, n_ip = [], 0
    for j, ((d, c), t) in enumerate(zip(cats, stamps)):
        sla = 7
        spent = float(g.rng.uniform(0.05, 10))
        fin = t + timedelta(days=spent)
        org = g.pick(ORGS)
        if kind == "kos" and n_ip < 30 and c in CAT["kos_p"] and j % 9 == 0:
            org, n_ip = IP_KOS, n_ip + 1
        iid = (900000 if kind == "kos" else 800000) + j + 1
        r = [str(iid), f"IM{iid}", iso_us(t), c, c, "УЛИЦА", str(sla), iso_us(fin), str(int(spent)),
             iso_us(t + timedelta(minutes=5)), "", str(spent > sla), "инцидент", "False",
             g.pick(["Call центр", "Call центр", "WhatsApp"]),
             "закрыто" if j % 20 else "закрыто инициатором", org, "0", "", "", g.pick(region),
             str(sla - int(spent))]
        if kind == "kos":
            r.append("Работы выполнены")
        r += ["2025-09-15 04:00:00.000000", "2025-09-15 04:00:00.000000"]
        rows.append(r)
    if kind == "kos":
        assert n_ip == 30, n_ip
    return HEAD[kind], rows, []


def vko(g):
    dd = days("2025-03-01", "2025-08-24")
    spec = [(CAT["vko_p"], 197), (CAT["vko_i"], 297)]
    cats = g.classes(spec)
    ts = g.stamps(g.spread(len(cats), dd), "s", "vko")
    rows, fio_left = [], 37                              # 37 + 3 строки со сдвигом = 40 ФИО
    for j, (c, t) in enumerate(zip(cats, ts)):
        info = c in CAT["vko_i"]
        fio = ""
        if fio_left and j % 12 == 0:
            fio, fio_left = FIO[fio_left % 2], fio_left - 1
        rows.append([f"KZ25{j + 1:07d}", t.strftime("%d.%m.%Y %H:%M:%S"),
                     (t + timedelta(hours=2)).strftime("%d.%m.%Y %H:%M:%S"), "Усть-Каменогорск", "", "",
                     fio, "", "Консультации" if info else "Жалобы", "ЕКЦ 109", c, c,
                     g.pick(ORGS), "", "Выполнено", "Закрыто", ""])
    assert fio_left == 0, fio_left
    info_rows = [r for r in rows if r[10] in CAT["vko_i"]]
    dups = []
    for k, r in enumerate(info_rows[:3]):                # те же данные до секунды, другие номера заявок
        d = list(r); d[0] = f"KZ259{k + 1:06d}"; dups.append(d)
    shifted = []
    for k, t in enumerate(g.stamps([dd[40 + k] for k in range(3)], "s", "vko")):
        f = [f"KZ258{k + 1:06d}", t.strftime("%d.%m.%Y %H:%M:%S"), t.strftime("%d.%m.%Y %H:%M:%S"),
             "Усть-Каменогорск", "", None, FIO[k % 2], "", "Жалобы", "ЕКЦ 109", CAT["vko_p"][0],
             CAT["vko_p"][0], g.pick(ORGS), "", "Выполнено", "Закрыто"]      # operator — выпадает
        shifted.append(_line(f[:5]) + "," + broken("ул.", "Тестовая", "дом 0") + "," + _line(f[6:]))
    return HEAD["vko"], rows + dups, shifted


def almaty(g):
    dd = days("2025-03-01", "2025-08-24")
    cats = g.classes([(CAT["alm_p"], 318), (CAT["alm_i"], 170)])
    ts = g.stamps(g.spread(len(cats), dd), "s", "alm")
    rows = []
    for j, (c, t) in enumerate(zip(cats, ts)):
        info = c in CAT["alm_i"]
        closed = j % 15 != 0
        rows.append([f"KZ25{j + 1:07d}", t.strftime("%d.%m.%Y %H:%M:%S"),
                     (t + timedelta(hours=5)).strftime("%d.%m.%Y %H:%M:%S") if closed else "",
                     "", "Проблема устранена" if closed else "", g.pick(ORGS),
                     "Закрыто" if closed else "В работе", c, "Любые справки" if info else c,
                     "Закрыто" if closed else "В работе", g.pick(["ЕКЦ 109", "ЕКЦ 109", "Whatsapp"])])
    # 12 строк со сдвигом: текст заявителя со сломанными кавычками в contractor
    tails = ([f"{p}, прорвало трубу" for p in PHONES] + [f"ИИН {x}, нет света" for x in DIGITS12 * 2]
             + [f"{ADDR_FULL}, нет воды"] * 4)
    shifted = []
    for k, t in enumerate(g.stamps([dd[60 + k] for k in range(12)], "s", "alm")):
        f = [f"KZ258{k + 1:06d}", t.strftime("%d.%m.%Y %H:%M:%S"), "", "", "", None, "Закрыто",
             CAT["alm_p"][0], CAT["alm_p"][0], "Закрыто"]                   # submittal_channel выпадает
        shifted.append(_line(f[:5]) + "," + broken(FIO[0], "жалоба", tails[k]) + "," + _line(f[6:]))
    return HEAD["alm"], rows, shifted


def akmola(g):
    dd = days("2025-03-01", "2025-08-24")
    cats = g.classes([(CAT["akm_p"], 390), (CAT["akm_i"], 5), (CAT["akm_s"], 2)])
    ts = g.stamps(g.spread(len(cats), dd), "s", "akm")
    iso_ms = lambda t: t.strftime("%Y-%m-%d %H:%M:%S.") + f"{t.microsecond // 1000:03d}"
    rows = []
    for j, (c, t) in enumerate(zip(cats, ts)):
        plan = t + timedelta(days=5)
        rows.append([f"{t:%d%m%y}-000-{j + 1:03d}", g.pick(ORGS), iso_ms(t),
                     "Передано в службу " if j % 3 else "В работе", "AI-Komek 109 *Службы региона",
                     iso_ms(plan), iso_ms(plan), g.pick(["г. Кокшетау", "район Целиноградский"]), c])
    shifted = []
    for k, t in enumerate(g.stamps([dd[30 + k] for k in range(3)], "s", "akm")):
        st = "В работе" if k < 2 else ""                 # у 2 из 3 статус уезжает в current_project
        f = [f"{t:%d%m%y}-900-{k + 1:03d}", None, iso_ms(t), st, "AI-Komek 109 *Службы региона",
             iso_ms(t), iso_ms(t), "г. Кокшетау"]                           # direction выпадает
        shifted.append(_line(f[:1]) + "," + broken("ГУ", f"Отдел тестов №{k + 1}", "Тестового района")
                       + "," + _line(f[2:]))
    return HEAD["akm"], rows, shifted


def pavlodar(g):
    dd = days("2025-03-01", "2025-08-24")
    cats = g.classes([(CAT["pav_p"], 420), (CAT["pav_i"], 180)])
    ts = g.stamps(g.spread(len(cats), dd), "s", "pav")
    allc = CAT["pav_p"] + CAT["pav_i"]
    parts = ([], [])
    for j, (c, t) in enumerate(zip(cats, ts)):
        info = c in CAT["pav_i"]
        k = allc.index(c)
        rt = ("1", "CONSULTATION") if info else ("5", "INCIDENT")
        parts[j // 300].append([str(1_000_001 + j), f"KZP25{j + 1:07d}", iso_us(t),
                                "CLOSED" if j % 25 else "PROCESSING", str(40000 + k), c,
                                str(31900 + k), c, rt[0], rt[1], "2025-08-25 06:30:07.409588"])
    return parts


# ---------------------------------------------------------------- запись и самопроверка
def write(path: Path, head: str, rows, raw_lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [head] + [_line(r) for r in rows] + list(raw_lines)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def generate(out: Path, variant: str | None = None) -> dict:
    g = Gen(SEED)
    fam = {"kar": karaganda(g), "kos": kos_tur(g, "kos"), "tur": kos_tur(g, "tur"),
           "vko": vko(g), "alm": almaty(g), "akm": akmola(g)}
    p1, p2 = pavlodar(g)
    if variant == "new-category":
        # 40 строк ВКО с новой категорией, уходящей в info: проверка разметки обязана упасть
        head, rows, sh = fam["vko"]
        extra = []
        for k, d in enumerate(days("2025-07-01", "2025-08-09")):
            r = list(rows[0]); r[0] = f"KZ257{k + 1:06d}"
            r[1] = datetime(d.year, d.month, d.day, 12, 0, k).strftime("%d.%m.%Y %H:%M:%S")
            r[10] = r[11] = NEW_INFO_CATEGORY; r[6] = ""
            extra.append(r)
        fam["vko"] = (head, rows + extra, sh)
    elif variant:
        raise SystemExit(f"неизвестный вариант {variant!r}")
    for k, (head, rows, raw) in fam.items():
        write(out / FILES[k], head, rows, raw)
    write(out / FILES["pav1"], HEAD["pav"], p1, [])
    write(out / FILES["pav2"], HEAD["pav"], p2, [])
    man = {"seed": SEED, "variant": variant or "base",
           "files": {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(out.rglob("*.csv"))}}
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return man


def self_check(out: Path, variant: str | None = None) -> None:
    """Сверка с постановкой 5p по записанным файлам — тем же разбором, что у адаптеров."""
    import pandas as pd
    from src.checks.field_shift import almaty_shift_rules, karaganda_shift_rules, union, vko_shift_rules
    from src.checks.pii_scan import scan_dir, total

    rd = lambda k: pd.read_csv(out / FILES[k], dtype=str)
    n = {k: len(rd(k)) for k in FILES}
    want = {"kar": 10000, "kos": 500, "tur": 450, "vko": 500 + (40 if variant else 0), "alm": 500,
            "akm": 400, "pav1": 300, "pav2": 300}
    assert n == want, n
    kar = rd("kar")
    bad_date = pd.to_datetime(kar.created_date, format="%m/%d/%y %H:%M", errors="coerce").isna()
    assert int(bad_date.sum()) == 2, int(bad_date.sum())
    assert int(union(karaganda_shift_rules(kar[~bad_date])).sum()) == 20
    assert int(union(almaty_shift_rules(rd("alm"))).sum()) == 12
    assert int(union(vko_shift_rules(rd("vko"))).sum()) == 3
    akm = rd("akm")
    assert int(pd.to_datetime(akm.creation_date, format="ISO8601", errors="coerce").isna().sum()) == 3
    assert not set(rd("pav1").id) & set(rd("pav2").id)

    # потолок 9 жалоб в день по теме — по итоговому набору, ПОСЛЕ вставки копий (5p)
    frames = []
    for k, col, dcol, fmt in (("kar", "sub_category", "created_date", "%m/%d/%y %H:%M"),
                              ("kos", "servicelevel1", "createddate", "ISO8601"),
                              ("tur", "servicelevel1", "createddate", "ISO8601"),
                              ("vko", "category", "creation_date", "%d.%m.%Y %H:%M:%S"),
                              ("alm", "category", "creation_date", "%d.%m.%Y %H:%M:%S"),
                              ("akm", "direction", "creation_date", "ISO8601"),
                              ("pav1", "category_name", "create_date", "ISO8601"),
                              ("pav2", "category_name", "create_date", "ISO8601")):
        d = rd(k)
        day = pd.to_datetime(d[dcol], format=fmt, errors="coerce").dt.normalize()
        frames.append(pd.DataFrame({"reg": k[:3], "day": day, "cat": d[col]}).dropna())
    f = pd.concat(frames)
    f = f[f.cat.map(classify_appeal) == "problem"]
    f["topic"] = f.cat.map(map_topic)
    c = f.groupby(["reg", "day", "topic"]).size()
    allowed = {("kos", pd.Timestamp("2025-08-27"), "благоустройство и озеленение"): 40,
               ("tur", pd.Timestamp("2025-06-02"), "электроснабжение и освещение"): None}
    over = {k: int(v) for k, v in c[c > 9].items() if k not in allowed}
    assert not over, f"потолок 9 нарушен: {over}"
    assert int(c[("kos", pd.Timestamp("2025-08-27"), "благоустройство и озеленение")]) == 40

    # ПДн: ровно заложенное (5p, «Положительный тест»)
    t = total(scan_dir(out))
    exp = {"телефон": 4, "12 цифр": 4, "адрес": 7, "отчество": 52, "названий-людей": 2, "строк с ними": 55}
    got = {k: t.get(k, 0) for k in exp}
    assert got == exp, f"ПДн в наборе: {got}, ожидалось {exp}"
    print("самопроверка: строки, сдвиг, даты, id Павлодара, потолок 9, ПДн — сходится с 5p")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=paths.FAKE_RAW)
    ap.add_argument("--variant", choices=["new-category"])
    a = ap.parse_args()
    man = generate(a.out, a.variant)
    self_check(a.out, a.variant)
    print(f"{a.out}: {len(man['files'])} файлов, seed {SEED}, вариант {man['variant']}")


if __name__ == "__main__":
    main()
