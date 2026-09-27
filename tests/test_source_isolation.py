"""Регрессии 5p: отказ источника до обработки, real-артефакты не меняются.

Без real dataset проверяется и отсутствие появления рабочих файлов (CI).
С локальными данными после каждого случая дополнительно сверяются SHA256.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ARTIFACTS = ("data/unified.parquet", "data/SOURCE", "reports/metrics.json",
             "reports/baseline_vs_model.md", "reports/predictions.csv",
             "reports/SOURCE", "models/model.pkl")


def snapshot():
    result = {}
    for name in ARTIFACTS:
        p = ROOT / name
        result[name] = None
        if p.is_file():
            with p.open("rb") as stream:
                result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def status():
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)


def main():
    baseline, git_before = snapshot(), status()
    env = {k: v for k, v in os.environ.items()
           if k not in ("NAZAR_SOURCE", "NAZAR_RAW_DIR", "NAZAR_WORK_DIR")}
    fixture = ROOT / "tests/fixtures/fake_export"
    csv = fixture / "Обращения граждан 109 - Карагандинская область.csv"
    count = 0

    def reject(name, command, settings, expected):
        nonlocal count
        r = subprocess.run([PY, *command], cwd=ROOT, env={**env, **settings},
                           capture_output=True, text=True)
        assert r.returncode == 2, f"{name}: код {r.returncode}, ожидался 2 (лог скрыт)"
        assert "ОШИБКА ИСТОЧНИКА" in r.stderr and expected in r.stderr, name
        assert "ШАГ 2" not in r.stdout and "ШАГ 5" not in r.stdout, name
        assert snapshot() == baseline, f"{name}: изменились настоящие артефакты"
        assert status() == git_before, f"{name}: изменился git status"
        count += 1
        print(f"OK {name}: exit=2; real SHA256/absence и git status без изменений")

    with tempfile.TemporaryDirectory(prefix="nazar-source-guard-") as tmp:
        tmp = Path(tmp)
        copied = tmp / "renamed-input.csv"
        shutil.copyfile(csv, copied)
        reject("A copied renamed CSV", ["train.py", "--csv", str(copied)],
               {"NAZAR_SOURCE": "real"}, "SHA256 fingerprint")
        copied_dir = tmp / "customer-delivery"
        shutil.copytree(fixture, copied_dir)
        build = [str(Path(PY).parent / "nazar-build-data")]
        reject("B copied raw directory", build,
               {"NAZAR_SOURCE": "real", "NAZAR_RAW_DIR": str(copied_dir)}, "SHA256 fingerprint")
        copy_file_link, copy_dir_link = tmp / "renamed-link.csv", tmp / "delivery-link"
        copy_file_link.symlink_to(copied)
        copy_dir_link.symlink_to(copied_dir, target_is_directory=True)
        reject("C copied CSV symlink", ["train.py", "--csv", str(copy_file_link)],
               {"NAZAR_SOURCE": "real"}, "SHA256 fingerprint")
        reject("C copied directory symlink", build,
               {"NAZAR_SOURCE": "real", "NAZAR_RAW_DIR": str(copy_dir_link)}, "SHA256 fingerprint")
        manifest = json.loads((fixture / "manifest.json").read_text())
        probe = "from src.paths import validate_raw_source; import sys; validate_raw_source(sys.argv[1])"
        for i, name in enumerate(manifest["files"]):
            renamed = tmp / f"unrelated-name-{i}.bin"
            shutil.copyfile(fixture / name, renamed)
            reject(f"fingerprint raw {i + 1}", ["-c", probe, str(renamed)],
                   {"NAZAR_SOURCE": "real"}, "SHA256 fingerprint")
        mixed = tmp / "mixed-input"
        mixed.mkdir()
        (mixed / "new.csv").write_text("new customer input\\n")
        shutil.copyfile(csv, mixed / "unexpected.dat")
        reject("one fake among other files", build,
               {"NAZAR_SOURCE": "real", "NAZAR_RAW_DIR": str(mixed)}, "SHA256 fingerprint")
        nested = tmp / "nested-input"
        nested.mkdir()
        (nested / "linked-subdir").symlink_to(mixed, target_is_directory=True)
        reject("nested directory symlink", build,
               {"NAZAR_SOURCE": "real", "NAZAR_RAW_DIR": str(nested)}, "SHA256 fingerprint")
        fresh = tmp / "fresh.csv"
        fresh.write_text("unknown customer fingerprint\\n")
        accepted = subprocess.run([PY, "-c", probe, str(fresh)], cwd=ROOT,
                                  env={**env, "NAZAR_SOURCE": "real"}, capture_output=True)
        assert accepted.returncode == 0, "неизвестный fingerprint должен приниматься"
        assert snapshot() == baseline and status() == git_before
        print("OK unknown fingerprint accepted; real SHA256/absence и git status без изменений")
        file_link, dir_link = tmp / "input.csv", tmp / "input-dir"
        file_link.symlink_to(csv)
        dir_link.symlink_to(fixture, target_is_directory=True)
        for name, path in (("absolute", csv), ("relative", csv.relative_to(ROOT)),
                           ("dotdot", fixture / ".." / "fake_export" / csv.name),
                           ("file-symlink", file_link), ("dir-symlink", dir_link / csv.name)):
            reject("real --csv " + name, ["train.py", "--csv", str(path)],
                   {"NAZAR_SOURCE": "real"}, "нельзя использовать в real")
        for name, path in (("relative", fixture.relative_to(ROOT)), ("symlink", dir_link)):
            reject("real RAW_DIR " + name, ["-c", "from src.cli import build_data; build_data()"],
                   {"NAZAR_SOURCE": "real", "NAZAR_RAW_DIR": str(path)}, "нельзя использовать в real")
        reject("synth corpus in real", ["train.py", "--csv", "data/synth/train_eval.csv"],
               {"NAZAR_SOURCE": "real"}, "нельзя использовать в real")
        for name, work in (("root", ROOT), ("dotdot root", ROOT / "tests/.."),
                           ("inside real data", ROOT / "data/guard-work")):
            reject("fake work " + name, ["-c", "from src import paths"],
                   {"NAZAR_SOURCE": "fake", "NAZAR_WORK_DIR": str(work)},
                   "корнем проекта" if "root" in name else "пересекаются")
        alias = tmp / "root-alias"
        alias.symlink_to(ROOT, target_is_directory=True)
        reject("fake root symlink", ["-c", "from src import paths"],
               {"NAZAR_SOURCE": "fake", "NAZAR_WORK_DIR": str(alias)}, "корнем проекта")
        work = tmp / "output-alias"
        work.mkdir()
        (work / "reports").symlink_to(ROOT / "reports", target_is_directory=True)
        reject("fake reports symlink", ["-c", "from src import paths"],
               {"NAZAR_SOURCE": "fake", "NAZAR_WORK_DIR": str(work)}, "пересекаются")
        work2 = tmp / "output-file-alias"
        (work2 / "reports").mkdir(parents=True)
        (work2 / "reports/metrics.json").symlink_to(ROOT / "reports/metrics.json")
        reject("fake metrics symlink", ["-c", "from src import paths"],
               {"NAZAR_SOURCE": "fake", "NAZAR_WORK_DIR": str(work2)}, "пересекаются")
        artifact = tmp / "downstream"
        artifact.mkdir()
        for kind, code in (("risk", "from src.risk_view import load_risk; load_risk"),
                           ("dashboard", "from src.dashboard import load_data; load_data")):
            reject(kind + " missing marker", ["-c", code + f"({str(artifact / 'unused')!r})"],
                   {"NAZAR_SOURCE": "real"}, "нет SOURCE")
        (artifact / "SOURCE").write_text("fake\n")
        for kind, code in (("risk", "from src.risk_view import load_risk; load_risk"),
                           ("dashboard", "from src.dashboard import load_data; load_data")):
            reject(kind + " wrong marker", ["-c", code + f"({str(artifact / 'unused')!r})"],
                   {"NAZAR_SOURCE": "real"}, "не совпадает")
        (artifact / "SOURCE").write_text("real\n")
        (artifact / "metrics.json").write_text('{"source": "fake"}')
        reject("metrics/marker mismatch", ["-c", "from src.risk_view import headline; "
               + f"headline({str(artifact / 'metrics.json')!r})"],
               {"NAZAR_SOURCE": "real"}, "противоречит")
    print(f"ИТОГ: {count} отказов; локальных real-артефактов под SHA256: "
          f"{sum(v is not None for v in baseline.values())}; остальные проверены на отсутствие")
    return 0


if __name__ == "__main__":
    sys.exit(main())
