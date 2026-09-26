"""Групповое разбиение корпуса A с защитой от почти-дублей (5n)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from src.topic_mapping import TOPICS

SEED = 42
JACCARD = 0.80
BLOCK = 128
SPLITS = ("train", "validation", "test")


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        x, y = self.find(a), self.find(b)
        if x == y:
            return
        if self.rank[x] < self.rank[y]:
            x, y = y, x
        self.parent[y] = x
        self.rank[x] += self.rank[x] == self.rank[y]


def groups(rows: list[dict]) -> tuple[list[list[int]], int]:
    """Объединяет одно происшествие и пары текстов с Jaccard 5-грамм >= .8."""
    uf = UnionFind(len(rows))
    first_incident: dict[str, int] = {}
    for i, row in enumerate(rows):
        incident = row.get("incident")
        if incident:
            if incident in first_incident:
                uf.union(i, first_incident[incident])
            else:
                first_incident[incident] = i

    cv = CountVectorizer(analyzer="char", ngram_range=(5, 5), binary=True,
                         lowercase=True, dtype=np.int32)
    matrix = cv.fit_transform(r["text"] for r in rows).tocsr()
    lengths = np.diff(matrix.indptr)
    near_pairs = 0
    for start in range(0, len(rows), BLOCK):
        shared = (matrix[start:start + BLOCK] @ matrix.T).tocsr()
        for local in range(shared.shape[0]):
            i = start + local
            for pos in range(shared.indptr[local], shared.indptr[local + 1]):
                j = int(shared.indices[pos])
                if j <= i:
                    continue
                shorter, longer = sorted((int(lengths[i]), int(lengths[j])))
                if shorter < JACCARD * longer:
                    continue
                common = int(shared.data[pos])
                if common / (longer + shorter - common) >= JACCARD:
                    uf.union(i, j)
                    near_pairs += 1

    by_root: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        by_root[uf.find(i)].append(i)
    return list(by_root.values()), near_pairs


def effective_size(texts) -> int:
    """Сколько различных текстов остаётся после схлопывания почти-дублей
    тем же правилом, что в проверке корпуса (Jaccard 5-грамм >= JACCARD)."""
    return len(groups([{"text": str(t)} for t in texts])[0])


def assign(rows: list[dict]) -> tuple[list[dict], dict]:
    """Возвращает строки со split/group и числовую диагностику разбиения."""
    if not rows:
        raise ValueError("корпус пуст")
    rng = np.random.default_rng(SEED)
    components, near_pairs = groups(rows)
    by_topic: dict[str, dict[str, list[list[int]]]] = {
        topic: {"incident": [], "ordinary": []} for topic in TOPICS
    }
    assignment: dict[int, str] = {}
    mixed = []
    held_groups = 0
    for g in components:
        labels = {rows[i]["label"] for i in g}
        if len(labels) != 1:
            mixed.append([rows[i]["id"] for i in g])
            continue
        topic = next(iter(labels))
        if any(rows[i].get("seed_holdout") for i in g):
            for i in g:
                assignment[i] = "seed_holdout"
            held_groups += 1
        else:
            kind = "incident" if any(rows[i].get("incident") for i in g) else "ordinary"
            by_topic[topic][kind].append(g)
    if mixed:
        raise ValueError(f"почти-дубли с разными метками: {mixed[:5]}")

    for topic, strata in by_topic.items():
        for kind, group_list in strata.items():
            total = sum(len(g) for g in group_list)
            if not total:
                continue
            shares = (0.60, 0.10, 0.30) if kind == "incident" else (0.70, 0.15, 0.15)
            target = np.array(shares) * total
            current = np.zeros(3, dtype=int)
            rng.shuffle(group_list)
            group_list.sort(key=len, reverse=True)
            for g in group_list:
                scores = []
                for s in range(3):
                    after = current.copy()
                    after[s] += len(g)
                    scores.append(float(np.square(after - target).sum()))
                chosen = int(np.argmin(scores))
                current[chosen] += len(g)
                for i in g:
                    assignment[i] = SPLITS[chosen]

    if len(assignment) != len(rows):
        raise AssertionError("не всем строкам назначен split")
    out = []
    for group_id, g in enumerate(components):
        for i in g:
            out.append({**rows[i], "split": assignment[i], "group": group_id})
    out.sort(key=lambda r: r["id"])
    summary = {
        "groups": len(components),
        "near_duplicate_pairs": near_pairs,
        "held_groups": held_groups,
        "split_counts": {s: sum(r["split"] == s for r in out)
                         for s in (*SPLITS, "seed_holdout")},
        "incident_counts": {s: len({r["incident"] for r in out
                                    if r["split"] == s and r.get("incident")})
                            for s in (*SPLITS, "seed_holdout")},
    }
    return out, summary
