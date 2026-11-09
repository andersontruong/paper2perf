"""Bounded source navigation and contract-directed retrieval; no hardware defaults."""
import re


def contents_targets(chunks, max_targets=8):
    targets = []
    pattern = re.compile(r'^\s*(.{5,180}?)(?:\.{2,}|\s{3,})(\d{1,3})\s*$')
    for chunk in chunks:
        rows = [pattern.match(line) for line in chunk['text'].splitlines()]
        rows = [m for m in rows if m]
        if len(rows) < 4:  # Avoid interpreting ordinary prose as a page index.
            continue
        for match in rows:
            title = match.group(1)
            if not re.search(r'\b(?:memory|cache|HBM\d*|SM|multiprocessor)\b', title, re.I):
                continue
            target = (chunk['document'], int(match.group(2)))
            if target not in targets:
                targets.append(target)
            if len(targets) >= max_targets:
                return targets
    return targets


def follow_contents(retrieved, all_chunks):
    targets = set(contents_targets(retrieved))
    return [c for c in all_chunks if (c['document'], c['page']) in targets][:16]


def contract_queries(gaps):
    text = ' '.join(gaps).lower()
    queries = []
    if '.capacity' in text or 'memory resource' in text:
        queries.append('GPU device memory HBM capacity')
    if 'source topology is disconnected' in text or 'connectivity' in text:
        queries.append('HBM L2 memory spaces accessible all SMs shared memory data path')
    if 'bandwidth' in text or 'clock' in text:
        queries.append('GPU cache shared memory bandwidth bytes clock boost MHz')
    return queries[:3]
