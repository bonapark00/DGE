import os
import time
from contextlib import contextmanager
from typing import Dict, List, Tuple


class LatencyLogger:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.name_to_total_s: Dict[str, float] = {}
        self.entries: List[Tuple[str, float]] = []

    @contextmanager
    def timeit(self, name: str):
        start = time.time()
        try:
            yield
        finally:
            dt = time.time() - start
            self.name_to_total_s[name] = self.name_to_total_s.get(name, 0.0) + dt
            self.entries.append((name, dt))

    def record(self, name: str, seconds: float) -> None:
        self.name_to_total_s[name] = self.name_to_total_s.get(name, 0.0) + seconds
        self.entries.append((name, seconds))

    def write_summary(self, filename: str = "summary.txt", dest_dir: str | None = None) -> None:
        # Build a hierarchical tree from dotted names.
        # Each name contributes only once to its own node ("self time"),
        # and parent nodes aggregate children so that hierarchy
        # is clear without double-counting the raw measurements.
        #
        # Example names:
        #   "2d_editing.segmentation"
        #   "2d_editing.clip_score"
        #   "3d_finetune.training_step"
        #
        # will be grouped under top-level categories:
        #   "2d_editing", "3d_finetune"

        # Node structure: {name: {"self": float, "children": set[str]}}
        nodes: Dict[str, Dict[str, object]] = {}

        def ensure_node(name: str) -> None:
            if name not in nodes:
                nodes[name] = {"self": 0.0, "children": set()}

        # 1) Register self times for every measured name.
        for name, secs in self.name_to_total_s.items():
            ensure_node(name)
            nodes[name]["self"] = nodes[name]["self"] + secs  # type: ignore[index]

            # 2) Register parent/child links for dotted names.
            if "." in name:
                parent = name.rsplit(".", 1)[0]
                ensure_node(parent)
                nodes[parent]["children"].add(name)  # type: ignore[index]

        # 3) Roots are names that are never registered as a child.
        all_children = set()
        for info in nodes.values():
            all_children.update(info["children"])  # type: ignore[arg-type]
        roots = [name for name in nodes.keys() if name not in all_children]

        # 4) Aggregate time for each node = self + sum(children).
        aggregate_cache: Dict[str, float] = {}

        def aggregate(name: str) -> float:
            if name in aggregate_cache:
                return aggregate_cache[name]
            info = nodes[name]
            total_self = float(info["self"])  # type: ignore[index]
            child_total = sum(aggregate(child) for child in info["children"])  # type: ignore[index]
            agg = total_self + child_total
            aggregate_cache[name] = agg
            return agg

        # 5) Total runtime is the sum over root aggregates.
        total = sum(aggregate(root) for root in roots)

        lines: List[str] = []
        lines.append(f"Latency Summary (seconds) - Total Time: {total:.3f}s\n")

        def add_children_recursive(name: str, indent_level: int) -> None:
            children = sorted(
                list(nodes[name]["children"]),  # type: ignore[index]
                key=lambda n: -aggregate(n),
            )
            for child in children:
                secs = aggregate(child)
                pct = (secs / total * 100.0) if total > 0 else 0.0
                display_name = child.split(".")[-1]
                indent = "  " * indent_level + "└─ "
                lines.append(f"{indent}{display_name}: {secs:.3f}s ({pct:.2f}%)")
                add_children_recursive(child, indent_level + 1)

        # 6) Print roots sorted by aggregate time.
        for root in sorted(roots, key=lambda n: -aggregate(n)):
            secs = aggregate(root)
            pct = (secs / total * 100.0) if total > 0 else 0.0
            lines.append(f"{root}: {secs:.3f}s ({pct:.2f}%)")
            add_children_recursive(root, 1)

        out_dir = dest_dir if dest_dir is not None else self.base_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "w") as f:
            f.write("\n".join(lines))

