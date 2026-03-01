import os
import time
from contextlib import contextmanager
from typing import Dict, List, Tuple
import torch


class LatencyLogger:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.name_to_total_s: Dict[str, float] = {}
        self.entries: List[Tuple[str, float]] = []
        self._wall_start: float = time.perf_counter()
        self._cpu_start: float = time.process_time()

    @contextmanager
    def timeit(self, name: str, sync_cuda: bool = False):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - start
            self.name_to_total_s[name] = self.name_to_total_s.get(name, 0.0) + dt
            self.entries.append((name, dt))

    def record(self, name: str, seconds: float) -> None:
        self.name_to_total_s[name] = self.name_to_total_s.get(name, 0.0) + seconds
        self.entries.append((name, seconds))

    def write_summary(self, filename: str = "summary.txt") -> None:
        # Split into top-level (no dot) and nested (has dot)
        top_level_categories = {}
        nested_categories = {}
        for name, secs in self.name_to_total_s.items():
            if "." in name:
                nested_categories[name] = secs
            else:
                top_level_categories[name] = secs

        def get_direct_children_names(parent_prefix: str, all_items: Dict[str, float]) -> List[str]:
            """Return the full names of direct children (one segment deeper than parent)."""
            prefix_with_dot = parent_prefix + "." if parent_prefix else ""
            seen: set = set()
            result: List[str] = []
            for name in all_items:
                if not name.startswith(prefix_with_dot):
                    continue
                remaining = name[len(prefix_with_dot):]
                if not remaining:
                    continue
                first_segment = remaining.split(".")[0]
                child_prefix = prefix_with_dot + first_segment
                if child_prefix not in seen:
                    seen.add(child_prefix)
                    result.append(child_prefix)
            return result

        def get_direct_children_sum(parent_prefix: str, all_items: Dict[str, float]) -> float:
            """Sum the recorded time of direct children of parent (one segment deeper).
            Each child's value is taken directly from all_items[child_prefix] (not aggregated
            from descendants), so the hierarchy never double-counts nested measurements."""
            return sum(all_items.get(cp, 0.0) for cp in get_direct_children_names(parent_prefix, all_items))

        def get_direct_children(parent_prefix: str, all_items: Dict[str, float]) -> List[Tuple[str, float]]:
            return [(cp, all_items.get(cp, 0.0)) for cp in get_direct_children_names(parent_prefix, all_items)]

        # For each top-level, use at least the sum of direct children so hierarchy never inverts
        displayed_top_level: Dict[str, float] = {}
        for name, secs in top_level_categories.items():
            children_sum = get_direct_children_sum(name, nested_categories)
            displayed_top_level[name] = max(secs, children_sum)

        total = sum(displayed_top_level.values())

        wall_elapsed = time.perf_counter() - self._wall_start
        cpu_elapsed = time.process_time() - self._cpu_start

        lines: List[str] = []
        lines.append(f"Latency Summary (seconds) - Total Time: {total:.3f}s\n")
        lines.append(f"Wall time: {wall_elapsed:.3f}s | CPU time: {cpu_elapsed:.3f}s\n")

        def add_children_recursive(parent_prefix: str, indent_level: int, all_items: Dict[str, float]):
            children = get_direct_children(parent_prefix, all_items)
            if not children:
                return
            children = sorted(children, key=lambda x: -x[1])
            for child_name, child_secs in children:
                child_pct = (child_secs / total * 100.0) if total > 0 else 0.0
                display_name = child_name.split(".")[-1]
                indent = "  " * indent_level + "└─ "
                lines.append(f"{indent}{display_name}: {child_secs:.3f}s ({child_pct:.2f}%)")
                add_children_recursive(child_name, indent_level + 1, all_items)

        for name, secs in sorted(displayed_top_level.items(), key=lambda x: -x[1]):
            pct = (secs / total * 100.0) if total > 0 else 0.0
            lines.append(f"{name}: {secs:.3f}s ({pct:.2f}%)")
            add_children_recursive(name, 1, nested_categories)

        out_path = os.path.join(self.base_dir, filename)
        with open(out_path, "w") as f:
            f.write("\n".join(lines))




