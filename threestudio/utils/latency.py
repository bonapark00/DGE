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

    def write_summary(self, filename: str = "summary.txt") -> None:
        # Calculate total time from top-level categories only (without nested items)
        top_level_categories = {}
        nested_categories = {}
        
        for name, secs in self.name_to_total_s.items():
            if '.' in name:
                # This is a nested item
                nested_categories[name] = secs
            else:
                # This is a top-level category
                top_level_categories[name] = secs
        
        # Total time is sum of only top-level categories (not including nested)
        total = sum(top_level_categories.values())
        
        lines: List[str] = []
        lines.append(f"Latency Summary (seconds) - Total Time: {total:.3f}s\n")
        
        # Sort by total time
        for name, secs in sorted(top_level_categories.items(), key=lambda x: -x[1]):
            pct = (secs / total * 100.0) if total > 0 else 0.0
            lines.append(f"{name}: {secs:.3f}s ({pct:.2f}%)")
            
            # Add nested measurements with proper indentation
            nested_items = [(k, v) for k, v in nested_categories.items() if k.startswith(name + '.')]
            if nested_items:
                for nested_name, nested_secs in sorted(nested_items, key=lambda x: -x[1]):
                    nested_pct = (nested_secs / total * 100.0) if total > 0 else 0.0
                    # Count the number of dots to determine indentation level
                    dot_count = nested_name.count('.')
                    indent = "  " * dot_count + "└─ "
                    display_name = nested_name.split('.', dot_count)[dot_count]
                    lines.append(f"{indent}{display_name}: {nested_secs:.3f}s ({nested_pct:.2f}%)")
        
        out_path = os.path.join(self.base_dir, filename)
        with open(out_path, "w") as f:
            f.write("\n".join(lines))




