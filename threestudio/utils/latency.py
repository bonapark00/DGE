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
        
        def get_direct_children(parent_prefix: str, all_items: Dict[str, float]) -> List[Tuple[str, float]]:
            """Get direct children of a parent (not grandchildren)"""
            children = []
            parent_depth = parent_prefix.count('.') if parent_prefix else 0
            
            for name, secs in all_items.items():
                if name.startswith(parent_prefix + '.'):
                    # Check if this is a direct child (one more level deep)
                    remaining = name[len(parent_prefix) + 1:]  # Remove parent prefix and '.'
                    if '.' not in remaining:  # Direct child has no more dots
                        children.append((name, secs))
            
            return children
        
        def add_children_recursive(parent_prefix: str, indent_level: int, all_items: Dict[str, float]):
            """Recursively add children maintaining hierarchy"""
            children = get_direct_children(parent_prefix, all_items)
            if not children:
                return
            
            # Sort direct children by time
            children = sorted(children, key=lambda x: -x[1])
            
            for child_name, child_secs in children:
                child_pct = (child_secs / total * 100.0) if total > 0 else 0.0
                display_name = child_name.split('.')[-1]  # Get last part after last dot
                indent = "  " * indent_level + "└─ "
                lines.append(f"{indent}{display_name}: {child_secs:.3f}s ({child_pct:.2f}%)")
                
                # Recursively add grandchildren
                add_children_recursive(child_name, indent_level + 1, all_items)
        
        # Sort by total time
        for name, secs in sorted(top_level_categories.items(), key=lambda x: -x[1]):
            pct = (secs / total * 100.0) if total > 0 else 0.0
            lines.append(f"{name}: {secs:.3f}s ({pct:.2f}%)")
            
            # Add nested measurements with proper hierarchy
            add_children_recursive(name, 1, nested_categories)
        
        out_path = os.path.join(self.base_dir, filename)
        with open(out_path, "w") as f:
            f.write("\n".join(lines))




