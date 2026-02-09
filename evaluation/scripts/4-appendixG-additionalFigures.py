"""
Appendix Additional Figures

This script copies non-main metric figures from R1-R4 scripts to the AppendixFigure folder.
The main metric figures go to the Figure folder (handled by the R scripts themselves).
Non-main metric figures are renamed with G- prefix and copied to AppendixFigure.

Source folders:

"""

import os
import sys
import shutil
from pathlib import Path
import yaml

TOPIC_NAME = '4-appendix-additionalFigures'

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

def load_config(config_path: Path) -> dict:
    copied_files = []
    missing_files = []

    for r_name, source_dir in SOURCE_DIRS.items():
        print(f"\n{'='*80}")
        print(f"Processing {r_name}: {source_dir.name}")
        print("=" * 80)

        if not source_dir.exists():
            print(f"  Warning: Source directory not found: {source_dir}")
            continue

        patterns = FIGURE_PATTERNS.get(r_name, [])

        for pattern in patterns:
            for metric in NON_MAIN_METRICS:
                source_file = source_dir / f"{pattern}_{metric}.pdf"

                if source_file.exists():
                    dest_name = f"G-{pattern}_{metric}.pdf"
                    dest_file = appendix_figure_dir / dest_name

                    copied_files.append(dest_name)
                    print(f"  Copied: {source_file.name} -> {dest_name}")
                else:
                    missing_files.append(f"{pattern}_{metric}.pdf")
                    print(f"  Missing: {source_file.name}")

    return copied_files, missing_files

# =============================================================================
# MAIN EXECUTION
# =============================================================================

print("\n" + "=" * 80)
print("COPYING NON-MAIN METRIC FIGURES")
print("=" * 80)

copied_files, missing_files = copy_non_main_figures()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nCopied {len(copied_files)} files to: {appendix_figure_dir}")

if copied_files:
    print(f"\nCopied files:")
    for f in sorted(copied_files):
        print(f"  {f}")

if missing_files:
    print(f"\nMissing files ({len(missing_files)}):")
    for f in sorted(set(missing_files)):
        print(f"  {f}")

print(f"\nAll G- files in AppendixFigure:")
g_files = sorted(appendix_figure_dir.glob("G-*.pdf"))
for f in g_files:
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name} ({size_kb:.1f} KB)")

print("\n" + "=" * 80)
print("Script completed!")
print("=" * 80)

if __name__ == '__main__':
    print("\nScript execution completed!")
