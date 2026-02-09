"""
EventGlucose CLI Scripts

This package contains command-line tools for the EventGlucose benchmark:

- remote_sync: Sync data between local workspace and remote storage (S3/Google Drive)
- run_baselines: Run baseline experiments from JSON specifications
- run_individual: Run individual task/method experiments
- compile_stderr_results: Compile and analyze stderr results

These scripts are exposed as CLI commands when the package is installed:
- eglu-remote-sync / hai-remote-sync
- eglu-run
- eglu-run-individual
- eglu-compile-results
"""

__version__ = "0.1.0"
