# -*- coding: utf-8 -*-
# tools_runcode.py
# Simple Python script executor for workspace structure
from __future__ import annotations

import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def run_workspace_scripts(workspace_path: str, python_executor: str = "python3") -> None:
    """
    Run all Python scripts in workspace structure in order: data -> information -> knowledge -> wisdom

    Args:
        workspace_path: Path to workspace directory containing code/
        python_executor: Python command to use (e.g. "python3", "docker exec container python3")
    """
    workspace = Path(workspace_path).absolute()
    code_dir = workspace / "code"

    # Execution order
    categories = ["data", "information", "knowledge", "wisdom"]

    logger.info(f"Starting script execution in: {workspace}")
    logger.info(f"Python executor: {python_executor}")

    total_success = 0
    total_files = 0

    for category in categories:
        category_dir = code_dir / category

        if not category_dir.exists():
            logger.warning(f"Directory {category_dir} does not exist, skipping...")
            continue

        # Find all .py files and sort by name
        py_files = sorted(category_dir.glob("*.py"))

        if not py_files:
            logger.info(f"No Python files found in {category}/")
            continue

        logger.info(f"\n📂 Processing {category}/ ({len(py_files)} files)")

        for py_file in py_files:
            total_files += 1

            # Build command
            if python_executor.startswith("docker"):
                # Convert to container path if using Docker
                container_path = f"/workspace/code/{category}/{py_file.name}"
                cmd = python_executor.split() + [container_path]
            else:
                # Local execution
                cmd = python_executor.split() + [str(py_file)]

            logger.info(f"  Running: {py_file.name}")
            logger.debug(f"  Command: {' '.join(cmd)}")

            try:
                # Execute the script
                result = subprocess.run(
                    cmd,
                    cwd=workspace,  # Set working directory
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )

                if result.returncode == 0:
                    logger.info(f"  ✅ SUCCESS: {py_file.name}")
                    total_success += 1
                else:
                    logger.error(f"  ❌ FAILED: {py_file.name} (exit code: {result.returncode})")
                    if result.stderr:
                        logger.error(f"    Error: {result.stderr.strip()}")

            except subprocess.TimeoutExpired:
                logger.error(f"  ⏰ TIMEOUT: {py_file.name} (exceeded 10 minutes)")
            except Exception as e:
                logger.error(f"  💥 ERROR: {py_file.name} - {str(e)}")

    # Summary
    logger.info(f"\n🎯 Execution Complete!")
    logger.info(f"📊 {total_success}/{total_files} scripts executed successfully")
    if total_success == total_files and total_files > 0:
        logger.info("🎉 All scripts completed successfully!")


def run_workspace_docker(workspace_path: str, container_name: str) -> None:
    """
    Convenience function to run scripts in a Docker container

    Args:
        workspace_path: Local path to workspace
        container_name: Docker container name
    """
    docker_cmd = f"docker exec -w /workspace {container_name} python3"
    run_workspace_scripts(workspace_path, docker_cmd)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tools_runcode.py <workspace_path> [python_executor]")
        print("\nExamples:")
        print("  python tools_runcode.py ./projspace/proj_demo/")
        print("  python tools_runcode.py ./projspace/proj_demo/ python3")
        print("  python tools_runcode.py ./projspace/proj_demo/ 'docker exec my_container python3'")
        sys.exit(1)

    workspace = sys.argv[1]
    executor = sys.argv[2] if len(sys.argv) > 2 else "python3"

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Run all scripts
    run_workspace_scripts(workspace, executor)