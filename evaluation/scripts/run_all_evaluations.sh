#!/bin/bash
#
# Master Script: Run All Evaluation Scripts
#
# This script executes all evaluation scripts in sequence:
# 0. Convert Result to model-task-instance-score format
# 1. Describe model result data quality
# 2. Generate Table1 and Table2
# 3. Generate all figures (R1, R2, R3, R4)
#
# Usage:
#   ./run_all_evaluations.sh                 # Run all scripts (continues on error)
#   ./run_all_evaluations.sh --stop-on-error # Stop at first failure
#   ./run_all_evaluations.sh --from 2        # Run from step 2 onwards
#   ./run_all_evaluations.sh --only 3        # Run only step 3 (all R figures)
#   ./run_all_evaluations.sh --skip 0        # Skip step 0
#

# Continue on error by default (can be changed with --stop-on-error)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Parse arguments
CONTINUE_ON_ERROR=1  # Default: continue on error
SKIP_STEPS=()
ONLY_STEP=""
FROM_STEP=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --stop-on-error)
            CONTINUE_ON_ERROR=0
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=1
            shift
            ;;
        --skip)
            shift
            SKIP_STEPS+=("$1")
            shift
            ;;
        --only)
            shift
            ONLY_STEP="$1"
            shift
            ;;
        --from)
            shift
            FROM_STEP="$1"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --stop-on-error        Stop execution if any script fails (default: continue)"
            echo "  --continue-on-error    Continue execution even if a script fails (default)"
            echo "  --skip STEP            Skip a specific step (e.g., --skip 0)"
            echo "  --only STEP            Run only scripts from this step (e.g., --only 3)"
            echo "  --from STEP            Run from this step onwards (e.g., --from 2)"
            echo "  --dry-run              Show which scripts would run without executing"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                             # Run all scripts (continues on error by default)"
            echo "  $0 --stop-on-error             # Stop at first failure"
            echo "  $0 --skip 0                    # Skip step 0"
            echo "  $0 --only 3                    # Run only step 3 (all R figures)"
            echo "  $0 --from 2                    # Run from step 2 onwards"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to print headers
print_header() {
    local text="$1"
    local char="${2:-=}"
    local width=80

    echo ""
    printf '%*s\n' "$width" | tr ' ' "$char"
    printf "%*s\n" $(((${#text}+$width)/2)) "$text"
    printf '%*s\n' "$width" | tr ' ' "$char"
    echo ""
}

# Function to print step header
print_step_header() {
    local step="$1"
    local description="$2"

    echo ""
    echo "================================================================================"
    echo -e "${BLUE}STEP $step: $description${NC}"
    echo "================================================================================"
    echo ""
}

# Function to check if step should be skipped
should_skip() {
    local step="$1"

    # Check --skip
    for skip_step in "${SKIP_STEPS[@]}"; do
        if [[ "$step" == "$skip_step"* ]]; then
            return 0  # Should skip
        fi
    done

    # Check --only
    if [[ -n "$ONLY_STEP" ]]; then
        if [[ "$step" != "$ONLY_STEP"* ]]; then
            return 0  # Should skip
        fi
    fi

    # Check --from
    if [[ -n "$FROM_STEP" ]]; then
        local step_num="${step%%-*}"  # Get number before hyphen
        if [[ "$step_num" -lt "$FROM_STEP" ]]; then
            return 0  # Should skip
        fi
    fi

    return 1  # Should not skip
}

# Function to run a script
run_script() {
    local step="$1"
    local script_name="$2"
    local description="$3"
    local script_path="$SCRIPT_DIR/$script_name"

    print_step_header "$step" "$description"

    if [[ ! -f "$script_path" ]]; then
        echo -e "${RED}ERROR: Script not found: $script_path${NC}"
        return 1
    fi

    echo "Running: $script_name"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "--------------------------------------------------------------------------------"

    local start_time=$(date +%s)

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY RUN] Would execute: python $script_path"
        return 0
    fi

    # Run the script
    if python "$script_path"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        echo "--------------------------------------------------------------------------------"
        echo -e "${GREEN}✓ SUCCESS: Step $step completed in ${duration}s${NC}"
        echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local exit_code=$?

        echo "--------------------------------------------------------------------------------"
        echo -e "${RED}✗ FAILED: Step $step failed after ${duration}s (exit code: $exit_code)${NC}"
        return 1
    fi
}

# Main execution
print_header "EVENTGLUCOSE EVALUATION PIPELINE"

echo "Scripts directory: $SCRIPT_DIR"
echo "Python: $(which python)"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Define all scripts
declare -a STEPS=(
    "0|0-convert-Result-to-model-task-instance-score.py|Convert Result to model-task-instance-score format"
    "1|1-describe-model-result-data-quality.py|Describe model result data quality"
    "2|2-generate-Table1-and-Table2.py|Generate Table1 and Table2"
    "3-R1|3-R1-llm-vs-baselines-by-context-FigureR1.py|Generate FigureR1 (LLM vs Baselines by Context)"
    "3-R2|3-R2-llm-with-without-context-FigureR2.py|Generate FigureR2 (LLM With/Without Context)"
    "3-R3|3-R3-Parameter-Performance-FigureR3.py|Generate FigureR3 (Parameter vs Performance)"
    "3-R4|3-R4-instance-group-analysis-FigureR4.py|Generate FigureR4 (Instance Group Analysis)"
    "5|5-democompare.py|Generate FigureR5 (Demo Compare - Context Level Comparison)"
)

# Collect scripts to run
declare -a SCRIPTS_TO_RUN=()

for step_info in "${STEPS[@]}"; do
    IFS='|' read -r step script description <<< "$step_info"

    if should_skip "$step"; then
        echo -e "${YELLOW}⊘ Skipping step $step: $description${NC}"
    else
        SCRIPTS_TO_RUN+=("$step_info")
    fi
done

if [[ ${#SCRIPTS_TO_RUN[@]} -eq 0 ]]; then
    echo "No scripts to run based on the provided filters."
    exit 0
fi

echo ""
echo "Scripts to run: ${#SCRIPTS_TO_RUN[@]}"
for step_info in "${SCRIPTS_TO_RUN[@]}"; do
    IFS='|' read -r step script description <<< "$step_info"
    echo "  • Step $step: $script"
done

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "Dry run - no scripts will be executed."
    echo ""
fi

# Run all scripts
total_start=$(date +%s)
success_count=0
fail_count=0
declare -a RESULTS=()

for step_info in "${SCRIPTS_TO_RUN[@]}"; do
    IFS='|' read -r step script description <<< "$step_info"

    if run_script "$step" "$script" "$description"; then
        ((success_count++))
        RESULTS+=("$step|SUCCESS|$description")
    else
        ((fail_count++))
        RESULTS+=("$step|FAILED|$description")

        if [[ $CONTINUE_ON_ERROR -eq 0 ]]; then
            echo ""
            echo -e "${RED}✗ Stopping execution due to failure in step $step${NC}"
            break
        fi
    fi
done

total_end=$(date +%s)
total_duration=$((total_end - total_start))

# Print summary
print_header "EXECUTION SUMMARY" "="

echo "Total execution time: ${total_duration}s"
echo "Scripts executed: ${#RESULTS[@]}"
echo -e "${GREEN}Successful: $success_count${NC}"
if [[ $fail_count -gt 0 ]]; then
    echo -e "${RED}Failed: $fail_count${NC}"
else
    echo "Failed: $fail_count"
fi

echo ""
echo "Detailed Results:"
echo "--------------------------------------------------------------------------------"
printf "%-10s %-10s %s\n" "Step" "Status" "Description"
echo "--------------------------------------------------------------------------------"

for result in "${RESULTS[@]}"; do
    IFS='|' read -r step status description <<< "$result"

    if [[ "$status" == "SUCCESS" ]]; then
        printf "%-10s ${GREEN}%-10s${NC} %s\n" "$step" "✓ SUCCESS" "$description"
    else
        printf "%-10s ${RED}%-10s${NC} %s\n" "$step" "✗ FAILED" "$description"
    fi
done

echo "--------------------------------------------------------------------------------"

if [[ $fail_count -gt 0 ]]; then
    echo ""
    echo -e "${YELLOW}⚠ Warning: $fail_count script(s) failed${NC}"
    echo ""
    echo "Failed steps:"
    for result in "${RESULTS[@]}"; do
        IFS='|' read -r step status description <<< "$result"
        if [[ "$status" == "FAILED" ]]; then
            echo "  • Step $step: $description"
        fi
    done
    exit 1
else
    echo ""
    echo -e "${GREEN}✓ All scripts completed successfully!${NC}"
    exit 0
fi
