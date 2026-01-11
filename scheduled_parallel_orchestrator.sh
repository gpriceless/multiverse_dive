#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
INTERVAL_MINUTES=10
DURATION_HOURS=12
LOGFILE="/home/gprice/projects/multiverse_dive/parallel_orchestrator_runs.log"
ORCHESTRATOR_SCRIPT="/home/gprice/projects/multiverse_dive/parallel_orchestrator.sh"

# Calculate total runs
TOTAL_MINUTES=$((DURATION_HOURS * 60))
MAX_RUNS=$((TOTAL_MINUTES / INTERVAL_MINUTES))

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Scheduled Parallel Orchestrator${NC}"
echo -e "${BLUE}   Multi-Track Autonomous Build System${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"
echo -e "Configuration:"
echo -e "  - Run interval: ${YELLOW}${INTERVAL_MINUTES} minutes${NC}"
echo -e "  - Duration: ${YELLOW}${DURATION_HOURS} hours${NC}"
echo -e "  - Maximum runs: ${YELLOW}${MAX_RUNS}${NC}"
echo -e "  - Parallel tracks: ${YELLOW}Up to 4 simultaneous${NC}"
echo -e "  - Log file: ${YELLOW}${LOGFILE}${NC}\n"

# Initialize log file
echo "==================================" > "$LOGFILE"
echo "Scheduled Parallel Orchestrator Run Log" >> "$LOGFILE"
echo "Started: $(date)" >> "$LOGFILE"
echo "==================================" >> "$LOGFILE"
echo "" >> "$LOGFILE"

START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION_HOURS * 3600))
RUN_COUNT=0

while true; do
    CURRENT_TIME=$(date +%s)

    # Check if we've exceeded the duration
    if [ $CURRENT_TIME -ge $END_TIME ]; then
        echo -e "\n${GREEN}✓ Scheduled orchestrator completed!${NC}"
        echo -e "Total runs: ${RUN_COUNT}"
        echo "" >> "$LOGFILE"
        echo "==================================" >> "$LOGFILE"
        echo "Completed: $(date)" >> "$LOGFILE"
        echo "Total runs: ${RUN_COUNT}" >> "$LOGFILE"
        echo "==================================" >> "$LOGFILE"
        break
    fi

    RUN_COUNT=$((RUN_COUNT + 1))
    ELAPSED_MINUTES=$(( (CURRENT_TIME - START_TIME) / 60 ))
    REMAINING_MINUTES=$(( (END_TIME - CURRENT_TIME) / 60 ))

    echo -e "\n${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}[Run ${RUN_COUNT}/${MAX_RUNS}]${NC} $(date)"
    echo -e "Elapsed: ${ELAPSED_MINUTES}m | Remaining: ${REMAINING_MINUTES}m"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

    # Log the run
    echo "----------------------------------------" >> "$LOGFILE"
    echo "Run #${RUN_COUNT} - $(date)" >> "$LOGFILE"
    echo "----------------------------------------" >> "$LOGFILE"

    # Execute the parallel orchestrator
    echo -e "${YELLOW}Launching parallel orchestrator...${NC}\n"

    # Try --resume first (for IN PROGRESS tracks), then without (for PENDING tracks)
    if "$ORCHESTRATOR_SCRIPT" --resume >> "$LOGFILE" 2>&1; then
        echo -e "${GREEN}✓ Parallel orchestrator run #${RUN_COUNT} (resume) completed successfully${NC}"
        echo "Status: SUCCESS (resume mode)" >> "$LOGFILE"
    else
        RESUME_EXIT=$?
        # Check if resume mode found no tracks
        if grep -q "No unclaimed tracks available" "$LOGFILE" | tail -5; then
            echo -e "${YELLOW}No IN PROGRESS tracks, trying NEW tracks...${NC}"
            echo "Resume mode: No IN PROGRESS tracks found, trying NEW mode" >> "$LOGFILE"

            # Try without --resume for PENDING tracks
            if "$ORCHESTRATOR_SCRIPT" >> "$LOGFILE" 2>&1; then
                echo -e "${GREEN}✓ Parallel orchestrator run #${RUN_COUNT} (new) completed successfully${NC}"
                echo "Status: SUCCESS (new mode)" >> "$LOGFILE"
            else
                EXIT_CODE=$?
                echo -e "${RED}✗ Parallel orchestrator run #${RUN_COUNT} failed with exit code ${EXIT_CODE}${NC}"
                echo "Status: FAILED (exit code ${EXIT_CODE})" >> "$LOGFILE"
                echo -e "${YELLOW}Continuing to next scheduled run...${NC}"
            fi
        else
            echo -e "${RED}✗ Parallel orchestrator run #${RUN_COUNT} failed with exit code ${RESUME_EXIT}${NC}"
            echo "Status: FAILED (exit code ${RESUME_EXIT})" >> "$LOGFILE"
            echo -e "${YELLOW}Continuing to next scheduled run...${NC}"
        fi
    fi

    echo "" >> "$LOGFILE"

    # Calculate time until next run
    NEXT_RUN_TIME=$((CURRENT_TIME + INTERVAL_MINUTES * 60))
    WAIT_SECONDS=$((NEXT_RUN_TIME - $(date +%s)))

    # If we still have time for another run
    if [ $NEXT_RUN_TIME -lt $END_TIME ]; then
        echo -e "\n${BLUE}Next run in ${INTERVAL_MINUTES} minutes...${NC}"
        echo -e "${YELLOW}Waiting...${NC}"
        sleep $WAIT_SECONDS
    fi
done

echo -e "\n${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   ✨ All scheduled runs complete! ✨${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"
echo -e "Check ${YELLOW}${LOGFILE}${NC} for detailed logs of all runs."
echo -e "Individual track logs: ${YELLOW}orchestrator_logs/${NC}\n"
