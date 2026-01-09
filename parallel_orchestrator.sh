#!/bin/bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
WORK_DIR="/home/gprice/projects/multiverse_dive"
LOCK_DIR="${WORK_DIR}/.orchestrator_locks"
LOG_DIR="${WORK_DIR}/orchestrator_logs"
SINGLE_ORCHESTRATOR="${WORK_DIR}/orchestrator.sh"

# Check for resume flag
RESUME_MODE=false
if [ "$1" = "--resume" ] || [ "$1" = "-r" ]; then
    RESUME_MODE=true
    echo -e "${YELLOW}Resume mode enabled - continuing from IN PROGRESS tracks${NC}\n"
fi

mkdir -p "$LOCK_DIR"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
if [ "$RESUME_MODE" = true ]; then
    echo -e "${BLUE}   Parallel Orchestrator - RESUME MODE${NC}"
else
    echo -e "${BLUE}   Parallel Orchestrator - Multi-Track Builder${NC}"
fi
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

# Function to identify parallel tracks in current group
identify_parallel_tracks() {
    RESUME=$1
    python3 << EOF
import re
import sys

roadmap_path = "/home/gprice/projects/multiverse_dive/ROADMAP.md"
resume_mode = "$RESUME" == "true"

with open(roadmap_path, 'r') as f:
    content = f.read()

# Find the first group that's not marked as DONE
group_sections = re.split(r'### \*\*Group [A-Z]:', content)[1:]

for i, section in enumerate(group_sections):
    group_letter = chr(ord('A') + i)

    # Check if this group is done
    if '[DONE]' in section.split('\n')[0]:
        continue

    # This is the current group - find parallel tracks
    track_matches = re.findall(r'\*\*Track (\d+):[^*]+\*\*[^\n]*', section)

    if track_matches:
        print(f"GROUP:{group_letter}")

        # For each track, check status
        for track_num in track_matches:
            track_pattern = rf'\*\*Track {track_num}:[^*]+\*\*([^\n]*)'
            track_line = re.search(track_pattern, section)
            if track_line:
                track_text = track_line.group(0)
                is_done = '[DONE]' in track_text.upper()
                is_in_progress = 'IN PROGRESS' in track_text.upper() or 'WORKING' in track_text.upper() or '🔄' in track_text

                if resume_mode:
                    # In resume mode, pick up IN PROGRESS tracks
                    if is_in_progress and not is_done:
                        print(f"TRACK:{track_num}:RESUME")
                else:
                    # Normal mode, skip IN PROGRESS and DONE
                    if not is_done and not is_in_progress:
                        print(f"TRACK:{track_num}:NEW")
        break
EOF
}

# Get available tracks
echo -e "${CYAN}Analyzing ROADMAP.md...${NC}"
TRACK_INFO=$(identify_parallel_tracks "$RESUME_MODE")

if [ -z "$TRACK_INFO" ]; then
    echo -e "${YELLOW}No tracks found${NC}"
    if [ "$RESUME_MODE" = true ]; then
        echo -e "${YELLOW}No IN PROGRESS tracks to resume. Try without --resume flag.${NC}"
    else
        echo -e "${YELLOW}All tracks complete or in progress. Use --resume to continue.${NC}"
    fi
    exit 0
fi

CURRENT_GROUP=$(echo "$TRACK_INFO" | grep "GROUP:" | cut -d: -f2)
AVAILABLE_TRACKS=($(echo "$TRACK_INFO" | grep "TRACK:" | cut -d: -f2))
TRACK_MODES=($(echo "$TRACK_INFO" | grep "TRACK:" | cut -d: -f3))

echo -e "${GREEN}Current Group: ${CURRENT_GROUP}${NC}"
echo -e "${GREEN}Available Tracks: ${#AVAILABLE_TRACKS[@]}${NC}"
if [ "$RESUME_MODE" = true ]; then
    echo -e "${YELLOW}Mode: RESUMING IN PROGRESS tracks${NC}"
else
    echo -e "Mode: NEW tracks"
fi
echo -e "Tracks: ${YELLOW}${AVAILABLE_TRACKS[@]}${NC}\n"

if [ ${#AVAILABLE_TRACKS[@]} -eq 0 ]; then
    echo -e "${YELLOW}No unclaimed tracks available${NC}"
    exit 0
fi

# Determine how many to run in parallel
NUM_PARALLEL=${#AVAILABLE_TRACKS[@]}
if [ $NUM_PARALLEL -gt 4 ]; then
    NUM_PARALLEL=4
    echo -e "${YELLOW}Limiting to 4 parallel tracks (${#AVAILABLE_TRACKS[@]} available)${NC}\n"
fi

echo -e "${MAGENTA}Launching ${NUM_PARALLEL} orchestrators in parallel...${NC}\n"

# Launch orchestrators for each track
PIDS=()
for i in $(seq 0 $((NUM_PARALLEL - 1))); do
    TRACK=${AVAILABLE_TRACKS[$i]}
    MODE=${TRACK_MODES[$i]}
    LOGFILE="${LOG_DIR}/group_${CURRENT_GROUP}_track_${TRACK}.log"

    if [ "$MODE" = "RESUME" ]; then
        echo -e "${CYAN}[Orchestrator $((i+1))/${NUM_PARALLEL}]${NC} ${YELLOW}RESUMING${NC} Group ${CURRENT_GROUP}, Track ${TRACK}"
    else
        echo -e "${CYAN}[Orchestrator $((i+1))/${NUM_PARALLEL}]${NC} Starting on Group ${CURRENT_GROUP}, Track ${TRACK}"
    fi
    echo -e "  Log: ${YELLOW}${LOGFILE}${NC}"

    # Create a lock file to claim this track
    LOCK_FILE="${LOCK_DIR}/group_${CURRENT_GROUP}_track_${TRACK}.lock"
    echo "$$" > "$LOCK_FILE"

    # Launch orchestrator with specific track assignment
    (
        cd "$WORK_DIR"
        export TARGET_GROUP="$CURRENT_GROUP"
        export TARGET_TRACK="$TRACK"
        export RESUME_MODE="$MODE"

        # Append to existing log if resuming
        if [ "$MODE" = "RESUME" ]; then
            echo "" >> "$LOGFILE"
            echo "═══════════════════════════════════════════════════════" >> "$LOGFILE"
            echo "RESUMING - $(date)" >> "$LOGFILE"
            echo "═══════════════════════════════════════════════════════" >> "$LOGFILE"
            echo "" >> "$LOGFILE"
        fi

        # Modified orchestrator call that targets specific track
        claude --dangerously-skip-permissions --print "
🔧 PARALLEL WORK AGENT - Group ${CURRENT_GROUP}, Track ${TRACK} $([ "$MODE" = "RESUME" ] && echo "(RESUMING)")

$([ "$MODE" = "RESUME" ] && echo "⚡ RESUME MODE: You are continuing work on this track that was previously started." || echo "IMPORTANT: You are one of ${NUM_PARALLEL} agents working in parallel on Group ${CURRENT_GROUP}.")

Your assigned track is: Track ${TRACK}

$([ "$MODE" = "RESUME" ] && echo "RESUME INSTRUCTIONS:
- Check what was already implemented for this track
- Continue from where it left off
- If implementation is complete, move to review/test phase
- If review is complete, move to commit phase
- Announce: echo '[Track ${TRACK}] RESUMING - checking progress...'" || echo "")

Throughout your work, announce your progress with your track number:
echo '[Track ${TRACK}] Reading ROADMAP.md...'
echo '[Track ${TRACK}] Checking FIXES.md for related bugs...'
echo '[Track ${TRACK}] Found task: [task name]'
echo '[Track ${TRACK}] Fixing [FIX-XXX]...'
echo '[Track ${TRACK}] Implementing [component]...'
echo '[Track ${TRACK}] Writing [filename]...'

Your mission:

STEP 0: Check FIXES.md for bugs
   - Read FIXES.md to see documented bugs
   - Identify P0/P1 bugs in files related to Track ${TRACK}
   - Fix these bugs AS YOU WORK on related code
   - Update FIXES.md to mark bugs as [FIXED]

STEP 1: Read ROADMAP.md and find Group ${CURRENT_GROUP}, Track ${TRACK}

STEP 2: $([ "$MODE" = "RESUME" ] && echo "Check what's already done, continue from there" || echo "Mark Track ${TRACK} as 'IN PROGRESS' in ROADMAP.md")

STEP 3: READ EXISTING CODE before writing new code
   - Understand patterns used in related modules
   - Reference larger code sets as you expand

STEP 4: Implement ONLY the items in Track ${TRACK}
   - Fix any FIXES.md bugs in files you touch
   - Do NOT work on other tracks

STEP 5: Update FIXES.md for any bugs you fixed
   - Mark with [FIXED] and brief note

When done: echo '[Track ${TRACK}] Work Agent complete - [what you built] + [N bugs fixed]'

Work efficiently. Reference existing code. Fix bugs as you go!
        " >> "$LOGFILE" 2>&1

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "[Track ${TRACK}] Phase 1 completed successfully" >> "$LOGFILE"

            # Phase 2: Review & Test
            claude --dangerously-skip-permissions --print "
🔍 PARALLEL REVIEW AGENT - Group ${CURRENT_GROUP}, Track ${TRACK}

You are reviewing Track ${TRACK} from Group ${CURRENT_GROUP}.

Announce progress with: echo '[Track ${TRACK}] [action]'

Your mission:

STEP 1: Read the Code Review Checklist from ROADMAP.md
   - Find 'Agent Code Review Checklist' section
   - This checklist is MANDATORY

STEP 2: Review Track ${TRACK} code against the checklist
   - Correctness & Safety (division guards, bounds checks, NaN handling)
   - Consistency (names match across files, defaults match)
   - Completeness (all features implemented, docstrings, type hints)
   - Robustness (specific exceptions, thread safety)
   - Performance (no O(n²) loops, caching)
   - Security (input validation, no secrets logged)
   - Maintainability (no magic numbers, no duplication)

STEP 3: Check FIXES.md for remaining bugs
   - Read FIXES.md for bugs in Track ${TRACK} files
   - Fix any P0/P1 bugs in code you're reviewing
   - Update FIXES.md to mark bugs as [FIXED]
   - Announce: '[Track ${TRACK}] Fixed [FIX-XXX]'

STEP 4: Write comprehensive tests
   - Unit tests, integration tests, edge cases
   - Run: PYTHONPATH=. .venv/bin/pytest tests/ -v

STEP 5: Announce completion
   echo '[Track ${TRACK}] Review complete - [X] tests passing, [Y] bugs fixed'

Do NOT modify code from other tracks.
            " >> "$LOGFILE" 2>&1

            if [ $? -eq 0 ]; then
                echo "[Track ${TRACK}] Phase 2 completed successfully" >> "$LOGFILE"

                # Phase 3: Commit (but don't push yet - we'll push all together)
                claude --dangerously-skip-permissions --print "
📝 PARALLEL COMMIT AGENT - Group ${CURRENT_GROUP}, Track ${TRACK}

You are finalizing Track ${TRACK} from Group ${CURRENT_GROUP}.

Announce progress with: echo '[Track ${TRACK}] [action]'

Your mission:
1. Review changes for Track ${TRACK}
2. Update ROADMAP.md to mark Track ${TRACK} as 'DONE'
3. Create a git commit (but DO NOT push yet):

   Commit message format:
   'Implement [Feature] - Group ${CURRENT_GROUP}, Track ${TRACK}

   - Description of what was added
   - Key components/files
   - Bugs fixed: [FIX-XXX, FIX-YYY] (if any)
   - Test results

   Co-Authored-By: Claude <noreply@anthropic.com>'

4. Announce: '[Track ${TRACK}] Committed (waiting for group sync before push)'

IMPORTANT: Do NOT push to GitHub yet. We'll push all tracks together.
                " >> "$LOGFILE" 2>&1

                echo "[Track ${TRACK}] Phase 3 completed successfully" >> "$LOGFILE"
            fi
        fi

        # Clean up lock file
        rm -f "$LOCK_FILE"

        exit $EXIT_CODE
    ) &

    PID=$!
    PIDS+=($PID)
    echo -e "  ${GREEN}Started (PID: ${PID})${NC}\n"

    sleep 2  # Stagger starts slightly
done

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}All ${NUM_PARALLEL} orchestrators launched!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

echo -e "Monitor progress:"
for i in $(seq 0 $((NUM_PARALLEL - 1))); do
    TRACK=${AVAILABLE_TRACKS[$i]}
    LOGFILE="${LOG_DIR}/group_${CURRENT_GROUP}_track_${TRACK}.log"
    echo -e "  Track ${TRACK}: ${YELLOW}tail -f ${LOGFILE}${NC}"
done
echo ""

# Wait for all to complete
echo -e "${CYAN}Waiting for all orchestrators to complete...${NC}\n"

FAILED=0
for i in $(seq 0 $((NUM_PARALLEL - 1))); do
    PID=${PIDS[$i]}
    TRACK=${AVAILABLE_TRACKS[$i]}

    echo -e "${CYAN}[Track ${TRACK}]${NC} Waiting for PID ${PID}..."

    if wait $PID; then
        echo -e "${GREEN}✓ Track ${TRACK} completed successfully${NC}"
    else
        echo -e "${RED}✗ Track ${TRACK} failed${NC}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}   ✨ All tracks completed successfully! ✨${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

    echo -e "${CYAN}Pushing all changes to GitHub...${NC}"
    cd "$WORK_DIR"

    if git push origin main 2>&1; then
        echo -e "${GREEN}✓ All changes pushed to GitHub${NC}\n"
    else
        echo -e "${RED}✗ Push failed - you may need to configure authentication${NC}"
        echo -e "${YELLOW}Run: ./setup_github_auth.sh${NC}\n"
    fi
else
    echo -e "${RED}${FAILED} track(s) failed${NC}"
    echo -e "${YELLOW}Check individual log files for details${NC}\n"
    exit 1
fi
