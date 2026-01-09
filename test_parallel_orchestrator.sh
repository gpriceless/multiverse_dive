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

mkdir -p "$LOCK_DIR"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   TEST Parallel Orchestrator (Dry Run)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

# Check GitHub authentication
echo -e "${CYAN}Checking GitHub authentication...${NC}"
if git ls-remote origin &>/dev/null; then
    echo -e "${GREEN}✓ GitHub authentication is configured${NC}"
    DRY_RUN=false
else
    echo -e "${YELLOW}⚠ GitHub authentication not configured${NC}"
    echo -e "${YELLOW}  This will be a DRY RUN - no push to GitHub${NC}"
    echo -e "${YELLOW}  Run ./setup_github_auth.sh to configure${NC}\n"
    DRY_RUN=true
fi

echo ""

# Function to identify parallel tracks in current group
identify_parallel_tracks() {
    python3 << 'EOF'
import re
import sys

roadmap_path = "/home/gprice/projects/multiverse_dive/ROADMAP.md"

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

        # For each track, check if it's completed or in progress
        for track_num in track_matches:
            track_pattern = rf'\*\*Track {track_num}:[^*]+\*\*([^\n]*)'
            track_line = re.search(track_pattern, section)
            if track_line:
                track_text = track_line.group(0)
                # Check if marked as done or in progress
                if '[DONE]' not in track_text.upper() and 'IN PROGRESS' not in track_text.upper() and 'WORKING' not in track_text.upper():
                    print(f"TRACK:{track_num}")
        break
EOF
}

# Get available tracks
echo -e "${CYAN}Analyzing ROADMAP.md...${NC}"
TRACK_INFO=$(identify_parallel_tracks)

if [ -z "$TRACK_INFO" ]; then
    echo -e "${YELLOW}No parallel tracks found or all work is complete${NC}"
    exit 0
fi

CURRENT_GROUP=$(echo "$TRACK_INFO" | grep "GROUP:" | cut -d: -f2)
AVAILABLE_TRACKS=($(echo "$TRACK_INFO" | grep "TRACK:" | cut -d: -f2))

echo -e "${GREEN}Current Group: ${CURRENT_GROUP}${NC}"
echo -e "${GREEN}Available Tracks: ${#AVAILABLE_TRACKS[@]}${NC}"
echo -e "Tracks: ${YELLOW}${AVAILABLE_TRACKS[@]}${NC}\n"

if [ ${#AVAILABLE_TRACKS[@]} -eq 0 ]; then
    echo -e "${YELLOW}No unclaimed tracks available${NC}"
    exit 0
fi

# For testing, limit to 2 parallel tracks
NUM_PARALLEL=${#AVAILABLE_TRACKS[@]}
if [ $NUM_PARALLEL -gt 2 ]; then
    NUM_PARALLEL=2
    echo -e "${YELLOW}TEST MODE: Limiting to 2 parallel tracks (${#AVAILABLE_TRACKS[@]} available)${NC}\n"
fi

echo -e "${MAGENTA}TEST: Launching ${NUM_PARALLEL} orchestrators in parallel...${NC}\n"

read -p "Press ENTER to continue or Ctrl+C to cancel..."
echo ""

# Launch orchestrators for each track
PIDS=()
for i in $(seq 0 $((NUM_PARALLEL - 1))); do
    TRACK=${AVAILABLE_TRACKS[$i]}
    LOGFILE="${LOG_DIR}/test_group_${CURRENT_GROUP}_track_${TRACK}.log"

    echo -e "${CYAN}[Orchestrator $((i+1))/${NUM_PARALLEL}]${NC} Starting on Group ${CURRENT_GROUP}, Track ${TRACK}"
    echo -e "  Log: ${YELLOW}${LOGFILE}${NC}"

    # Create a lock file to claim this track
    LOCK_FILE="${LOCK_DIR}/test_group_${CURRENT_GROUP}_track_${TRACK}.lock"
    echo "$$" > "$LOCK_FILE"

    # Launch orchestrator with specific track assignment
    (
        cd "$WORK_DIR"
        export TARGET_GROUP="$CURRENT_GROUP"
        export TARGET_TRACK="$TRACK"

        echo "═══════════════════════════════════════════════════════" > "$LOGFILE"
        echo "TEST Parallel Orchestrator - Group ${CURRENT_GROUP}, Track ${TRACK}" >> "$LOGFILE"
        echo "Started: $(date)" >> "$LOGFILE"
        echo "═══════════════════════════════════════════════════════" >> "$LOGFILE"
        echo "" >> "$LOGFILE"

        # Phase 1: Work Agent
        echo "Phase 1: Work Agent" >> "$LOGFILE"
        echo "-------------------" >> "$LOGFILE"
        claude --dangerously-skip-permissions --print "
🔧 TEST PARALLEL WORK AGENT - Group ${CURRENT_GROUP}, Track ${TRACK}

This is a TEST RUN. You are one of ${NUM_PARALLEL} agents working in parallel.

Your assigned track: Track ${TRACK} of Group ${CURRENT_GROUP}

IMPORTANT Instructions:
1. Throughout your work, announce progress with:
   echo '[Track ${TRACK}] [what you are doing]'

2. Your tasks:
   - Read ROADMAP.md and locate Group ${CURRENT_GROUP}, Track ${TRACK}
   - Identify what needs to be implemented for this track
   - Mark Track ${TRACK} as 'IN PROGRESS' in ROADMAP.md
   - Implement ONLY the items listed in Track ${TRACK}
   - Do NOT work on other tracks
   - Announce each file you create/modify

3. When complete, announce:
   echo '[Track ${TRACK}] Work complete - [summary of what you built]'

Work efficiently. Other agents are handling other tracks in parallel.
        " >> "$LOGFILE" 2>&1

        EXIT_CODE=$?
        echo "" >> "$LOGFILE"
        echo "Phase 1 exit code: $EXIT_CODE" >> "$LOGFILE"

        if [ $EXIT_CODE -eq 0 ]; then
            echo "[Track ${TRACK}] Phase 1 completed successfully" >> "$LOGFILE"

            # Phase 2: Review & Test
            echo "" >> "$LOGFILE"
            echo "Phase 2: Review & Test Agent" >> "$LOGFILE"
            echo "----------------------------" >> "$LOGFILE"
            claude --dangerously-skip-permissions --print "
🔍 TEST PARALLEL REVIEW AGENT - Group ${CURRENT_GROUP}, Track ${TRACK}

Reviewing Track ${TRACK} from Group ${CURRENT_GROUP}.

STEP 1: Read the Code Review Checklist from ROADMAP.md
   - Find the 'Agent Code Review Checklist' section
   - This checklist is MANDATORY for all reviews

STEP 2: Review Track ${TRACK} code against the checklist:

   CORRECTNESS & SAFETY:
   - [ ] Division operations guarded against zero
   - [ ] Array indexing validated for bounds
   - [ ] NaN/Inf handling with np.isnan(), np.isinf()
   - [ ] Edge cases handled
   - [ ] Shell variables quoted
   - [ ] No hardcoded credentials

   CONSISTENCY:
   - [ ] Names match across files (YAML class names match Python)
   - [ ] Default values in code match YAML/spec defaults

   COMPLETENESS:
   - [ ] All declared features implemented
   - [ ] Every public function has at least one test
   - [ ] Docstrings and type hints present

   ROBUSTNESS:
   - [ ] Specific exceptions caught (no bare except:)
   - [ ] Thread safety for global state

   PERFORMANCE:
   - [ ] No O(n²) loops on large data
   - [ ] Expensive computations cached

   SECURITY:
   - [ ] External input validated
   - [ ] Secrets not logged

   MAINTAINABILITY:
   - [ ] Magic numbers extracted to constants
   - [ ] No code duplication

STEP 3: Fix any issues found
   - Check Tech Debt Backlog in ROADMAP.md for known Critical issues

STEP 4: Write comprehensive tests

STEP 5: Run tests and verify they pass

STEP 6: Announce: '[Track ${TRACK}] Review complete - [X] tests passing, checklist verified'

Do NOT modify code from other tracks.
            " >> "$LOGFILE" 2>&1

            EXIT_CODE=$?
            echo "" >> "$LOGFILE"
            echo "Phase 2 exit code: $EXIT_CODE" >> "$LOGFILE"

            if [ $EXIT_CODE -eq 0 ]; then
                echo "[Track ${TRACK}] Phase 2 completed successfully" >> "$LOGFILE"

                # Phase 3: Prepare commit (but don't actually commit in test mode)
                echo "" >> "$LOGFILE"
                echo "Phase 3: Documentation & Commit Prep" >> "$LOGFILE"
                echo "------------------------------------" >> "$LOGFILE"
                claude --dangerously-skip-permissions --print "
📝 TEST PARALLEL COMMIT PREP - Group ${CURRENT_GROUP}, Track ${TRACK}

Preparing Track ${TRACK} from Group ${CURRENT_GROUP} for commit.

Your tasks:
1. Review what was implemented and tested for Track ${TRACK}
2. Update ROADMAP.md to mark Track ${TRACK} as 'DONE'
3. Prepare a commit message (but DO NOT commit yet):

   'Implement [Feature Name] - Group ${CURRENT_GROUP}, Track ${TRACK}

   - Description of what was added
   - Key files created/modified
   - Test results

   Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>'

4. Announce: '[Track ${TRACK}] Ready for commit'

TEST MODE: Do NOT create git commit or push. Just prepare the commit message.
                " >> "$LOGFILE" 2>&1

                EXIT_CODE=$?
                echo "" >> "$LOGFILE"
                echo "Phase 3 exit code: $EXIT_CODE" >> "$LOGFILE"
                echo "[Track ${TRACK}] Phase 3 completed successfully" >> "$LOGFILE"
            fi
        fi

        # Clean up lock file
        rm -f "$LOCK_FILE"

        echo "" >> "$LOGFILE"
        echo "═══════════════════════════════════════════════════════" >> "$LOGFILE"
        echo "Completed: $(date)" >> "$LOGFILE"
        echo "═══════════════════════════════════════════════════════" >> "$LOGFILE"

        exit $EXIT_CODE
    ) &

    PID=$!
    PIDS+=($PID)
    echo -e "  ${GREEN}Started (PID: ${PID})${NC}\n"

    sleep 3  # Stagger starts
done

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}All ${NUM_PARALLEL} test orchestrators launched!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

echo -e "Monitor progress:"
for i in $(seq 0 $((NUM_PARALLEL - 1))); do
    TRACK=${AVAILABLE_TRACKS[$i]}
    LOGFILE="${LOG_DIR}/test_group_${CURRENT_GROUP}_track_${TRACK}.log"
    echo -e "  Track ${TRACK}: ${YELLOW}tail -f ${LOGFILE}${NC}"
done
echo ""

# Wait for all to complete
echo -e "${CYAN}Waiting for all test orchestrators to complete...${NC}\n"

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
    echo -e "${GREEN}   ✨ TEST COMPLETE - All tracks succeeded! ✨${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}DRY RUN: No changes were committed or pushed${NC}"
        echo -e "${YELLOW}To enable push, run: ./setup_github_auth.sh${NC}\n"
    else
        echo -e "${CYAN}Would push to GitHub now (but this is a test)${NC}\n"
    fi

    echo -e "${GREEN}Check log files to see what each track did:${NC}"
    for i in $(seq 0 $((NUM_PARALLEL - 1))); do
        TRACK=${AVAILABLE_TRACKS[$i]}
        LOGFILE="${LOG_DIR}/test_group_${CURRENT_GROUP}_track_${TRACK}.log"
        echo -e "  ${YELLOW}cat ${LOGFILE}${NC}"
    done
    echo ""
else
    echo -e "${RED}${FAILED} track(s) failed${NC}"
    echo -e "${YELLOW}Check individual log files for details${NC}\n"
    exit 1
fi
