#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Progress tracking
PROGRESS_FILE="/tmp/orchestrator_progress_$$.txt"

announce() {
    local phase=$1
    local step=$2
    echo -e "${CYAN}[${phase}]${NC} ${step}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${phase}] ${step}" >> "$PROGRESS_FILE"
}

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Multiverse Dive Orchestrator - Autonomous Build${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"
echo -e "Progress log: ${YELLOW}${PROGRESS_FILE}${NC}\n"

# Phase 1: Work Agent
echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC} ${GREEN}Phase 1/3: Work Agent${NC}                                 ${BLUE}║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}\n"

announce "Phase 1" "Starting Work Agent..."
announce "Phase 1" "Reading ROADMAP.md to find next unclaimed item..."

claude --dangerously-skip-permissions --print "
🔧 WORK AGENT - Phase 1/3

IMPORTANT: Throughout your work, use echo statements to announce your progress:
- When you start reading files: echo 'Reading ROADMAP.md...'
- When you identify the task: echo 'Found next task: [task name]'
- When checking bugs: echo 'Checking FIXES.md for related bugs...'
- When fixing a bug: echo 'Fixing [FIX-XXX]: [description]...'
- When you start implementation: echo 'Implementing [component]...'
- When you write files: echo 'Writing [filename]...'
- When you finish: echo 'Work Agent complete!'

Your mission:

STEP 0: Check FIXES.md for bugs to fix
   - Read FIXES.md to see documented bugs
   - Identify any P0 (Critical) bugs in files you'll be working on
   - Fix these bugs AS YOU WORK on related code
   - When you fix a bug, update FIXES.md to mark it [FIXED]

STEP 1: Read ROADMAP.md and identify the next unclaimed item
   - Start with earliest group (A, B, C, D, etc.)
   - Look for items not marked as DONE or IN PROGRESS

STEP 2: Update ROADMAP.md to mark this item as 'IN PROGRESS'
   - Announce: 'Marking [item] as IN PROGRESS'

STEP 3: Implement the item completely
   - Follow the architecture in OPENSPEC.md
   - READ EXISTING CODE in related modules before writing new code
   - Understand patterns used elsewhere in the codebase
   - Fix any FIXES.md bugs you encounter in files you touch
   - Write clean, well-structured code
   - Add helpful docstrings
   - Announce each file you create/modify

STEP 4: Update FIXES.md for any bugs you fixed
   - Mark fixed bugs with [FIXED] and brief note
   - Example: '| **FIX-001** | ... | [FIXED] - corrected in commit abc123 |'

Do NOT write tests (that's Phase 2)
Do NOT commit or push (that's Phase 3)
When done: echo 'Work Agent complete - [what you built] + [N bugs fixed]'

Work efficiently. Reference existing code patterns. Fix bugs as you go.
"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Work Agent failed${NC}"
    announce "Phase 1" "FAILED"
    exit 1
fi

announce "Phase 1" "✓ Work Agent completed successfully"
echo -e "\n${GREEN}✓ Phase 1 Complete${NC}\n"
sleep 2

# Phase 2: Review and Test Agent
echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC} ${GREEN}Phase 2/3: Review & Test Agent${NC}                        ${BLUE}║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}\n"

announce "Phase 2" "Starting Review & Test Agent..."
announce "Phase 2" "Analyzing code from Phase 1..."

claude --dangerously-skip-permissions --print "
🔍 REVIEW & TEST AGENT - Phase 2/3

IMPORTANT: Announce your progress throughout:
- When you start: echo 'Review Agent starting...'
- When reviewing code: echo 'Reviewing [filename]...'
- When finding issues: echo 'Found issue in [file]: [brief description]'
- When fixing issues: echo 'Fixing [issue]...'
- When writing tests: echo 'Writing tests for [component]...'
- When running tests: echo 'Running test suite...'
- When completing checklist: echo 'Checklist item [N] verified'
- When done: echo 'Review Agent complete - [X] tests passing, checklist verified'

Your mission:

STEP 1: Read the Code Review Checklist from ROADMAP.md
   - Find the 'Agent Code Review Checklist' section
   - This checklist is MANDATORY - you must verify each item

STEP 2: Review code against the checklist
   For each category, verify and fix issues:

   CORRECTNESS & SAFETY:
   - [ ] Division operations guarded against zero
   - [ ] Array indexing validated for bounds
   - [ ] NaN/Inf handling with np.isnan(), np.isinf()
   - [ ] Edge cases handled (empty arrays, single elements)
   - [ ] Shell variables quoted: \"\\\$VAR\" not \\\$VAR
   - [ ] No hardcoded credentials

   CONSISTENCY:
   - [ ] Names match across files (YAML class names match Python)
   - [ ] Default values in code match YAML/spec defaults
   - [ ] Error handling patterns match codebase conventions

   COMPLETENESS:
   - [ ] All declared features implemented
   - [ ] Every public function has at least one test
   - [ ] Error paths tested, not just happy path
   - [ ] Docstrings present on public classes/functions
   - [ ] Type hints on function signatures

   ROBUSTNESS:
   - [ ] Specific exceptions caught (no bare except:)
   - [ ] Resources cleaned up in finally blocks
   - [ ] Thread safety for global state
   - [ ] Graceful degradation for partial failures

   PERFORMANCE:
   - [ ] No O(n²) loops on large data
   - [ ] Expensive computations cached if reused

   SECURITY:
   - [ ] External input validated
   - [ ] Secrets not logged in debug output

   MAINTAINABILITY:
   - [ ] Magic numbers extracted to named constants
   - [ ] No code duplication
   - [ ] Clear naming conventions

STEP 3: Check FIXES.md for remaining bugs
   - Read FIXES.md and check for bugs in files you're reviewing
   - Fix any P0/P1 bugs in code you're touching
   - Update FIXES.md to mark bugs as [FIXED]
   - Announce: 'Fixed [FIX-XXX]: [description]'

STEP 4: Fix any other issues found
   - Announce what you're fixing
   - Prioritize Critical issues from Tech Debt Backlog in ROADMAP.md

STEP 5: Write comprehensive tests
   - Unit tests for individual functions
   - Integration tests if applicable
   - Edge case coverage
   - Announce each test file you create

STEP 6: Run tests and ensure they pass
   - Run: PYTHONPATH=. .venv/bin/pytest tests/ -v
   - Show test results
   - Fix any failing tests

STEP 7: Summarize completion
   - Echo which checklist items were verified
   - List any bugs fixed from FIXES.md
   - Note any items that couldn't be verified and why

Do NOT commit or push (that's Phase 3)
When done: echo 'Review Agent complete - [X] tests passing, [Y] bugs fixed, checklist verified'
"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Review & Test Agent failed${NC}"
    announce "Phase 2" "FAILED"
    exit 1
fi

announce "Phase 2" "✓ Review & Test Agent completed successfully"
echo -e "\n${GREEN}✓ Phase 2 Complete${NC}\n"
sleep 2

# Phase 3: Documentation and Commit Agent
echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC} ${GREEN}Phase 3/3: Documentation & Commit Agent${NC}               ${BLUE}║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}\n"

announce "Phase 3" "Starting Documentation & Commit Agent..."
announce "Phase 3" "Preparing to finalize and commit changes..."

# Check git authentication before running agent
announce "Phase 3" "Checking GitHub authentication..."
if ! git ls-remote origin &>/dev/null; then
    echo -e "${RED}✗ GitHub authentication not configured${NC}"
    echo -e "${YELLOW}Please run: ./setup_github_auth.sh${NC}\n"
    announce "Phase 3" "FAILED - Authentication required"
    exit 1
fi
announce "Phase 3" "✓ GitHub authentication verified"

claude --dangerously-skip-permissions --print "
📝 DOCUMENTATION & COMMIT AGENT - Phase 3/3

IMPORTANT: Announce every step you take:
- When you start: echo 'Documentation Agent starting...'
- When reviewing: echo 'Reviewing implemented changes...'
- When updating docs: echo 'Updating [filename]...'
- When updating roadmap: echo 'Marking task as DONE in ROADMAP.md...'
- When committing: echo 'Creating git commit...'
- When pushing: echo 'Pushing to GitHub...'
- When done: echo 'Documentation Agent complete - Changes pushed!'

Your mission:
1. Review what was implemented and tested
   - Check git status to see all changes
   - Announce what you find

2. Update ROADMAP.md to mark the item as 'DONE' or 'COMPLETED'
   - Add ✅ or [DONE] marker
   - Announce the update

3. Update relevant documentation if needed
   - README.md if user-facing changes
   - Code docstrings (should already be done)
   - Announce any doc updates

4. Create a descriptive git commit:
   Format: 'Implement [Feature Name] - [Group X, Track Y]

   - Brief description of what was added
   - Key components/files created
   - Any important design decisions
   - Test results ([X] tests passing)

   Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>'

   Announce: 'Creating commit: [commit title]'

5. Push to GitHub
   - Announce: 'Pushing to GitHub...'
   - Handle any errors gracefully

6. When done, announce: 'Documentation Agent complete - Changes pushed to GitHub!'

Make sure to announce each step so progress is visible.
"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Documentation & Commit Agent failed${NC}"
    echo -e "${YELLOW}Check if GitHub authentication is working${NC}"
    announce "Phase 3" "FAILED"
    exit 1
fi

announce "Phase 3" "✓ Documentation & Commit Agent completed successfully"
echo -e "\n${GREEN}✓ Phase 3 Complete${NC}\n"

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   ✨ Orchestrator Complete! ✨${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"
echo -e "✅ All phases completed successfully"
echo -e "✅ Changes committed and pushed to GitHub"
echo -e "\nProgress log: ${YELLOW}${PROGRESS_FILE}${NC}\n"
