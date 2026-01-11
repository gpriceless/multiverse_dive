#!/bin/bash
echo "=== Orchestrator Status ==="
echo "Scheduled orchestrator PID: $(pgrep -f scheduled_parallel_orchestrator.sh || echo 'Not running')"
echo ""
echo "=== Active Group F Agents ==="
ps aux | grep "claude --dangerously" | grep "Group F" | grep -v grep | awk '{print "  Track", $0}' | grep -oE "Track [0-9]" | sort | uniq -c
echo ""
echo "=== Lock Files ==="
ls -1 .orchestrator_locks/ 2>/dev/null | sed 's/^/  /'
echo ""
echo "=== Recent Log Activity ==="
for log in orchestrator_logs/group_F_*.log; do
  if [ -f "$log" ]; then
    size=$(stat -f%z "$log" 2>/dev/null || stat -c%s "$log" 2>/dev/null)
    echo "  $(basename $log): ${size} bytes"
  fi
done
