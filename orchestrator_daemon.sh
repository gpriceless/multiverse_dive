#!/bin/bash
# Orchestrator Daemon - Runs completely detached from terminal
# Survives terminal disconnection, session logout, etc.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Use the scheduled parallel orchestrator for multi-track, long-running execution
ORCHESTRATOR="$SCRIPT_DIR/scheduled_parallel_orchestrator.sh"
LOGFILE="$SCRIPT_DIR/orchestrator_daemon.log"
PIDFILE="$SCRIPT_DIR/orchestrator_daemon.pid"

start_daemon() {
    # Check if already running
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Orchestrator daemon already running (PID: $PID)"
            echo "Log: $LOGFILE"
            echo ""
            echo "To stop: $0 stop"
            echo "To view log: tail -f $LOGFILE"
            exit 1
        else
            rm -f "$PIDFILE"
        fi
    fi

    echo "═══════════════════════════════════════════════════════"
    echo "   Starting Orchestrator Daemon (Fully Detached)"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    echo "The orchestrator will continue running even if you:"
    echo "  • Close this terminal"
    echo "  • Disconnect SSH"
    echo "  • Log out"
    echo ""

    # Initialize log
    echo "═══════════════════════════════════════════════════════" > "$LOGFILE"
    echo "Orchestrator Daemon Started: $(date)" >> "$LOGFILE"
    echo "═══════════════════════════════════════════════════════" >> "$LOGFILE"
    echo "" >> "$LOGFILE"

    # Start the orchestrator fully detached using nohup + setsid
    # setsid creates a new session, making it immune to terminal hangup
    nohup setsid bash -c "
        # Run the orchestrator
        '$ORCHESTRATOR' >> '$LOGFILE' 2>&1
        EXIT_CODE=\$?

        # Log completion
        echo '' >> '$LOGFILE'
        echo '═══════════════════════════════════════════════════════' >> '$LOGFILE'
        echo \"Orchestrator Completed: \$(date)\" >> '$LOGFILE'
        echo \"Exit Code: \$EXIT_CODE\" >> '$LOGFILE'
        echo '═══════════════════════════════════════════════════════' >> '$LOGFILE'

        # Clean up PID file
        rm -f '$PIDFILE'
    " > /dev/null 2>&1 &

    # Get the PID of the setsid process
    DAEMON_PID=$!
    echo "$DAEMON_PID" > "$PIDFILE"

    # Wait a moment to ensure it started
    sleep 1

    if ps -p "$DAEMON_PID" > /dev/null 2>&1; then
        echo "✓ Daemon started successfully!"
        echo ""
        echo "  PID: $DAEMON_PID"
        echo "  Log: $LOGFILE"
        echo "  PID file: $PIDFILE"
        echo ""
        echo "Commands:"
        echo "  View log:    tail -f $LOGFILE"
        echo "  Check status: $0 status"
        echo "  Stop daemon:  $0 stop"
        echo ""
        echo "You can now safely close this terminal."
    else
        echo "✗ Failed to start daemon"
        rm -f "$PIDFILE"
        exit 1
    fi
}

stop_daemon() {
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Stopping orchestrator daemon (PID: $PID)..."

            # Kill the process group to stop all child processes
            pkill -P "$PID" 2>/dev/null
            kill "$PID" 2>/dev/null

            sleep 2

            if ps -p "$PID" > /dev/null 2>&1; then
                echo "Force killing..."
                kill -9 "$PID" 2>/dev/null
            fi

            rm -f "$PIDFILE"
            echo "✓ Daemon stopped"
        else
            echo "Daemon not running (stale PID file)"
            rm -f "$PIDFILE"
        fi
    else
        echo "No daemon running (no PID file)"
    fi
}

status_daemon() {
    echo "═══════════════════════════════════════════════════════"
    echo "   Orchestrator Daemon Status"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Status: RUNNING"
            echo "PID: $PID"
            echo "Log: $LOGFILE"
            echo ""
            echo "Recent log output:"
            echo "───────────────────────────────────────────────────────"
            tail -20 "$LOGFILE" 2>/dev/null || echo "(no log output yet)"
            echo "───────────────────────────────────────────────────────"
        else
            echo "Status: STOPPED (stale PID file)"
            rm -f "$PIDFILE"
        fi
    else
        echo "Status: NOT RUNNING"

        if [ -f "$LOGFILE" ]; then
            echo ""
            echo "Last run log exists. Recent output:"
            echo "───────────────────────────────────────────────────────"
            tail -20 "$LOGFILE"
            echo "───────────────────────────────────────────────────────"
        fi
    fi
}

logs_daemon() {
    if [ -f "$LOGFILE" ]; then
        tail -f "$LOGFILE"
    else
        echo "No log file found at: $LOGFILE"
    fi
}

case "${1:-start}" in
    start)
        start_daemon
        ;;
    stop)
        stop_daemon
        ;;
    status)
        status_daemon
        ;;
    logs)
        logs_daemon
        ;;
    restart)
        stop_daemon
        sleep 2
        start_daemon
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs|restart}"
        exit 1
        ;;
esac
