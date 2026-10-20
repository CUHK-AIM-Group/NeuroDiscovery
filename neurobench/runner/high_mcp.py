"""Existing MCP seam, with model-visible calls recorded for this revision."""
import json
import os
import sys
from pathlib import Path

from neurobench.runner.artifacts import EventLog
from core import mcp_stdio


def main():
    original = mcp_stdio.dispatch
    path = os.environ.get('NEURORUNTIME_MCP_EVENT_LOG')
    log = EventLog(Path(path)) if path else None
    def recorded(request):
        if log and request.get('method') == 'tools/call':
            log.write('mcp_tool_request', params=request.get('params'))
        result = original(request)
        if log and request.get('method') == 'tools/call':
            log.write('mcp_tool_result', params=request.get('params'), result=result)
        return result
    mcp_stdio.dispatch = recorded
    try:
        return mcp_stdio.main()
    finally:
        if log:
            log.close()


if __name__ == '__main__':
    raise SystemExit(main())
