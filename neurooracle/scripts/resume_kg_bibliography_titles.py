"""Resume the completed R35 build as one correctly registered validation writer."""
import ctypes
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
import apply_kg_bibliography_titles as pipeline
from kg_accepted_candidate_lineage import require


def require_process_ended(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint]
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        require(ctypes.get_last_error() == 87, "cannot establish previous writer ended")
        return
    code = ctypes.c_uint()
    try:
        require(kernel.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value != 259, "previous writer still active")
    finally: kernel.CloseHandle(handle)


def main():
    require(not (pipeline.OUTPUT / "CURRENT_ACCEPTANCE.json").exists(), "R35 already adopted")
    state = journal.read_json(pipeline.OUTPUT / "BUILD_STATE.json")
    require(pipeline.bindings() == state["code"], "frozen build code changed")
    c = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(c["active_process"]["kind"] == "bibliography_titles" and c["current_graph"] == state["baseline"]["current_graph"], "unexpected active batch")
    require_process_ended(c["active_process"]["pid"])
    progress = journal.read_json(pipeline.OUTPUT / "RUN_STATE.json")
    require_process_ended(progress["pid"])
    c["active_process"]["pid"] = os.getpid()
    c.update(phase="R35完整候选独立验证；163条标题范围已锁定", updated_at=journal.utc_now())
    journal.atomic_json(journal.OUTPUT / "CAMPAIGN.json", c)
    night = journal.read_json(journal.OUTPUT / "NIGHT_WINDOW_20260909.json")
    night["active_process"] = c["active_process"]
    journal.atomic_json(journal.OUTPUT / "NIGHT_WINDOW_20260909.json", night)
    if not (pipeline.OUTPUT / "VALIDATED.json").exists(): pipeline.validate()
    pipeline.apply()


if __name__ == "__main__": main()
