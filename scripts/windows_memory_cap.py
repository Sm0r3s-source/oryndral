"""Run one benchmark under a Windows Job Object process-commit limit."""

import argparse
import json
import msvcrt
import os
import shutil
import subprocess
import sys
import uuid

import win32api
import win32con
import win32event
import win32job
import win32process


def run(command, mib, timeout):
    executable = shutil.which(command[0])
    if executable is None:
        raise FileNotFoundError(command[0])

    job = win32job.CreateJobObject(None, "OryndralBench-" + uuid.uuid4().hex)
    kind = win32job.JobObjectExtendedLimitInformation
    limit = mib * 1024 * 1024
    try:
        info = win32job.QueryInformationJobObject(job, kind)
        info["BasicLimitInformation"]["LimitFlags"] = (
            win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY | win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        info["ProcessMemoryLimit"] = limit
        win32job.SetInformationJobObject(job, kind, info)

        startup = win32process.STARTUPINFO()
        startup.dwFlags = win32con.STARTF_USESTDHANDLES
        handles = [msvcrt.get_osfhandle(stream.fileno()) for stream in (sys.stdin, sys.stdout, sys.stderr)]
        startup.hStdInput, startup.hStdOutput, startup.hStdError = handles
        original = {handle: os.get_handle_inheritable(handle) for handle in handles}
        try:
            for handle in original:
                os.set_handle_inheritable(handle, True)
            process, thread, _, _ = win32process.CreateProcess(
                executable, subprocess.list2cmdline(command), None, None, True,
                win32con.CREATE_SUSPENDED | subprocess.CREATE_NO_WINDOW, None, None, startup,
            )
        finally:
            for handle, inheritable in original.items():
                os.set_handle_inheritable(handle, inheritable)

        finished = False
        try:
            win32job.AssignProcessToJobObject(job, process)
            win32process.ResumeThread(thread)
            if win32event.WaitForSingleObject(process, timeout * 1000) != win32event.WAIT_OBJECT_0:
                raise TimeoutError(f"benchmark exceeded {timeout} seconds")
            finished = True
            status = win32process.GetExitCodeProcess(process)
            peak = win32job.QueryInformationJobObject(job, kind)["PeakProcessMemoryUsed"]
            print(json.dumps({"limit_bytes": limit, "peak_commit_bytes": peak, "exit_code": status}),
                  file=sys.stderr)
            return 0 if status == 0 else 1
        finally:
            if not finished:
                win32api.TerminateProcess(process, 1)
            thread.Close()
            process.Close()
    finally:
        job.Close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mib", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.mib < 1 or not 1 <= args.timeout <= 86400 or not command:
        parser.error("provide a positive limit and timeout followed by a command")
    return run(command, args.mib, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
