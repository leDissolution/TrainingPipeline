import os, subprocess, sys
pod_id = os.environ.get("RUNPOD_POD_ID")
if not pod_id:
    print("RUNPOD_POD_ID not set; skipping runpodctl stop.")
    sys.exit(0)
rc = subprocess.call(["runpodctl", "stop", "pod", pod_id])
if rc != 0:
    print(f"runpodctl exited with {rc}")
sys.exit(rc)