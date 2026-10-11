import pathlib
from pathlib import Path
import subprocess, time, sys, socket

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "streamlit_app.py"

used_ports = [8501]
for port in range(8501, 8520):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", port)) != 0:
            used_ports.append(port)
            break

port = used_ports[-1]
url = f"http://localhost:{port}"
print(f"using_port={port}")
print(f"url={url}")

cmd = [
    sys.executable, "-m", "streamlit", "run", str(APP),
    "--server.port", str(port),
    "--server.headless", "true",
    "--server.address", "127.0.0.1",
    "--browser.serverAddress", "localhost",
    "--browser.gatherUsageStats", "false",
]

p = subprocess.Popen(
    cmd,
    cwd=str(ROOT),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

try:
    for _ in range(30):
        time.sleep(2)
        line = p.stdout.readline()
        if not line:
            break
        print(line, end="")
        if "You can now view your Streamlit app" in line:
            break
finally:
    print("\nstreamlit_url=", url)
    print("streamlit_exit=", p.poll())
