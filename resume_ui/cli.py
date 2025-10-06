import os
import sys
import socket
import time
import webbrowser
import subprocess
from contextlib import closing
from pathlib import Path


def _is_port_free(host: str, port: int) -> bool:
	with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		try:
			s.bind((host, port))
			return True
		except OSError:
			return False


def _pick_port(host: str, preferred: int | None) -> int:
	if preferred and _is_port_free(host, preferred):
		return preferred
	# Ask OS for an ephemeral port
	with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
		s.bind((host, 0))
		return s.getsockname()[1]


def _wait_for_listen(host: str, port: int, timeout_s: float = 10.0) -> bool:
	deadline = time.time() + timeout_s
	while time.time() < deadline:
		with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
			s.settimeout(0.5)
			try:
				s.connect((host, port))
				return True
			except OSError:
				time.sleep(0.2)
	return False


def main() -> None:
	"""Launch the Streamlit UI on an available localhost port and open the browser."""
	app_path = Path(__file__).parent / "app.py"
	preferred_port_env = os.getenv("RESUME_UI_PORT")
	address = "127.0.0.1"
	preferred_port = int(preferred_port_env) if preferred_port_env and preferred_port_env.isdigit() else 8501
	port = _pick_port(address, preferred_port)

	cmd = [
		sys.executable,
		"-m",
		"streamlit",
		"run",
		str(app_path),
		"--server.port",
		str(port),
		"--server.address",
		str(address),
		"--server.headless=true",
	]

	env = os.environ.copy()
	proc = subprocess.Popen(cmd, env=env)

	url = f"http://{address}:{port}"
	if _wait_for_listen(address, port, timeout_s=12.0):
		try:
			webbrowser.open_new_tab(url)
		except Exception:
			pass
		print(f"Resume UI is running at {url}")
	else:
		print(f"Starting server on {url}...")

	try:
		proc.wait()
	except KeyboardInterrupt:
		proc.terminate()
		proc.wait(timeout=5)


if __name__ == "__main__":
	main()


