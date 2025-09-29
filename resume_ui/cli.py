import os
import sys
from pathlib import Path


def main() -> None:
	"""Launch the Streamlit UI bound to localhost:8501."""
	app_path = Path(__file__).parent / "app.py"
	# Allow override via env vars
	port = os.getenv("RESUME_UI_PORT", "8501")
	address = os.getenv("RESUME_UI_ADDR", "127.0.0.1")
	os.execvp(
		"streamlit",
		[
			"streamlit",
			"run",
			str(app_path),
			"--server.port",
			str(port),
			"--server.address",
			str(address),
		],
	)


if __name__ == "__main__":
	main()


