"""
Запуск: python main.py
Или напрямую: mitmproxy -s addon.py
"""
import subprocess
import sys


def main():
    subprocess.run(
        [sys.executable, "-m", "mitmproxy", "-s", "addon.py"],
        check=True,
    )


if __name__ == "__main__":
    main()
