"""Thin wrapper around the `adb` binary: device discovery, screenshots, input."""

from __future__ import annotations

import io
import shutil
import subprocess
from dataclasses import dataclass, field

from PIL import Image

from linpuzs.actions import Action, Drag, Swipe, Tap, Wait


class AdbError(RuntimeError):
    pass


@dataclass(frozen=True)
class Device:
    serial: str
    state: str
    props: dict[str, str] = field(default_factory=dict)

    @property
    def model(self) -> str:
        return self.props.get("model", "unknown").replace("_", " ")

    @property
    def is_ready(self) -> bool:
        return self.state == "device"

    def label(self) -> str:
        return f"{self.model}  ({self.serial})"


def adb_path() -> str | None:
    return shutil.which("adb")


def _run(args: list[str], *, serial: str | None = None, timeout: float = 15) -> bytes:
    exe = adb_path()
    if exe is None:
        raise AdbError("`adb` not found on PATH. Install Android platform-tools first.")
    cmd = [exe] + (["-s", serial] if serial else []) + args
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise AdbError(f"adb timed out: {' '.join(args)}") from e
    if proc.returncode != 0:
        raise AdbError(proc.stderr.decode(errors="replace").strip() or f"adb failed: {' '.join(args)}")
    return proc.stdout


def list_devices() -> list[Device]:
    """Parse `adb devices -l` into Device records (includes unauthorized/offline ones)."""
    out = _run(["devices", "-l"]).decode()
    devices = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        serial, state, *rest = parts
        props = dict(p.split(":", 1) for p in rest if ":" in p)
        devices.append(Device(serial, state, props))
    return devices


class AdbDevice:
    """A connected device we can screenshot and send input to."""

    def __init__(self, device: Device):
        self.device = device
        self.serial = device.serial
        self._size: tuple[int, int] | None = None

    def shell(self, command: str, timeout: float = 30) -> str:
        return _run(["shell", command], serial=self.serial, timeout=timeout).decode(errors="replace")

    def screen_size(self) -> tuple[int, int]:
        if self._size is None:
            img = self.screenshot()
            self._size = img.size
        return self._size

    def screenshot(self) -> Image.Image:
        png = _run(["exec-out", "screencap", "-p"], serial=self.serial)
        img = Image.open(io.BytesIO(png))
        img.load()
        self._size = img.size
        return img.convert("RGB")

    def perform(self, actions: list[Action], chunk: int = 50) -> None:
        """Send actions to the device, batching many commands into one `adb shell` call."""
        cmds = [cmd for a in actions for cmd in _to_shell(a)]
        for i in range(0, len(cmds), chunk):
            self.shell("; ".join(cmds[i : i + chunk]), timeout=120)


def _to_shell(action: Action) -> list[str]:
    match action:
        case Tap(x, y):
            return [f"input tap {x} {y}"]
        case Swipe(x1, y1, x2, y2, duration_ms):
            return [f"input swipe {x1} {y1} {x2} {y2} {duration_ms}"]
        case Drag(points):
            # One continuous finger stroke through every point (Android 11+).
            (x0, y0), *mid = points
            *_, (xn, yn) = points
            return (
                [f"input motionevent DOWN {x0} {y0}"]
                + [f"input motionevent MOVE {x} {y}" for x, y in mid]
                + [f"input motionevent UP {xn} {yn}"]
            )
        case Wait(ms):
            return [f"sleep {ms / 1000:.3f}"]
    raise TypeError(f"Unsupported action: {action!r}")
