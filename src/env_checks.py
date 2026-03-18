import argparse
import importlib.util
import platform
import subprocess
import sys


MIN_GLIBC_FOR_BITSANDBYTES = (2, 24)
MIN_GCC_FOR_BITSANDBYTES_SOURCE = (9, 0)


def _parse_version(text):
    parts = []
    for item in text.split("."):
        digits = []
        for char in item:
            if char.isdigit():
                digits.append(char)
            else:
                break
        if not digits:
            break
        parts.append(int("".join(digits)))
    return tuple(parts)


def _format_version(version):
    if not version:
        return "unknown"
    return ".".join(str(part) for part in version)


def _run_command(command):
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    output = (completed.stdout or completed.stderr or "").strip()
    if completed.returncode != 0:
        return None
    return output or None


def get_glibc_version():
    libc_name, libc_version = platform.libc_ver()
    if libc_name != "glibc" or not libc_version:
        return None
    return _parse_version(libc_version)


def get_gcc_version():
    output = _run_command(["gcc", "--version"])
    if not output:
        return None
    first_line = output.splitlines()[0]
    return _parse_version(first_line)


def has_bitsandbytes():
    return importlib.util.find_spec("bitsandbytes") is not None


def get_torch_cuda_available():
    try:
        import torch
    except Exception:
        return None
    return torch.cuda.is_available()


def get_4bit_blocker():
    if not sys.platform.startswith("linux"):
        return "bitsandbytes 4bit 只适合在 Linux 服务器上使用。"

    glibc_version = get_glibc_version()
    if glibc_version and glibc_version < MIN_GLIBC_FOR_BITSANDBYTES:
        return (
            f"检测到 glibc {_format_version(glibc_version)}，"
            f"但 bitsandbytes 预编译包至少需要 glibc {_format_version(MIN_GLIBC_FOR_BITSANDBYTES)}。"
            "CentOS 7 裸机通常会卡在这里。"
        )

    if not has_bitsandbytes():
        return "当前环境还没有安装 bitsandbytes。"

    return None


def ensure_cuda_available():
    cuda_available = get_torch_cuda_available()
    if cuda_available is False:
        raise RuntimeError("没有检测到可用 CUDA，当前脚本需要在带 NVIDIA CUDA 的服务器上运行。")


def ensure_4bit_ready():
    blocker = get_4bit_blocker()
    if blocker:
        gcc_version = get_gcc_version()
        advice = []
        advice.append(blocker)
        advice.append("如果你要保留 7B + QLoRA，优先改用容器或更高 glibc 的用户态环境。")
        advice.append(
            "如果你要在裸机上自己编 bitsandbytes，通常至少要把 GCC 升到 "
            f"{_format_version(MIN_GCC_FOR_BITSANDBYTES_SOURCE)}+。"
        )
        if gcc_version and gcc_version < MIN_GCC_FOR_BITSANDBYTES_SOURCE:
            advice.append(f"当前检测到 gcc {_format_version(gcc_version)}。")
        raise RuntimeError(" ".join(advice))


def build_summary():
    lines = []
    lines.append(f"Platform: {platform.platform()}")
    lines.append(f"Python: {platform.python_version()}")
    glibc_version = get_glibc_version()
    lines.append(f"glibc: {_format_version(glibc_version)}")
    gcc_version = get_gcc_version()
    lines.append(f"gcc: {_format_version(gcc_version)}")
    cuda_available = get_torch_cuda_available()
    if cuda_available is None:
        lines.append("torch.cuda: unavailable (torch not installed or import failed)")
    else:
        lines.append(f"torch.cuda: {'available' if cuda_available else 'not available'}")
    lines.append(f"bitsandbytes: {'installed' if has_bitsandbytes() else 'not installed'}")
    blocker = get_4bit_blocker()
    if blocker:
        lines.append(f"4bit status: blocked - {blocker}")
    else:
        lines.append("4bit status: ready")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Check whether the server is ready for QLoRA/4bit.")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-4bit", action="store_true")
    args = parser.parse_args()

    print(build_summary())

    if args.require_cuda:
        ensure_cuda_available()
    if args.require_4bit:
        ensure_4bit_ready()


if __name__ == "__main__":
    main()
