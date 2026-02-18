#!/usr/bin/env python3
"""
dmesg_checker.py — Kernel log checker for NV/Tegra GPU testing on Jetson Orin AGX.

Usage:
  As a context manager around tests:
    with DmesgChecker() as dmesg:
        run_my_test()
    # Automatically checks for GPU errors on exit

  As a standalone check:
    checker = DmesgChecker()
    checker.clear()
    run_my_test()
    errors = checker.check()
    if errors:
        print("GPU errors found:", errors)

  As a decorator:
    @check_dmesg
    def test_something():
        ...

  From the command line:
    python3 tests/dmesg_checker.py                     # Show recent GPU messages
    python3 tests/dmesg_checker.py --watch              # Continuous monitoring
    python3 tests/dmesg_checker.py --clear              # Clear dmesg ring buffer
"""

import subprocess
import re
import time
import sys
import functools
from dataclasses import dataclass, field
from typing import Optional


# Patterns that indicate real GPU errors (not just informational)
ERROR_PATTERNS = [
    (re.compile(r'sked exception.*esr\s+(0x[0-9a-f]+)', re.IGNORECASE), 'SKED_EXCEPTION',
     'GPU scheduler exception — likely bad pushbuffer or QMD data'),
    (re.compile(r'CE engine \d+ is not idle', re.IGNORECASE), 'CE_NOT_IDLE',
     'Copy Engine not idle during reset — consequence of prior GPU error'),
    (re.compile(r'set gr exception notifier', re.IGNORECASE), 'GR_EXCEPTION',
     'Graphics engine exception notifier — accompanies sked/other GR errors'),
    (re.compile(r'mmu fault', re.IGNORECASE), 'MMU_FAULT',
     'GPU MMU fault — bad VA or unmapped memory access'),
    (re.compile(r'fifo_pbdma.*intr', re.IGNORECASE), 'PBDMA_INTR',
     'PBDMA interrupt error — pushbuffer DMA engine error'),
    (re.compile(r'ctxsw_timeout', re.IGNORECASE), 'CTXSW_TIMEOUT',
     'Context switch timeout — GPU hang'),
]

# Patterns that are warnings but not critical errors
WARNING_PATTERNS = [
    (re.compile(r'nvmap.*tag.*0\b|allocation tag.*warning', re.IGNORECASE), 'NVMAP_TAG',
     'nvmap allocation missing tag — cosmetic but indicates driver integration issue'),
    (re.compile(r'Error reporting is not supported', re.IGNORECASE), 'ERR_REPORT_UNSUPPORTED',
     'CIC error reporting not supported — informational, harmless'),
]

# Known harmless patterns to suppress
KNOWN_HARMLESS = [
    re.compile(r'tu104_gr_init_commit_rtv_cb', re.IGNORECASE),  # RTV circular buffer not available on ga10b
]

# Filter patterns for relevant kernel messages
GPU_FILTER = re.compile(r'nvgpu|ga10b|nvmap|__ga10b__', re.IGNORECASE)


@dataclass
class DmesgEvent:
    """A single kernel log event related to the GPU."""
    timestamp: float  # kernel timestamp in seconds
    raw_line: str
    category: str  # ERROR, WARNING, INFO, HARMLESS
    error_type: str  # e.g., SKED_EXCEPTION, NVMAP_TAG
    description: str
    esr_value: Optional[str] = None  # For sked exceptions


@dataclass
class DmesgReport:
    """Results from a dmesg check."""
    errors: list[DmesgEvent] = field(default_factory=list)
    warnings: list[DmesgEvent] = field(default_factory=list)
    info: list[DmesgEvent] = field(default_factory=list)
    harmless: list[DmesgEvent] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @property
    def is_clean(self) -> bool:
        return not self.has_errors and not self.has_warnings

    def summary(self) -> str:
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} ERROR(s)")
            for e in self.errors:
                parts.append(f"  - [{e.error_type}] {e.description}")
                if e.esr_value:
                    parts.append(f"    ESR: {e.esr_value}")
        if self.warnings:
            parts.append(f"{len(self.warnings)} WARNING(s)")
            for w in self.warnings:
                parts.append(f"  - [{w.error_type}] {w.description}")
        if self.harmless:
            parts.append(f"{len(self.harmless)} known harmless message(s) (suppressed)")
        if not parts:
            parts.append("Clean — no GPU errors or warnings")
        return "\n".join(parts)


def _get_dmesg() -> list[str]:
    """Read dmesg, trying sudo first, then without."""
    try:
        result = subprocess.run(['sudo', 'dmesg'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.splitlines()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    try:
        result = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.splitlines()
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        pass

    return []


def _parse_timestamp(line: str) -> float:
    """Extract kernel timestamp from dmesg line like '[ 1234.567890] ...'"""
    m = re.match(r'\[\s*(\d+\.\d+)\]', line)
    return float(m.group(1)) if m else 0.0


def _classify_line(line: str) -> Optional[DmesgEvent]:
    """Classify a single dmesg line as error, warning, harmless, or info."""
    if not GPU_FILTER.search(line):
        return None

    ts = _parse_timestamp(line)

    # Check known harmless first
    for pattern in KNOWN_HARMLESS:
        if pattern.search(line):
            return DmesgEvent(timestamp=ts, raw_line=line, category='HARMLESS',
                              error_type='KNOWN_HARMLESS', description='Known harmless message')

    # Check errors
    for pattern, error_type, description in ERROR_PATTERNS:
        m = pattern.search(line)
        if m:
            esr = m.group(1) if m.lastindex and m.lastindex >= 1 else None
            return DmesgEvent(timestamp=ts, raw_line=line, category='ERROR',
                              error_type=error_type, description=description, esr_value=esr)

    # Check warnings
    for pattern, warn_type, description in WARNING_PATTERNS:
        if pattern.search(line):
            return DmesgEvent(timestamp=ts, raw_line=line, category='WARNING',
                              error_type=warn_type, description=description)

    # GPU-related but not matched — informational
    return DmesgEvent(timestamp=ts, raw_line=line, category='INFO',
                      error_type='INFO', description='GPU-related kernel message')


class DmesgChecker:
    """
    Check kernel logs (dmesg) for GPU errors before/after test runs.

    Usage:
        with DmesgChecker() as dc:
            run_test()
        assert dc.report.is_clean, dc.report.summary()
    """

    def __init__(self, fail_on_error: bool = True, fail_on_warning: bool = False):
        self.fail_on_error = fail_on_error
        self.fail_on_warning = fail_on_warning
        self._start_timestamp: float = 0.0
        self.report: DmesgReport = DmesgReport()

    def clear(self):
        """Record the current timestamp so we only check new messages."""
        lines = _get_dmesg()
        if lines:
            self._start_timestamp = _parse_timestamp(lines[-1])
        else:
            self._start_timestamp = 0.0

    def check(self, since: Optional[float] = None) -> DmesgReport:
        """Check dmesg for GPU errors since the last clear() or given timestamp."""
        cutoff = since if since is not None else self._start_timestamp
        lines = _get_dmesg()

        report = DmesgReport()
        for line in lines:
            ts = _parse_timestamp(line)
            if ts <= cutoff:
                continue

            event = _classify_line(line)
            if event is None:
                continue

            report.raw_lines.append(line)
            if event.category == 'ERROR':
                report.errors.append(event)
            elif event.category == 'WARNING':
                report.warnings.append(event)
            elif event.category == 'HARMLESS':
                report.harmless.append(event)
            else:
                report.info.append(event)

        self.report = report
        return report

    def __enter__(self):
        self.clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Don't check if the test already raised an exception
        if exc_type is not None:
            return False

        report = self.check()
        if self.fail_on_error and report.has_errors:
            raise AssertionError(f"GPU kernel errors detected!\n{report.summary()}")
        if self.fail_on_warning and report.has_warnings:
            raise AssertionError(f"GPU kernel warnings detected!\n{report.summary()}")
        return False


def check_dmesg(fn=None, *, fail_on_warning=False):
    """
    Decorator to wrap a test function with dmesg checking.

    Usage:
        @check_dmesg
        def test_something():
            ...

        @check_dmesg(fail_on_warning=True)
        def test_strict():
            ...
    """
    if fn is None:
        return functools.partial(check_dmesg, fail_on_warning=fail_on_warning)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with DmesgChecker(fail_on_warning=fail_on_warning):
            return fn(*args, **kwargs)
    return wrapper


def show_recent_gpu_messages(count: int = 50):
    """Show recent GPU-related kernel messages with classification."""
    lines = _get_dmesg()
    gpu_lines = []
    for line in lines:
        event = _classify_line(line)
        if event is not None:
            gpu_lines.append(event)

    if not gpu_lines:
        print("No GPU-related kernel messages found.")
        return

    for event in gpu_lines[-count:]:
        prefix = {'ERROR': '❌', 'WARNING': '⚠️ ', 'HARMLESS': '✓ ', 'INFO': 'ℹ️ '}
        marker = prefix.get(event.category, '  ')
        print(f"{marker} [{event.error_type:20s}] {event.raw_line.strip()}")


def watch_gpu_messages(interval: float = 1.0):
    """Continuously monitor dmesg for new GPU messages."""
    print(f"Watching kernel logs for GPU messages (Ctrl+C to stop)...")
    checker = DmesgChecker()
    checker.clear()

    try:
        while True:
            time.sleep(interval)
            report = checker.check()
            if report.raw_lines:
                for event in report.errors + report.warnings + report.info:
                    prefix = {'ERROR': '❌', 'WARNING': '⚠️ ', 'INFO': 'ℹ️ '}
                    marker = prefix.get(event.category, '  ')
                    print(f"{marker} [{event.error_type}] {event.raw_line.strip()}")
                checker.clear()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GPU kernel log checker for Jetson Orin')
    parser.add_argument('--watch', action='store_true', help='Continuous monitoring mode')
    parser.add_argument('--clear', action='store_true', help='Clear dmesg ring buffer')
    parser.add_argument('--count', type=int, default=50, help='Number of recent messages to show')
    args = parser.parse_args()

    if args.clear:
        subprocess.run(['sudo', 'dmesg', '-C'], check=True)
        print("dmesg cleared.")
    elif args.watch:
        watch_gpu_messages()
    else:
        show_recent_gpu_messages(args.count)
