import socket
import threading
import time
import re
from collections import deque
from typing import List

import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("router_mcp")

# ---------------------------------------------------------------------------
# Shared text-cleaning utilities
# ---------------------------------------------------------------------------

# Single compiled regex that matches ALL ANSI/VT100/xterm escape sequences.
# Handles: CSI (with ; and : params), OSC (title strings), single-char ESC,
# character-set selection, bracketed-paste markers, etc.
_ESCAPE_RE = re.compile(
    r'\x1B'                              # ESC byte
    r'(?:'
    r'\][^\x07\x1B]*(?:\x07|\x1B\\)?'   # OSC: ESC ] <text> BEL|ST  (window title, etc.)
    r'|\[\?[0-9;:]*[a-zA-Z]'            # Private mode: ESC [ ? <n> h/l  (bracketed paste, cursor, etc.)
    r'|\[[0-9;:]*[ -/]*[A-Za-z@]'       # CSI: ESC [ <params> <letter>  (colors, cursor, erase, etc.)
    r'|[()][0-9A-Z]'                    # Charset: ESC ( / ESC )
    r'|[@-Z\\-_]'                       # Single-char: ESC <char>
    r')'
)

# Catch orphaned fragments that lost their ESC byte (common with socket splits)
_ORPHAN_RE = re.compile(
    r'\]0;[^\n\x07]*'               # Window-title body without ESC ]
    r'|\[\?[0-9]+[hl]'              # Private-mode body   without ESC [
    r'|\[[0-9;:]*m'                 # SGR color body       without ESC [
    r'|\[K'                         # Clear-to-EOL body    without ESC [
)

# Prompt pattern:  [user@host:path]$  or  [user@host:path]#
# Matches with or without leading/trailing junk after stripping
_PROMPT_RE = re.compile(r'\w+@\w+[:][^\]]*\]\s*[\$#]\s*$')


def strip_ansi(text: str) -> str:
    """Remove all ANSI/VT100 escape sequences, control chars, and terminal noise."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = _ESCAPE_RE.sub('', text)
    text = _ORPHAN_RE.sub('', text)
    # Control characters
    text = text.replace('\x07', '')   # BEL
    text = text.replace('\x00', '')   # NUL
    text = re.sub(r'[\x01-\x08\x0E-\x1F]', '', text)  # remaining C0 controls (keep \t \n)
    return text


def is_prompt_line(line: str) -> bool:
    """Return True if *line* looks like a shell prompt and nothing else."""
    s = line.strip()
    if not s:
        return False
    return bool(_PROMPT_RE.search(s))


def clean_output(raw: str, *, strip_prompts: bool = True) -> str:
    """Strip escapes, collapse blank runs, optionally remove prompt-only lines."""
    text = strip_ansi(raw)
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if strip_prompts and is_prompt_line(stripped):
            continue
        cleaned.append(line.rstrip())
    # Trim trailing blanks
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return '\n'.join(cleaned)


# ---------------------------------------------------------------------------
# State & socket helpers
# ---------------------------------------------------------------------------

class RouterState:
    def __init__(self):
        self.monitor_sock: socket.socket | None = None
        self.buffer: deque[str] = deque(maxlen=1000)
        self.partial: bytes = b""
        self.lock: threading.Lock = threading.Lock()
        # Persistent command connection for send_serial_command
        self.cmd_reader: asyncio.StreamReader | None = None
        self.cmd_writer: asyncio.StreamWriter | None = None
        self.cmd_lock: asyncio.Lock = asyncio.Lock()

state = RouterState()


def cleanup_monitor():
    if state.monitor_sock is not None:
        try:
            state.monitor_sock.close()
        except Exception:
            pass
        state.monitor_sock = None


def connect_monitor():
    cleanup_monitor()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect("/tmp/tio.sock")
    sock.settimeout(0.5)
    state.monitor_sock = sock


def monitoring_thread():
    """Background thread: keeps a persistent connection and captures ALL console output."""
    while True:
        if state.monitor_sock is None:
            try:
                connect_monitor()
            except Exception:
                time.sleep(1)
                continue

        try:
            data = state.monitor_sock.recv(4096)
            if not data:
                raise OSError("EOF")

            with state.lock:
                state.partial += data
                while True:
                    min_idx = -1
                    for sep in (b"\n", b"\r"):
                        idx = state.partial.find(sep)
                        if idx != -1 and (min_idx == -1 or idx < min_idx):
                            min_idx = idx
                    if min_idx == -1:
                        break

                    line_b = state.partial[:min_idx]
                    state.partial = state.partial[min_idx + 1:]

                    decoded = line_b.decode(errors="ignore").rstrip("\r\n")
                    if decoded.strip():
                        state.buffer.append(decoded)

        except socket.timeout:
            continue
        except Exception:
            cleanup_monitor()
            time.sleep(1)


async def get_cmd_connection():
    """Get or create a persistent connection for sending commands."""
    if state.cmd_reader is None or state.cmd_writer is None or state.cmd_writer.is_closing():
        try:
            state.cmd_reader, state.cmd_writer = await asyncio.open_unix_connection("/tmp/tio.sock")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to /tmp/tio.sock: {e}")
    return state.cmd_reader, state.cmd_writer


# ---------------------------------------------------------------------------
# MCP tool definitions
# ---------------------------------------------------------------------------

@server.list_tools()
async def tools() -> List[Tool]:
    return [
        Tool(
            name="send_serial_command",
            description="Send a command to the serial console and return its clean output. Fast, main command for typical shell usage.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"}
                },
                "required": ["cmd"]
            }
        ),
        Tool(
            name="send_serial_command_long",
            description="Send a command to the serial console with extended timeout for long-running commands.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"}
                },
                "required": ["cmd"]
            }
        ),
        Tool(
            name="get_console_transcript",
            description=(
                "Retrieve NEW console output since the last call (clears buffer afterward). "
                "On first call returns all accumulated output. "
                "Set full_history=true to keep the buffer intact."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to return (default: all)."
                    },
                    "full_history": {
                        "type": "boolean",
                        "description": "If true, return full buffer without clearing."
                    }
                },
                "additionalProperties": False
            }
        )
    ]


# ---------------------------------------------------------------------------
# MCP tool dispatch
# ---------------------------------------------------------------------------

# Prompt regex used during read-loop to detect when a command has finished
# (runs against raw/unstripped text so it must tolerate escape noise).
_RAW_PROMPT_RE = re.compile(r'\w+@\w+.*\$\s*$', re.MULTILINE)


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:

    async def _send_serial_command(cmd: str, max_total: float):
        async with state.cmd_lock:
            try:
                reader, writer = await get_cmd_connection()
            except RuntimeError as e:
                return [TextContent(type="text", text=f"Connection error: {e}")]

            # Drain any stale data sitting in the socket
            try:
                while True:
                    data = await asyncio.wait_for(reader.read(8192), timeout=0.3)
                    if not data:
                        break
            except asyncio.TimeoutError:
                pass

            writer.write(f"{cmd}\n".encode())
            await writer.drain()

            output = bytearray()
            start_time = time.time()
            last_data_time = start_time

            # Read until we see a new shell prompt or timeout
            while time.time() - start_time < max_total:
                try:
                    data = await asyncio.wait_for(reader.read(8192), timeout=0.05)
                    if not data:
                        break
                    output.extend(data)
                    last_data_time = time.time()
                    # Check for prompt in raw text
                    decoded = output.decode(errors="ignore")
                    if _RAW_PROMPT_RE.search(decoded):
                        break
                except asyncio.TimeoutError:
                    if time.time() - last_data_time > 0.3:
                        break
                    continue

            # Drain any remaining bytes
            try:
                while True:
                    more = await asyncio.wait_for(reader.read(8192), timeout=0.05)
                    if not more:
                        break
                    output.extend(more)
            except asyncio.TimeoutError:
                pass

        raw = output.decode(errors="ignore")
        result = clean_output(raw, strip_prompts=True)
        return [TextContent(type="text", text=result or "No output")]

    # ---- Dispatch ----

    if name == "send_serial_command":
        return await _send_serial_command(arguments["cmd"], max_total=2.0)

    elif name == "send_serial_command_long":
        return await _send_serial_command(arguments["cmd"], max_total=15.0)

    elif name == "get_console_transcript":
        max_lines = arguments.get("max_lines")
        full_history = arguments.get("full_history", False)

        with state.lock:
            if max_lines is not None:
                recent_lines = list(state.buffer)[-max_lines:]
            else:
                recent_lines = list(state.buffer)

            transcript_lines = recent_lines[:]

            if not full_history:
                state.buffer.clear()

        if not transcript_lines:
            return [TextContent(type="text", text="No new console output since last check.")]

        # Clean each buffered line through the same pipeline
        raw_text = "\n".join(transcript_lines)
        cleaned = clean_output(raw_text, strip_prompts=True)
        return [TextContent(type="text", text=cleaned or "No new console output since last check.")]

    else:
        raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    thread = threading.Thread(target=monitoring_thread, daemon=True)
    thread.start()

    async with stdio_server() as streams:
        try:
            await server.run(streams[0], streams[1], server.create_initialization_options())
        finally:
            cleanup_monitor()
            if state.cmd_writer is not None and not state.cmd_writer.is_closing():
                state.cmd_writer.close()
                try:
                    await state.cmd_writer.wait_closed()
                except Exception:
                    pass

if __name__ == "__main__":
    asyncio.run(main())
