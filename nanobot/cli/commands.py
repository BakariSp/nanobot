"""CLI commands for nanobot."""

import asyncio
import os
import signal
from pathlib import Path
import select
import sys

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from nanobot import __version__, __logo__

app = typer.Typer(
    name="nanobot",
    help=f"{__logo__} nanobot - Personal AI Assistant",
    no_args_is_help=True,
)

console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}

# ---------------------------------------------------------------------------
# CLI input: prompt_toolkit for editing, paste, history, and display
# ---------------------------------------------------------------------------

_PROMPT_SESSION: PromptSession | None = None
_SAVED_TERM_ATTRS = None  # original termios settings, restored on exit


def _flush_pending_tty_input() -> None:
    """Drop unread keypresses typed while the model was generating output."""
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
    except Exception:
        return

    try:
        import termios
        termios.tcflush(fd, termios.TCIFLUSH)
        return
    except Exception:
        pass

    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            if not os.read(fd, 4096):
                break
    except Exception:
        return


def _restore_terminal() -> None:
    """Restore terminal to its original state (echo, line buffering, etc.)."""
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


def _init_prompt_session() -> None:
    """Create the prompt_toolkit session with persistent file history."""
    global _PROMPT_SESSION, _SAVED_TERM_ATTRS

    # Save terminal state so we can restore it on exit
    try:
        import termios
        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass

    history_file = Path.home() / ".nanobot" / "history" / "cli_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    _PROMPT_SESSION = PromptSession(
        history=FileHistory(str(history_file)),
        enable_open_in_editor=False,
        multiline=False,   # Enter submits (single line mode)
    )


def _print_agent_response(response: str, render_markdown: bool) -> None:
    """Render assistant response with consistent terminal styling."""
    content = response or ""
    body = Markdown(content) if render_markdown else Text(content)
    console.print()
    console.print(f"[cyan]{__logo__} nanobot[/cyan]")
    console.print(body)
    console.print()


def _is_exit_command(command: str) -> bool:
    """Return True when input should end interactive chat."""
    return command.lower() in EXIT_COMMANDS


async def _read_interactive_input_async() -> str:
    """Read user input using prompt_toolkit (handles paste, history, display).

    prompt_toolkit natively handles:
    - Multiline paste (bracketed paste mode)
    - History navigation (up/down arrows)
    - Clean display (no ghost characters or artifacts)
    """
    if _PROMPT_SESSION is None:
        raise RuntimeError("Call _init_prompt_session() first")
    try:
        with patch_stdout():
            return await _PROMPT_SESSION.prompt_async(
                HTML("<b fg='ansiblue'>You:</b> "),
            )
    except EOFError as exc:
        raise KeyboardInterrupt from exc



def version_callback(value: bool):
    if value:
        console.print(f"{__logo__} nanobot v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    """nanobot - Personal AI Assistant."""
    pass


# ============================================================================
# Onboard / Setup
# ============================================================================


@app.command()
def onboard():
    """Initialize nanobot configuration and workspace."""
    from nanobot.config.loader import get_config_path, load_config, save_config
    from nanobot.config.schema import Config
    from nanobot.utils.helpers import get_workspace_path
    
    config_path = get_config_path()
    
    if config_path.exists():
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        console.print("  [bold]y[/bold] = overwrite with defaults (existing values will be lost)")
        console.print("  [bold]N[/bold] = refresh config, keeping existing values and adding new fields")
        if typer.confirm("Overwrite?"):
            config = Config()
            save_config(config)
            console.print(f"[green]✓[/green] Config reset to defaults at {config_path}")
        else:
            config = load_config()
            save_config(config)
            console.print(f"[green]✓[/green] Config refreshed at {config_path} (existing values preserved)")
    else:
        save_config(Config())
        console.print(f"[green]✓[/green] Created config at {config_path}")
    
    # Create workspace
    workspace = get_workspace_path()
    
    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green] Created workspace at {workspace}")
    
    # Create default bootstrap files
    _create_workspace_templates(workspace)
    
    console.print(f"\n{__logo__} nanobot is ready!")
    console.print("\nNext steps:")
    console.print("  1. Add your API key to [cyan]~/.nanobot/config.json[/cyan]")
    console.print("     Get one at: https://openrouter.ai/keys")
    console.print("  2. Chat: [cyan]nanobot agent -m \"Hello!\"[/cyan]")
    console.print("\n[dim]Want Telegram/WhatsApp? See: https://github.com/HKUDS/nanobot#-chat-apps[/dim]")




def _create_workspace_templates(workspace: Path):
    """Create default workspace template files."""
    templates = {
        "AGENTS.md": """# Agent Instructions

You are a helpful AI assistant. Be concise, accurate, and friendly.

## Guidelines

- Always explain what you're doing before taking actions
- Ask for clarification when the request is ambiguous
- Use tools to help accomplish tasks
- Remember important information in memory/MEMORY.md; past events are logged in memory/HISTORY.md
""",
        "SOUL.md": """# Soul

I am nanobot, a lightweight AI assistant.

## Personality

- Helpful and friendly
- Concise and to the point
- Curious and eager to learn

## Values

- Accuracy over speed
- User privacy and safety
- Transparency in actions
""",
        "USER.md": """# User

Information about the user goes here.

## Preferences

- Communication style: (casual/formal)
- Timezone: (your timezone)
- Language: (your preferred language)
""",
    }
    
    for filename, content in templates.items():
        file_path = workspace / filename
        if not file_path.exists():
            file_path.write_text(content)
            console.print(f"  [dim]Created {filename}[/dim]")
    
    # Create memory directory and MEMORY.md
    memory_dir = workspace / "memory"
    memory_dir.mkdir(exist_ok=True)
    memory_file = memory_dir / "MEMORY.md"
    if not memory_file.exists():
        memory_file.write_text("""# Long-term Memory

This file stores important information that should persist across sessions.

## User Information

(Important facts about the user)

## Preferences

(User preferences learned over time)

## Important Notes

(Things to remember)
""")
        console.print("  [dim]Created memory/MEMORY.md[/dim]")
    
    history_file = memory_dir / "HISTORY.md"
    if not history_file.exists():
        history_file.write_text("")
        console.print("  [dim]Created memory/HISTORY.md[/dim]")

    # Create skills directory for custom user skills
    skills_dir = workspace / "skills"
    skills_dir.mkdir(exist_ok=True)


def _make_provider(config):
    """Create LiteLLMProvider from config. Exits if no API key found."""
    from nanobot.providers.litellm_provider import LiteLLMProvider
    p = config.get_provider()
    model = config.agents.defaults.model
    if not (p and p.api_key) and not model.startswith("bedrock/"):
        console.print("[red]Error: No API key configured.[/red]")
        console.print("Set one in ~/.nanobot/config.json under providers section")
        raise typer.Exit(1)
    return LiteLLMProvider(
        api_key=p.api_key if p else None,
        api_base=config.get_api_base(),
        default_model=model,
        extra_headers=p.extra_headers if p else None,
        provider_name=config.get_provider_name(),
    )


# ============================================================================
# Gateway / Server
# ============================================================================


@app.command()
def gateway(
    port: int = typer.Option(18790, "--port", "-p", help="Gateway port"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Start the nanobot gateway."""
    from nanobot.config.loader import load_config, get_data_dir
    from nanobot.bus.queue import MessageBus
    from nanobot.agent.loop import AgentLoop
    from nanobot.channels.manager import ChannelManager
    from nanobot.session.manager import SessionManager
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronJob
    from nanobot.heartbeat.service import HeartbeatService
    
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    console.print(f"{__logo__} Starting nanobot gateway on port {port}...")
    
    config = load_config()
    bus = MessageBus()
    provider = _make_provider(config)
    session_manager = SessionManager(config.workspace_path)
    
    # Create cron service first (callback set after agent creation)
    cron_store_path = get_data_dir() / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    # Create task ledgers (secretary architecture)
    from nanobot.agent.tools.task_ledger import TaskLedger, DualLedger
    task_ledger = TaskLedger()
    dual_ledger = DualLedger()

    # Create agent with cron service
    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        memory_window=config.agents.defaults.memory_window,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=session_manager,
        task_ledger=task_ledger,
        dual_ledger=dual_ledger,
    )
    
    # Set cron callback (needs agent)
    async def on_cron_job(job: CronJob) -> str | None:
        """Execute a cron job through the agent."""
        response = await agent.process_direct(
            job.payload.message,
            session_key=f"cron:{job.id}",
            channel=job.payload.channel or "cli",
            chat_id=job.payload.to or "direct",
        )
        if job.payload.deliver and job.payload.to:
            from nanobot.bus.events import OutboundMessage
            await bus.publish_outbound(OutboundMessage(
                channel=job.payload.channel or "cli",
                chat_id=job.payload.to,
                content=response or ""
            ))
        return response
    cron.on_job = on_cron_job
    
    # Create heartbeat service
    async def on_heartbeat(prompt: str) -> str:
        """Execute heartbeat through the agent."""
        return await agent.process_direct(prompt, session_key="heartbeat")
    
    heartbeat = HeartbeatService(
        workspace=config.workspace_path,
        on_heartbeat=on_heartbeat,
        interval_s=30 * 60,  # 30 minutes
        enabled=True
    )
    
    # Create channel manager
    channels = ChannelManager(config, bus)
    
    if channels.enabled_channels:
        console.print(f"[green]✓[/green] Channels enabled: {', '.join(channels.enabled_channels)}")
    else:
        console.print("[yellow]Warning: No channels enabled[/yellow]")
    
    cron_status = cron.status()
    if cron_status["jobs"] > 0:
        console.print(f"[green]✓[/green] Cron: {cron_status['jobs']} scheduled jobs")
    
    console.print(f"[green]✓[/green] Heartbeat: every 30m")
    
    from nanobot.ralph_loop import ralph_loop as _ralph_loop

    async def run():
        try:
            await cron.start()
            await heartbeat.start()
            await asyncio.gather(
                agent.run(),
                channels.start_all(),
                _ralph_loop(),
            )
        except KeyboardInterrupt:
            console.print("\nShutting down...")
            heartbeat.stop()
            cron.stop()
            agent.stop()
            await channels.stop_all()
    
    asyncio.run(run())




# ============================================================================
# Agent Commands
# ============================================================================


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option("cli:direct", "--session", "-s", help="Session ID"),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="Render assistant output as Markdown"),
    logs: bool = typer.Option(False, "--logs/--no-logs", help="Show nanobot runtime logs during chat"),
):
    """Interact with the agent directly."""
    from nanobot.config.loader import load_config
    from nanobot.bus.queue import MessageBus
    from nanobot.agent.loop import AgentLoop
    from loguru import logger
    
    config = load_config()
    
    bus = MessageBus()
    provider = _make_provider(config)

    if logs:
        logger.enable("nanobot")
    else:
        logger.disable("nanobot")

    # Create task ledgers (secretary architecture)
    from nanobot.agent.tools.task_ledger import TaskLedger, DualLedger
    task_ledger = TaskLedger()
    dual_ledger = DualLedger()

    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        memory_window=config.agents.defaults.memory_window,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        task_ledger=task_ledger,
        dual_ledger=dual_ledger,
    )
    
    # Show spinner when logs are off (no output to miss); skip when logs are on
    def _thinking_ctx():
        if logs:
            from contextlib import nullcontext
            return nullcontext()
        # Animated spinner is safe to use with prompt_toolkit input handling
        return console.status("[dim]nanobot is thinking...[/dim]", spinner="dots")

    if message:
        # Single message mode
        async def run_once():
            with _thinking_ctx():
                response = await agent_loop.process_direct(message, session_id)
            _print_agent_response(response, render_markdown=markdown)
        
        asyncio.run(run_once())
    else:
        # Interactive mode
        _init_prompt_session()
        console.print(f"{__logo__} Interactive mode (type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit)\n")

        def _exit_on_sigint(signum, frame):
            _restore_terminal()
            console.print("\nGoodbye!")
            os._exit(0)

        signal.signal(signal.SIGINT, _exit_on_sigint)
        
        async def run_interactive():
            while True:
                try:
                    _flush_pending_tty_input()
                    user_input = await _read_interactive_input_async()
                    command = user_input.strip()
                    if not command:
                        continue

                    if _is_exit_command(command):
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
                    
                    with _thinking_ctx():
                        response = await agent_loop.process_direct(user_input, session_id)
                    _print_agent_response(response, render_markdown=markdown)
                except KeyboardInterrupt:
                    _restore_terminal()
                    console.print("\nGoodbye!")
                    break
                except EOFError:
                    _restore_terminal()
                    console.print("\nGoodbye!")
                    break
        
        asyncio.run(run_interactive())


# ============================================================================
# Channel Commands
# ============================================================================


channels_app = typer.Typer(help="Manage channels")
app.add_typer(channels_app, name="channels")


@channels_app.command("status")
def channels_status():
    """Show channel status."""
    from nanobot.config.loader import load_config

    config = load_config()

    table = Table(title="Channel Status")
    table.add_column("Channel", style="cyan")
    table.add_column("Enabled", style="green")
    table.add_column("Configuration", style="yellow")

    # WhatsApp
    wa = config.channels.whatsapp
    table.add_row(
        "WhatsApp",
        "✓" if wa.enabled else "✗",
        wa.bridge_url
    )

    dc = config.channels.discord
    table.add_row(
        "Discord",
        "✓" if dc.enabled else "✗",
        dc.gateway_url
    )

    # Feishu
    fs = config.channels.feishu
    fs_config = f"app_id: {fs.app_id[:10]}..." if fs.app_id else "[dim]not configured[/dim]"
    table.add_row(
        "Feishu",
        "✓" if fs.enabled else "✗",
        fs_config
    )

    # Mochat
    mc = config.channels.mochat
    mc_base = mc.base_url or "[dim]not configured[/dim]"
    table.add_row(
        "Mochat",
        "✓" if mc.enabled else "✗",
        mc_base
    )
    
    # Telegram
    tg = config.channels.telegram
    tg_config = f"token: {tg.token[:10]}..." if tg.token else "[dim]not configured[/dim]"
    table.add_row(
        "Telegram",
        "✓" if tg.enabled else "✗",
        tg_config
    )

    # Slack
    slack = config.channels.slack
    slack_config = "socket" if slack.app_token and slack.bot_token else "[dim]not configured[/dim]"
    table.add_row(
        "Slack",
        "✓" if slack.enabled else "✗",
        slack_config
    )

    console.print(table)


def _get_bridge_dir() -> Path:
    """Get the bridge directory, setting it up if needed."""
    import shutil
    import subprocess
    
    # User's bridge location
    user_bridge = Path.home() / ".nanobot" / "bridge"
    
    # Check if already built
    if (user_bridge / "dist" / "index.js").exists():
        return user_bridge
    
    # Check for npm
    if not shutil.which("npm"):
        console.print("[red]npm not found. Please install Node.js >= 18.[/red]")
        raise typer.Exit(1)
    
    # Find source bridge: first check package data, then source dir
    pkg_bridge = Path(__file__).parent.parent / "bridge"  # nanobot/bridge (installed)
    src_bridge = Path(__file__).parent.parent.parent / "bridge"  # repo root/bridge (dev)
    
    source = None
    if (pkg_bridge / "package.json").exists():
        source = pkg_bridge
    elif (src_bridge / "package.json").exists():
        source = src_bridge
    
    if not source:
        console.print("[red]Bridge source not found.[/red]")
        console.print("Try reinstalling: pip install --force-reinstall nanobot")
        raise typer.Exit(1)
    
    console.print(f"{__logo__} Setting up bridge...")
    
    # Copy to user directory
    user_bridge.parent.mkdir(parents=True, exist_ok=True)
    if user_bridge.exists():
        shutil.rmtree(user_bridge)
    shutil.copytree(source, user_bridge, ignore=shutil.ignore_patterns("node_modules", "dist"))
    
    # Install and build
    try:
        console.print("  Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=user_bridge, check=True, capture_output=True)
        
        console.print("  Building...")
        subprocess.run(["npm", "run", "build"], cwd=user_bridge, check=True, capture_output=True)
        
        console.print("[green]✓[/green] Bridge ready\n")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Build failed: {e}[/red]")
        if e.stderr:
            console.print(f"[dim]{e.stderr.decode()[:500]}[/dim]")
        raise typer.Exit(1)
    
    return user_bridge


@channels_app.command("login")
def channels_login():
    """Link device via QR code."""
    import subprocess
    from nanobot.config.loader import load_config
    
    config = load_config()
    bridge_dir = _get_bridge_dir()
    
    console.print(f"{__logo__} Starting bridge...")
    console.print("Scan the QR code to connect.\n")
    
    env = {**os.environ}
    if config.channels.whatsapp.bridge_token:
        env["BRIDGE_TOKEN"] = config.channels.whatsapp.bridge_token
    
    try:
        subprocess.run(["npm", "start"], cwd=bridge_dir, check=True, env=env)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Bridge failed: {e}[/red]")
    except FileNotFoundError:
        console.print("[red]npm not found. Please install Node.js.[/red]")


# ============================================================================
# Cron Commands
# ============================================================================

cron_app = typer.Typer(help="Manage scheduled tasks")
app.add_typer(cron_app, name="cron")


@cron_app.command("list")
def cron_list(
    all: bool = typer.Option(False, "--all", "-a", help="Include disabled jobs"),
):
    """List scheduled jobs."""
    from nanobot.config.loader import get_data_dir
    from nanobot.cron.service import CronService
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    jobs = service.list_jobs(include_disabled=all)
    
    if not jobs:
        console.print("No scheduled jobs.")
        return
    
    table = Table(title="Scheduled Jobs")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Schedule")
    table.add_column("Status")
    table.add_column("Next Run")
    
    import time
    for job in jobs:
        # Format schedule
        if job.schedule.kind == "every":
            sched = f"every {(job.schedule.every_ms or 0) // 1000}s"
        elif job.schedule.kind == "cron":
            sched = job.schedule.expr or ""
        else:
            sched = "one-time"
        
        # Format next run
        next_run = ""
        if job.state.next_run_at_ms:
            next_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(job.state.next_run_at_ms / 1000))
            next_run = next_time
        
        status = "[green]enabled[/green]" if job.enabled else "[dim]disabled[/dim]"
        
        table.add_row(job.id, job.name, sched, status, next_run)
    
    console.print(table)


@cron_app.command("add")
def cron_add(
    name: str = typer.Option(..., "--name", "-n", help="Job name"),
    message: str = typer.Option(..., "--message", "-m", help="Message for agent"),
    every: int = typer.Option(None, "--every", "-e", help="Run every N seconds"),
    cron_expr: str = typer.Option(None, "--cron", "-c", help="Cron expression (e.g. '0 9 * * *')"),
    at: str = typer.Option(None, "--at", help="Run once at time (ISO format)"),
    deliver: bool = typer.Option(False, "--deliver", "-d", help="Deliver response to channel"),
    to: str = typer.Option(None, "--to", help="Recipient for delivery"),
    channel: str = typer.Option(None, "--channel", help="Channel for delivery (e.g. 'telegram', 'whatsapp')"),
):
    """Add a scheduled job."""
    from nanobot.config.loader import get_data_dir
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronSchedule
    
    # Determine schedule type
    if every:
        schedule = CronSchedule(kind="every", every_ms=every * 1000)
    elif cron_expr:
        schedule = CronSchedule(kind="cron", expr=cron_expr)
    elif at:
        import datetime
        dt = datetime.datetime.fromisoformat(at)
        schedule = CronSchedule(kind="at", at_ms=int(dt.timestamp() * 1000))
    else:
        console.print("[red]Error: Must specify --every, --cron, or --at[/red]")
        raise typer.Exit(1)
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    job = service.add_job(
        name=name,
        schedule=schedule,
        message=message,
        deliver=deliver,
        to=to,
        channel=channel,
    )
    
    console.print(f"[green]✓[/green] Added job '{job.name}' ({job.id})")


@cron_app.command("remove")
def cron_remove(
    job_id: str = typer.Argument(..., help="Job ID to remove"),
):
    """Remove a scheduled job."""
    from nanobot.config.loader import get_data_dir
    from nanobot.cron.service import CronService
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    if service.remove_job(job_id):
        console.print(f"[green]✓[/green] Removed job {job_id}")
    else:
        console.print(f"[red]Job {job_id} not found[/red]")


@cron_app.command("enable")
def cron_enable(
    job_id: str = typer.Argument(..., help="Job ID"),
    disable: bool = typer.Option(False, "--disable", help="Disable instead of enable"),
):
    """Enable or disable a job."""
    from nanobot.config.loader import get_data_dir
    from nanobot.cron.service import CronService
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    job = service.enable_job(job_id, enabled=not disable)
    if job:
        status = "disabled" if disable else "enabled"
        console.print(f"[green]✓[/green] Job '{job.name}' {status}")
    else:
        console.print(f"[red]Job {job_id} not found[/red]")


@cron_app.command("run")
def cron_run(
    job_id: str = typer.Argument(..., help="Job ID to run"),
    force: bool = typer.Option(False, "--force", "-f", help="Run even if disabled"),
):
    """Manually run a job."""
    from nanobot.config.loader import get_data_dir
    from nanobot.cron.service import CronService
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    async def run():
        return await service.run_job(job_id, force=force)
    
    if asyncio.run(run()):
        console.print(f"[green]✓[/green] Job executed")
    else:
        console.print(f"[red]Failed to run job {job_id}[/red]")


# ============================================================================
# Status Commands
# ============================================================================


@app.command()
def doctor(
    check_only: bool = typer.Option(False, "--check", "-c", help="Run diagnostics only, don't start gateway"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed diagnostic info"),
):
    """Health check + start gateway with auto-reload.

    Default: run diagnostics, then start gateway with file-watch hot-reload.
    With --check: run diagnostics only and exit.
    """
    import importlib
    import time as _time

    from nanobot.config.loader import load_config, get_config_path

    # ── Phase 1: Health checks ────────────────────────────────────
    console.print(f"{__logo__} [bold]nanobot doctor[/bold]\n")

    config_path = get_config_path()
    all_ok = True

    # Config file
    if config_path.exists():
        console.print(f"  [green]✓[/green] Config: {config_path}")
    else:
        console.print(f"  [red]✗[/red] Config: {config_path} [red]missing[/red]")
        all_ok = False

    config = load_config()
    workspace = config.workspace_path

    # Workspace
    if workspace.exists():
        console.print(f"  [green]✓[/green] Workspace: {workspace}")
    else:
        console.print(f"  [red]✗[/red] Workspace: {workspace} [red]missing[/red]")
        all_ok = False

    # SOUL.md
    soul = workspace / "SOUL.md"
    if soul.exists():
        console.print(f"  [green]✓[/green] SOUL.md")
    else:
        console.print(f"  [yellow]![/yellow] SOUL.md not found (will use default persona)")

    # Model
    console.print(f"  [green]✓[/green] Model: {config.agents.defaults.model}")

    # Provider API keys
    from nanobot.providers.registry import PROVIDERS
    for spec in PROVIDERS:
        p = getattr(config.providers, spec.name, None)
        if p is None:
            continue
        if spec.is_local:
            if p.api_base:
                console.print(f"  [green]✓[/green] {spec.label}: {p.api_base}")
            elif verbose:
                console.print(f"  [dim]  {spec.label}: not set[/dim]")
        else:
            if p.api_key:
                console.print(f"  [green]✓[/green] {spec.label}: ****{p.api_key[-4:]}")
            elif verbose:
                console.print(f"  [dim]  {spec.label}: not set[/dim]")

    # Channels
    enabled = []
    for ch_name in ["telegram", "whatsapp", "discord", "feishu", "slack", "mochat", "dingtalk"]:
        ch_cfg = getattr(config.channels, ch_name, None)
        if ch_cfg and getattr(ch_cfg, "enabled", False):
            enabled.append(ch_name)
    if enabled:
        console.print(f"  [green]✓[/green] Channels: {', '.join(enabled)}")
    else:
        console.print(f"  [yellow]![/yellow] No channels enabled")

    # Import check
    try:
        importlib.import_module("nanobot.agent.loop")
        importlib.import_module("nanobot.agent.prompts")
        console.print(f"  [green]✓[/green] Imports OK")
    except Exception as e:
        console.print(f"  [red]✗[/red] Import error: {e}")
        all_ok = False

    if not all_ok:
        console.print("\n[red]Some checks failed. Fix the issues above before running.[/red]")
        if check_only:
            raise typer.Exit(1)

    console.print()

    # ── Phase 2: diagnostics-only exit ────────────────────────────
    if check_only:
        console.print("[green]All checks passed.[/green]")
        return

    # ── Phase 3: Run gateway with hot-reload (single control plane) ─
    console.print("[bold]Starting gateway with hot-reload...[/bold]\n")

    from nanobot.doctor.state import (
        load_state, save_state, new_run_id,
        record_crash, record_crash_loop, record_stable_run,
        should_attempt_rollback, record_rollback, STABLE_THRESHOLD_S,
    )
    from nanobot.doctor.notify import (
        make_event, append_event, send_telegram_alert,
    )
    from nanobot.doctor.git_ops import snapshot as git_snapshot, safe_rollback

    # Load doctor config
    doc_cfg = config.doctor
    MAX_CRASHES = doc_cfg.max_crashes
    CRASH_WINDOW = doc_cfg.crash_window_s
    stable_threshold = doc_cfg.stable_threshold_s

    # Telegram notification targets
    tg = config.channels.telegram
    tg_token = tg.token if tg.enabled else ""
    tg_chat_ids = tg.notify_chat_ids or tg.allow_from
    tg_proxy = tg.proxy

    # Doctor state
    state = load_state()
    run_id = new_run_id()
    state["current_run_id"] = run_id
    state["current_run_start"] = _time.monotonic()
    save_state(state)

    # Optional startup notification
    if doc_cfg.notify_on_startup and tg_token:
        evt = make_event("gateway", None, run_id=run_id, event_type="startup")
        append_event(evt)
        send_telegram_alert(tg_token, tg_chat_ids, evt, tg_proxy)

    # Detect nanobot package root for .py watching
    src_dir = Path(__file__).parent.parent  # nanobot/ package root
    # Stderr capture log for crash diagnosis
    stderr_log = Path.home() / ".nanobot" / "logs" / "gateway_stderr.log"
    stderr_log.parent.mkdir(parents=True, exist_ok=True)

    def _snapshot() -> dict[str, float]:
        """Get mtime snapshot of all .py files."""
        snap = {}
        for py in src_dir.rglob("*.py"):
            try:
                snap[str(py)] = py.stat().st_mtime
            except OSError:
                pass
        return snap

    crash_times: list[float] = []
    _reload_requested = False
    _stable_recorded = False
    _doctor_proc = None  # reference for signal handler

    def _doctor_sigint(signum, frame):
        """Ctrl+C handler: kill subprocess immediately and exit."""
        console.print("\n[dim]Shutting down...[/dim]")
        if _doctor_proc and _doctor_proc.poll() is None:
            _doctor_proc.kill()
            _doctor_proc.wait(timeout=5)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _doctor_sigint)

    while True:
        _reload_requested = False
        _stable_recorded = False
        stable_since = _time.monotonic()
        last_snap = _snapshot()
        console.print(
            f"[dim]{_time.strftime('%H:%M:%S')}[/dim] "
            f"[green]▶[/green] Gateway starting..."
        )

        proc = None
        stderr_fh = None
        try:
            # Start gateway subprocess with stderr captured to file
            stderr_fh = open(stderr_log, "w", encoding="utf-8")
            proc = _run_gateway_subprocess(stderr_fh)
            _doctor_proc = proc

            # Poll for file changes while gateway runs
            while proc.poll() is None:
                _time.sleep(1.5)

                # Check for file changes → hot-reload
                new_snap = _snapshot()
                changed = [
                    f
                    for f in set(list(last_snap.keys()) + list(new_snap.keys()))
                    if last_snap.get(f) != new_snap.get(f)
                ]
                if changed:
                    short = [Path(f).name for f in changed[:5]]
                    console.print(
                        f"[dim]{_time.strftime('%H:%M:%S')}[/dim] "
                        f"[yellow]↻[/yellow] Changed: {', '.join(short)} — reloading..."
                    )
                    _reload_requested = True
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        proc.kill()
                    break
                last_snap = new_snap

                # Check stable timer → record last_good_commit
                if (
                    not _stable_recorded
                    and (_time.monotonic() - stable_since) > stable_threshold
                ):
                    try:
                        gs = git_snapshot(str(src_dir.parent))
                        record_stable_run(state, gs.commit, gs.branch)
                        _stable_recorded = True
                        console.print(
                            f"[dim]{_time.strftime('%H:%M:%S')}[/dim] "
                            f"[green]✓[/green] Stable — recorded {gs.commit[:8]} as last_good"
                        )
                    except Exception as e:
                        _stable_recorded = True  # don't retry on error
                        if verbose:
                            console.print(f"[dim]  git snapshot skipped: {e}[/dim]")

            # Close stderr file before reading
            if stderr_fh:
                stderr_fh.close()
                stderr_fh = None

            exit_code = proc.returncode if proc and hasattr(proc, "returncode") else None

            # ── Reload path: skip crash counting ──
            if _reload_requested:
                console.print(
                    f"[dim]{_time.strftime('%H:%M:%S')}[/dim] "
                    f"[green]↻[/green] Reload complete."
                )
                continue

            # ── Clean exit ──
            if exit_code is None or exit_code == 0:
                continue

            # ── Crash path ──
            stderr_tail = ""
            try:
                stderr_tail = stderr_log.read_text(encoding="utf-8")[-500:]
            except Exception:
                pass

            now = _time.monotonic()
            crash_times.append(now)
            crash_times[:] = [t for t in crash_times if now - t < CRASH_WINDOW]

            # Create & log crash event
            evt = make_event(
                component="gateway",
                exit_code=exit_code,
                stderr_tail=stderr_tail,
                crash_count=len(crash_times),
                window_s=CRASH_WINDOW,
                run_id=run_id,
                event_type="crash",
            )
            append_event(evt)
            record_crash(state, evt)

            # Telegram notification (rate-limited)
            if doc_cfg.notify_on_crash and tg_token:
                send_telegram_alert(tg_token, tg_chat_ids, evt, tg_proxy)

            if len(crash_times) >= MAX_CRASHES:
                # ── Crash loop detected ──
                console.print(
                    f"[red]✗ {MAX_CRASHES} crashes in {CRASH_WINDOW}s — "
                    f"crash loop detected.[/red]"
                )

                loop_evt = make_event(
                    component="gateway",
                    exit_code=exit_code,
                    stderr_tail=stderr_tail,
                    crash_count=len(crash_times),
                    window_s=CRASH_WINDOW,
                    run_id=run_id,
                    event_type="crash_loop",
                )
                append_event(loop_evt)
                record_crash_loop(state)

                if tg_token:
                    send_telegram_alert(tg_token, tg_chat_ids, loop_evt, tg_proxy)

                # Auto-rollback (if enabled and applicable)
                if doc_cfg.auto_rollback and should_attempt_rollback(state):
                    target = state.get("last_good_commit", "")
                    console.print(
                        f"[yellow]Attempting rollback to {target[:8]}...[/yellow]"
                    )
                    cwd = str(src_dir.parent)
                    try:
                        gs = git_snapshot(cwd)
                        from_commit = gs.commit
                    except Exception:
                        from_commit = "unknown"

                    success, msg = safe_rollback(cwd, target)
                    record_rollback(state, from_commit, target, success)

                    rb_evt = make_event(
                        component="gateway",
                        exit_code=None,
                        stderr_tail=msg,
                        run_id=run_id,
                        event_type="rollback",
                    )
                    append_event(rb_evt)
                    if tg_token:
                        send_telegram_alert(tg_token, tg_chat_ids, rb_evt, tg_proxy)

                    if success:
                        console.print(f"[green]Rollback succeeded: {msg}[/green]")
                        crash_times.clear()
                        continue
                    else:
                        console.print(f"[red]Rollback failed: {msg}[/red]")

                # Wait for file change before retrying
                console.print("[dim]Waiting for file change...[/dim]")
                while True:
                    _time.sleep(1.5)
                    new_snap = _snapshot()
                    if any(
                        last_snap.get(f) != new_snap.get(f)
                        for f in set(list(last_snap.keys()) + list(new_snap.keys()))
                    ):
                        crash_times.clear()
                        break
                    last_snap = new_snap
            else:
                console.print(
                    f"[dim]{_time.strftime('%H:%M:%S')}[/dim] "
                    f"[red]✗[/red] Exit code {exit_code} — restarting in 2s..."
                )
                _time.sleep(2)

        except KeyboardInterrupt:
            console.print("\n[dim]Shutting down...[/dim]")
            if proc and hasattr(proc, "terminate"):
                proc.terminate()
            break
        finally:
            if stderr_fh:
                stderr_fh.close()


def _run_gateway_subprocess(stderr_fh=None):
    """Start `nanobot gateway` as a subprocess and return the Popen handle.

    stderr is piped through a background thread that tees every line to both
    the terminal (sys.stderr) and the log file (stderr_fh), so the user sees
    gateway output in real-time while crash logs are still captured to disk.
    """
    import subprocess
    import threading

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        [sys.executable, "-m", "nanobot", "gateway"],
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        env=env,
    )

    # Use raw bytes for terminal output to avoid Windows encoding issues
    stderr_raw = getattr(sys.stderr, "buffer", sys.stderr)

    def _tee_stderr():
        """Read stderr line by line, write to both terminal and log file."""
        try:
            for raw_line in iter(proc.stderr.readline, b""):
                # Write raw bytes to terminal (preserves UTF-8)
                try:
                    stderr_raw.write(raw_line)
                    stderr_raw.flush()
                except (TypeError, AttributeError):
                    sys.stderr.write(raw_line.decode("utf-8", errors="replace"))
                    sys.stderr.flush()
                # Write decoded text to log file
                if stderr_fh:
                    stderr_fh.write(raw_line.decode("utf-8", errors="replace"))
                    stderr_fh.flush()
        except (ValueError, OSError):
            pass  # pipe closed

    t = threading.Thread(target=_tee_stderr, daemon=True)
    t.start()
    proc._tee_thread = t  # keep reference to prevent GC

    return proc


@app.command()
def status():
    """Show nanobot status."""
    from nanobot.config.loader import load_config, get_config_path

    config_path = get_config_path()
    config = load_config()
    workspace = config.workspace_path

    console.print(f"{__logo__} nanobot Status\n")

    console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    console.print(f"Workspace: {workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")

    if config_path.exists():
        from nanobot.providers.registry import PROVIDERS

        console.print(f"Model: {config.agents.defaults.model}")
        
        # Check API keys from registry
        for spec in PROVIDERS:
            p = getattr(config.providers, spec.name, None)
            if p is None:
                continue
            if spec.is_local:
                # Local deployments show api_base instead of api_key
                if p.api_base:
                    console.print(f"{spec.label}: [green]✓ {p.api_base}[/green]")
                else:
                    console.print(f"{spec.label}: [dim]not set[/dim]")
            else:
                has_key = bool(p.api_key)
                console.print(f"{spec.label}: {'[green]✓[/green]' if has_key else '[dim]not set[/dim]'}")


if __name__ == "__main__":
    app()
