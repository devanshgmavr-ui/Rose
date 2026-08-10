#!/usr/bin/env python3
"""Main entry point for the local agent."""

import sys
import os
import ctypes
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add CUDA bin directories to DLL search path (needed for llama.cpp CUDA backend)
_cuda_bin_dirs = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin",
]
for _cuda_dir in _cuda_bin_dirs:
    if os.path.isdir(_cuda_dir):
        try:
            ctypes.windll.kernel32.SetDllDirectoryW(_cuda_dir)
        except Exception:
            pass
        os.environ["PATH"] = _cuda_dir + ";" + os.environ.get("PATH", "")

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.logging import RichHandler

from agent.core.config import Config
from agent.core.agent import Agent


def setup_logging(config: Config):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format=config.log_format,
        handlers=[
            RichHandler(rich_tracebacks=True),
            logging.FileHandler(config.log_file, encoding='utf-8'),
        ]
    )


def print_welcome(config: Config, console: Console):
    """Print welcome message."""
    welcome_text = f"""
# Local AI Agent

**Version:** {config.version}
**Stage:** {config.stage}
**Model:** {config.model_name}

Type your message and press Enter to chat.
Type `quit` or `exit` to stop.
Type `clear` to clear conversation history.
Type `health` to check system status.
"""
    console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="green"))


def main():
    """Main function."""
    console = Console()
    
    try:
        # Load configuration
        console.print("[bold blue]Loading configuration...[/]")
        config = Config()
        setup_logging(config)
        
        # Create and initialize agent
        console.print("[bold blue]Initializing agent...[/]")
        agent = Agent(config)
        
        if not agent.initialize():
            console.print("[bold red]Failed to initialize agent. Check logs for details.[/]")
            sys.exit(1)
        
        console.print("[bold green]Agent ready![/]")
        print_welcome(config, console)
        
        # Main loop
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]You[/]")
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("[bold yellow]Goodbye![/]")
                    break
                
                if user_input.lower() == 'clear':
                    agent.clear_history()
                    console.print("[bold yellow]History cleared.[/]")
                    continue
                
                if user_input.lower() == 'health':
                    status = agent.health_check()
                    console.print(Panel(
                        str(status),
                        title="Health Status",
                        border_style="blue"
                    ))
                    continue
                
                if not user_input.strip():
                    continue
                
                # Generate response
                with console.status("[bold green]Thinking...[/]"):
                    response = agent.chat(user_input)
                
                # Print response
                console.print(f"\n[bold green]Agent[/]")
                console.print(Markdown(response.text))
                
                # Show stats if verbose
                if config.agent_verbose:
                    console.print(
                        f"[dim]Tokens: {response.tokens_used} | "
                        f"Time: {response.metadata.get('elapsed_seconds', 0):.2f}s[/]"
                    )
                
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Interrupted. Type 'quit' to exit.[/]")
                continue
            except Exception as e:
                console.print(f"[bold red]Error: {e}[/]")
                logging.error(f"Chat error: {e}", exc_info=True)
    
    except Exception as e:
        console.print(f"[bold red]Fatal error: {e}[/]")
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        # Shutdown agent
        if 'agent' in locals():
            agent.shutdown()


if __name__ == "__main__":
    main()
