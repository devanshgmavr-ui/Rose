"""Rose - Autonomous AI Agent.

Main entry point for the Rose application.
Supports multiple launch modes:
  python run.py              - Interactive CLI mode
  python run.py --web        - Web server mode
  python run.py --ui         - PySide6 GUI mode
  python run.py --headless   - Headless mode for testing
  python run.py --status     - Show system status
"""

import os
import sys
import time
import signal
import logging
import argparse
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import ctypes
    ctypes.windll.kernel32.SetDllDirectoryW(str(ROOT / "llama_cpp_bin" / "bin"))
except Exception:
    pass

from agent.core.config import Config
from agent.core.agent import Agent
from agent.startup import StartupManager, StartupConfig


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def print_banner():
    print("""
    ███████╗███████╗███╗   ██╗████████╗
    ██╔════╝██╔════╝████╗  ██║╚══██╔══╝
    ███████╗█████╗  ██╔██╗ ██║   ██║
    ╚════██║██╔══╝  ██║╚██╗██║   ██║
    ███████║███████╗██║ ╚████║   ██║
    ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝
    Rose v1.0.0 - Autonomous AI Agent
    """)


def cmd_run(args):
    """Run Rose in the specified mode."""
    setup_logging(args.log_level)
    print_banner()

    config = Config()
    startup_config = StartupConfig(
        workspace_dir=str(ROOT / ".rose"),
        log_level=args.log_level,
        headless=args.headless,
        web_mode=args.web,
        web_port=args.port,
    )

    manager = StartupManager(startup_config)
    print("Running startup checks...")
    report = manager.run_startup()

    if report.errors:
        print("Startup errors:")
        for e in report.errors:
            print(f"  - {e}")
        return 1

    if report.warnings:
        print("Warnings:")
        for w in report.warnings:
            print(f"  - {w}")

    if report.first_run:
        print("First run detected! Configuration generated.")

    print(f"Startup complete ({report.startup_time:.1f}s)")
    print(f"Model: {report.model_path or 'Not found'}")
    print()

    agent = Agent(config=config)

    if args.web:
        return _run_web(agent, args)
    elif args.headless:
        return _run_headless(agent)
    else:
        return _run_interactive(agent)


def _run_interactive(agent):
    """Run in interactive CLI mode."""
    print("Starting Rose in interactive mode...\n")

    if not agent.initialize():
        print("Failed to initialize agent. Check logs for details.")
        return 1

    # Show system status
    health = agent.health_check()
    llm_status = health.get("llm", {})
    vision_status = health.get("vision", {})
    tools_status = health.get("tools", {})
    memory_status = health.get("memory", {})
    model_health = health.get("model_health", {})
    
    print("=" * 50)
    print("ROSE — LOCAL AUTONOMOUS AI")
    print("=" * 50)
    print(f"Model: {model_health.get('model_name', 'unknown')}")
    print(f"Vision: {'READY' if llm_status.get('vision_capable') else 'NOT AVAILABLE'}")
    print(f"GPU: {'READY' if llm_status.get('cuda_available') else 'CPU MODE'}")
    print(f"Memory: {'READY' if memory_status.get('initialized') else 'NOT AVAILABLE'}")
    print(f"Tools: {tools_status.get('count', 0)} registered")
    print("=" * 50)
    print()
    print("Rose is ready. Type 'help' for commands, 'quit' to exit.\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                break
            elif user_input.lower() == "help":
                _print_help()
                continue
            elif user_input.lower() == "status":
                health = agent.health_check()
                print(f"Status: {health.get('status', 'unknown')}")
                continue
            elif user_input.lower() == "tools":
                tools = agent.get_tool_info()
                print(f"Available tools ({len(tools)}):")
                for t in tools:
                    print(f"  - {t.get('name', 'unknown')}")
                continue

            try:
                response = agent.chat(user_input)
                print(f"Rose: {response.text}\n")
            except Exception as e:
                print(f"Error: {e}\n")

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        agent.shutdown()
        print("Goodbye!")

    return 0


def _run_web(agent, args):
    """Run in web server mode."""
    from agent.web.server import WebServer, WebConfig
    from agent.web.application import ApplicationService

    print(f"Starting web server on {args.host}:{args.port}...")

    app_service = ApplicationService(agent=agent)
    if not app_service.initialize():
        print("Failed to initialize application service.")
        return 1

    web_config = WebConfig(host=args.host, port=args.port, debug=args.debug)
    server = WebServer(config=web_config, app_service=app_service)

    def shutdown_handler(signum, frame):
        print("\nShutting down...")
        server.stop()
        app_service.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    server.start()
    print(f"Rose web interface running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.\n")

    try:
        while server.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        app_service.shutdown()

    return 0


def _run_headless(agent):
    """Run in headless mode (no UI, for testing)."""
    print("Starting Rose in headless mode...")
    if not agent.initialize():
        print("Failed to initialize.")
        return 1

    test_messages = [
        "Hello, Rose!",
        "What tools are available?",
        "Tell me about yourself.",
    ]

    print("Running headless test...\n")
    for msg in test_messages:
        print(f"Input: {msg}")
        try:
            response = agent.chat(msg)
            print(f"Output: {response.text[:200]}")
        except Exception as e:
            print(f"Error: {e}")
        print()

    agent.shutdown()
    print("Headless test complete.")
    return 0


def _print_help():
    print("""
Available commands:
  help     - Show this help message
  status   - Show agent status
  tools    - List available tools
  quit     - Exit Rose
  exit     - Exit Rose
  q        - Exit Rose
    """)


def cmd_status(args):
    """Show system status."""
    setup_logging("WARNING")
    startup_config = StartupConfig(
        workspace_dir=str(ROOT / ".rose"),
        skip_model_check=args.skip_model,
    )
    manager = StartupManager(startup_config)
    status = manager.get_quick_status()

    print("Rose System Status")
    print("=" * 40)
    print(f"First Run:      {'Yes' if status['first_run'] else 'No'}")
    print(f"Dependencies:   {'OK' if status['deps_ok'] else 'Missing'}")
    print(f"Models Found:   {status['models_found']}")
    print()

    if not args.skip_full:
        print("Running full check...")
        report = manager.run_startup()
        print(f"Success:        {'Yes' if report.success else 'No'}")
        print(f"Platform:       {report.platform_info.get('system', 'unknown')}")
        print(f"Python:         {report.platform_info.get('python_version', 'unknown')}")
        print(f"GPU:            {report.platform_info.get('gpu', 'unknown')}")
        print(f"Model:          {report.model_path or 'None'}")

        if report.dependencies:
            print(f"\nDependencies ({len(report.dependencies)}):")
            for dep in report.dependencies:
                status_icon = "OK" if dep.installed else "MISSING"
                print(f"  [{status_icon}] {dep.name} {dep.version or ''}")

        if report.warnings:
            print("\nWarnings:")
            for w in report.warnings:
                print(f"  - {w}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Rose - Autonomous AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run Rose")
    run_parser.add_argument("--web", action="store_true", help="Run in web server mode")
    run_parser.add_argument("--ui", action="store_true", help="Run with PySide6 GUI")
    run_parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    run_parser.add_argument("--host", default="127.0.0.1", help="Web server host")
    run_parser.add_argument("--port", type=int, default=8080, help="Web server port")
    run_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    run_parser.add_argument("--skip-model", action="store_true", help="Skip model check")
    run_parser.add_argument("--skip-deps", action="store_true", help="Skip dependency check")

    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.add_argument("--skip-model", action="store_true")
    status_parser.add_argument("--skip-full", action="store_true")

    args = parser.parse_args()

    if args.command == "status":
        return cmd_status(args)
    else:
        return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
