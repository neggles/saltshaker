import sys
from pathlib import Path
from typing import NoReturn, Optional

import typer

from saltshaker import __version__, console, web

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    if value:
        console.print(f"{__package__} v{__version__}")
        raise typer.Exit()


@app.callback()
def callback(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True, help="Show version"
    ),
) -> None:
    pass


@app.command()
def train(
    verbose: bool = typer.Option(
        False, "-v", "--verbose", is_flag=True, help="Enable verbose console output."
    ),
    config_path: Path = typer.Option(
        "./data/config.json",
        "-c",
        "--config",
        exists=True,
        dir_okay=False,
        help="Path to JSON config file. Will be created if it does not exist.",
    ),
) -> NoReturn:
    """
    Main entrypoint for your application.
    """
    console.log(f"verbose: {verbose}")
    console.log(f"config file path: {config_path}")
    console.log("done")
    sys.exit(0)


@app.command()
def launch(
    verbose: bool = typer.Option(
        False, "-v", "--verbose", is_flag=True, help="Enable verbose console output."
    ),
    host: str = typer.Option("0.0.0.0", "-h", "--host", help="Host address to bind to."),
    port: Optional[int] = typer.Option(None, "-p", "--port", help="Port to bind to."),
    share: bool = typer.Option(False, "-s", "--share", help="Enable public sharing via Gradio tunnel."),
) -> NoReturn:
    """
    Launch tbe Gradio webui.
    """
    console.log(f"verbose: {verbose}")
    web.launch(host=host, port=port, share=share)
    sys.exit(0)
