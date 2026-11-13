from typer.testing import CliRunner

from nextcoder import __version__
from nextcoder.cli import app


def test_version_option_exits_before_model_configuration() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"NextCoder {__version__}"
