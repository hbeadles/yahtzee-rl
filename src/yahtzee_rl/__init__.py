"""
Yahtzee RL - A reinforcement learning approach to Yahtzee.
"""
import typer
from yahtzee_rl.train.train_cli import app as train_app
from yahtzee_rl.evaluation.eval_cli import app as eval_app



cli = typer.Typer(help="Yahtzee RL command-line interface.")
cli.add_typer(train_app, name="train", help="Train RL agents on the Yahtzee environment.")
cli.add_typer(eval_app, name="eval", help="Evaluate trained RL agents on the Yahtzee environment.")

def main() -> None:
    """Entry point for the ``yahtzee-rl`` console script."""
    cli()
