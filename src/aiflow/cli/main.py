from __future__ import annotations

import typer

from aiflow.flows.pipeline import inference_optimization_flow

app = typer.Typer(help="AIFlow CLI")


@app.command()
def hello() -> None:
    typer.echo("AIFlow CLI is ready.")


@app.command()
def run(s3_uri: str = typer.Argument(..., help="S3 URI to model, e.g. s3://bucket/key"),
        output_dir: str = typer.Option("./outputs", help="Directory to write results")) -> None:
    """
    Run the Prefect flow to process a model from S3.
    """
    result_path = inference_optimization_flow(s3_uri=s3_uri, output_dir=output_dir)
    typer.echo(f"Results written to: {result_path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()


