#!/usr/bin/env python
import argparse
import logging
import os
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    # Create/Start W&B run
    run = wandb.init(project="exercise_5",
        group="attempts",
        job_type="process_data"
        )

    # Download input artifact
    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    # Read artifact into pandas DataFrame
    df = pd.read_parquet(artifact_path)

    # Drop duplicates
    logger.info("Dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=True)

    # Add new text feature
    df['title'].fillna(value='', inplace=True)
    df['song_name'].fillna(value='', inplace=True)
    df['text_feature'] = df['title'] + ' ' + df['song_name']

    # Save data to csv
    outfile = args.artifact_name
    df.to_csv(outfile)

    # Create output artifact
    output_artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description
        )
    
    # Attach data to output artifact
    output_artifact.add_file(outfile)

    # Upload artifact to W&B
    logger.info("Logging artifact")
    run.log_artifact(output_artifact)

    # Remove output data file
    os.remove(outfile)

    # Don't need to call run.finish() because when there is
    # only 1 W&B run in a script, it is automatically closed
    # when the script finishes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name",
        type=str,
        help="Name for the artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type for the artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)
