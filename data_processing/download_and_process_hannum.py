import os
import sys

import GEOparse
import pandas as pd
from loguru import logger
from tqdm import tqdm

# Configurable arguments
ROOT_DIR = "../data/ehan31/hannum/raw"  # Root directory for all data
LOG_LEVEL = "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Configure logger
os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(
    "logs/download_and_process_hannum.log",
    rotation="10 MB",
    level=LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)
logger.add(sys.stdout, level=LOG_LEVEL)


def process_data(gse: GEOparse.GSE) -> tuple:
    """Process GEO data and metadata."""
    logger.info("Processing GEO data and metadata...")

    metadata_list = []
    data_list = []

    for gsm_name, gsm in tqdm(gse.gsms.items(), desc="Processing GSMs"):
        # Extract all metadata
        metadata = {
            key: value[0] if isinstance(value, list) and len(value) == 1 else value
            for key, value in gsm.metadata.items()
        }

        # Extract characteristics_ch1 and add as individual columns
        if "characteristics_ch1" in metadata:
            characteristics = dict(item.split(": ", 1) for item in metadata["characteristics_ch1"])
            metadata.update(characteristics)
            del metadata["characteristics_ch1"]  # Remove the original characteristics_ch1 entry

        # Append metadata to list
        metadata_list.append(metadata)

        # Extract data
        data = gsm.table.set_index("ID_REF")["VALUE"]
        data.name = gsm_name
        data_list.append(data)

    # Create metadata DataFrame
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.index = list(metadata_df["geo_accession"])

    # Create data DataFrame
    data_df = pd.concat(data_list, axis=1)
    data_df.index.name = ""
    data_df = data_df.T

    # Flatten any remaining nested structures in the metadata
    metadata_df = metadata_df.apply(
        lambda x: x.apply(lambda y: y[0] if isinstance(y, list) and len(y) == 1 else y)
    )

    logger.info(f"Processed data shape: {data_df.shape}")
    logger.info(f"Processed metadata shape: {metadata_df.shape}")
    return data_df, metadata_df


def main():
    """Main function to orchestrate the Hannum data processing workflow."""
    logger.info("Starting Hannum data processing")

    os.makedirs(ROOT_DIR, exist_ok=True)

    # Initialize GEOparse object
    gse = GEOparse.get_GEO(geo="GSE40279", destdir=os.path.join(ROOT_DIR, "supplementary"))

    data_df, metadata_df = process_data(gse)

    # Create directories for betas and metadata
    os.makedirs(os.path.join(ROOT_DIR, "betas"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "metadata"), exist_ok=True)

    # Save data as gse_betas.arrow
    betas_file_path = os.path.join(ROOT_DIR, "betas", "gse_betas.arrow")
    data_df.to_feather(betas_file_path)
    logger.info(f"Saved betas data to {betas_file_path}")

    # Save metadata as metadata.arrow
    metadata_file_path = os.path.join(ROOT_DIR, "metadata", "metadata.arrow")
    metadata_df.to_feather(metadata_file_path)
    logger.info(f"Saved metadata to {metadata_file_path}")

    logger.success("Hannum data processing completed")


if __name__ == "__main__":
    main()
