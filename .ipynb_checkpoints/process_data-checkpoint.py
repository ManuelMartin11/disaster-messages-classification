import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import sqlalchemy

def run_etl():
    """Function that performs the ETL pipeline"""

    # Load data
    try:
        logging.debug("Loading data from csv...")
        mesdf = pd.read_csv(Path(r"data\messages.csv"))
        catdf = pd.read_csv(Path(r"data\categories.csv"))
    except Exception as ex:
        raise Exception(f"""There has been an exception during
                        data loading: {str(ex)}""")
    # Merge data
    logging.debug("Merging data from datasets...")
    df = mesdf.merge(catdf, on=catdf["id"], how="inner",
                     suffixes=["_mes", "_cat"])
    
    # Trnasform data
    logging.debug("Transforming categories...")
    catdf = catdf.categories.str.split(";", expand=True)
    row = catdf.iloc[0]
    
    category_colnames = row.apply(lambda x: x.split("-")[0])
    catdf.columns = category_colnames

    for col in catdf:
        catdf[col] = catdf[col].apply(lambda x: x.split("-")[1])
        catdf[col] = catdf[col].astype(int)

    df = df.drop("categories", axis=1)

    df = pd.concat([df, catdf], axis=1)
    df = df.drop_duplicates()
    df = df[df.related.isna() == False]

    try:
        logging.debug("Save data in database")
        engine = sqlalchemy.create_engine("sqlite:///disastermessages.db")
        df.to_sql(f"messages_{datetime.now().strftime('%d_%m_%y_%H')}", engine, index=False)
    except Exception as ex:
        raise Warning(f"Warning during database generation: {str(ex)}")

if __name__ == "__main__":
    run_etl()

    