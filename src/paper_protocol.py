from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_SEED = 42

def make_leave_one_out_purchase_splits(events_df: pd.DataFrame, min_interactions: int = 3):
    """
    Make leave-one-out splits for purchase events.
    """
    # Filter for transaction events
    trans = events_df[events_df["event"] == "transaction"].copy()
    # Count transactions per user
    user_counts = trans["visitorid"].value_counts()
    # Identify users with enough interactions
    valid_users = set(user_counts[user_counts >= min_interactions].index)
    
    # Filter the dataframe to include only valid users
    subset = events_df[events_df["visitorid"].isin(valid_users)].copy()
    # Sort the subset by user and timestamp
    subset = subset.sort_values(["visitorid", "timestamp"])
    
    # Initialize lists for train, validation, and test rows
    train_rows = []
    val_rows = []
    test_rows = []
    
    # Iterate over each user's data
    for uid, g in tqdm(subset.groupby("visitorid")):
        # Filter and sort transaction events for the user
        g_trans = g[g["event"] == "transaction"].sort_values("timestamp")

        # Convert transactions to a list of records
        trans_list = g_trans.to_dict("records")

        # Select the last transaction as the test item
        test_item = trans_list[-1]
        # Select the second to last transaction as the validation item
        val_item = trans_list[-2]
        # Get the timestamp of the validation item
        val_ts = val_item["timestamp"]

        # Select all events before the validation timestamp for training
        train_subset = g[g["timestamp"] < val_ts]
        
        # Identify the target items to exclude
        target_items = {val_item["itemid"], test_item["itemid"]}
        # Exclude target items from the training subset
        train_subset = train_subset[~train_subset["itemid"].isin(target_items)]
        
        # Add the training records to the list
        train_rows.extend(train_subset.to_dict("records"))

        # Add the validation item to the list
        val_rows.append(val_item)
        # Add the test item to the list
        test_rows.append(test_item)
        
    # Return the train, validation, and test dataframes
    return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows)



