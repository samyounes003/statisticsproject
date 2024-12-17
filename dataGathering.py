import requests
import pandas as pd

def extract_relevant_features(data, exclude_keys=None):
    """
    Extract relevant features from a nested dictionary,
    excluding keys that contain unique identifiers like 'ID', 'slug', 'href', or 'links'.
    """

    filtered_data = {}

    for key, value in data.items():
        # Skip keys containing unique identifier substrings (case-insensitive)
        if any(excluded in key.lower() for excluded in exclude_keys):
            continue
        if key in exclude_keys:
            continue

        # Recursively process nested dictionaries
        if isinstance(value, dict):
            filtered_value = extract_relevant_features(value, exclude_keys)
            if filtered_value:  # Add only if not empty
                filtered_data[key] = filtered_value

        # Process lists and filter elements if they are dictionaries
        elif isinstance(value, list):
            filtered_list = [
                extract_relevant_features(item, exclude_keys) if isinstance(item, dict) else item
                for item in value
            ]
            filtered_data[key] = filtered_list

        # Add scalar values (e.g., strings, numbers)
        else:
            filtered_data[key] = value

    return filtered_data



def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary to a single-level dictionary with dot-separated keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k  # Append key with separator
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f"{new_key}[{i}]", sep=sep).items())
                else:
                    items.append((f"{new_key}[{i}]", item))
        else:
            items.append((new_key, v))
    return dict(items)

def dict_to_dataframe(data, exclude_keys=None):
    """
    Extract relevant features from nested dictionary and convert to a DataFrame.
    """
    # Step 1: Clean the dictionary
    cleaned_data = extract_relevant_features(data, exclude_keys)
    
    # Step 2: Flatten the dictionary
    flat_data = flatten_dict(cleaned_data)
    
    # Step 3: Convert to DataFrame
    df = pd.DataFrame([flat_data])  # Wrap flat_data in a list to make a single row
    return df


def fetch_all_housing_market():
    responses = []
    counter = 1
    while len(responses) < 6000:
        if counter > 8:
            break

        real_url = f"""https://api.boligsiden.dk/search/list/cases?addressTypes=villa%2Ccondo%2Cterraced+house%2Choliday+house%2Ccooperative%2Cfarm%2Chobby+farm%2Cfull+year+plot%2Cvilla+apartment%2Choliday+plot%2Chouseboat&sortBy=timeOnMarket&sortAscending=true&per_page=1000&page={counter}"""
        response = requests.get(real_url).json()
        
        responses += response['cases']

        counter += 1
    return responses


def make_housing_market_df(responses:list):
    dfs = []
    for i in range(len(responses)):
        df = dict_to_dataframe(responses[i], exclude_keys=["id", "slug", "href", "link", "image", "gstkvhx", "nextOpenHouse"])
        dfs.append(df)

    df = pd.concat(dfs).reset_index(drop=True)
    df.to_csv('DanishHousingMarket.csv', index=False)
    return df


# responses = fetch_all_housing_market()
# make_housing_market_df(responses=responses)