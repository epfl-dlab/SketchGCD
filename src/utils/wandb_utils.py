def object_to_dict(obj):
    """
    Convert an object to a dictionary, including its public properties.
    Exclude any methods or non-serializable attributes.
    """
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue  # Skip private properties and built-in attributes
        value = getattr(obj, key)
        # Check if the value is serializable (optional)
        if isinstance(value, (int, float, str, bool, list, dict, tuple, type(None))):
            result[key] = value
        elif hasattr(value, "__dict__"):
            result[key] = object_to_dict(value)  # Recursively convert objects
    return result


def args_to_string(args):
    # Define abbreviations for argument names
    abbreviations = {
        "model_name": "mdl",
        "dataset": "ds",
        "num_examples": "n",
        "batch_size": "bs",
        "prompter": "pr",
        "max_new_tokens": "mxt",
        "num_beams": "nb",
        "use_wandb": "wb",
    }

    # Create a compact string representation of each key arg
    args_dict = vars(args)
    key_args = [
        "model_name",
        "dataset",
        "prompter",
        "num_beams",
    ]  # Only include crucial arguments
    return "_".join(
        [
            f"{abbreviations.get(key, key)[:2]}{str(value).replace('/', '-').replace(' ', '')[:10]}"
            for key, value in args_dict.items()
            if key in key_args
        ]
    )
