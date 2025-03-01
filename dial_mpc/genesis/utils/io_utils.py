import importlib.resources


def load_dataclass_from_dict(dataclass, data_dict, convert_list_to_array=False):
    keys = dataclass.__dataclass_fields__.keys() & data_dict.keys()
    kwargs = {key: data_dict[key] for key in keys}
    if convert_list_to_array:
        import torch
        import genesis as gs

        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = torch.tensor(value, device=gs.device)
    return dataclass(**kwargs)


def get_example_path(example_name):
    with importlib.resources.path("dial_mpc.genesis.examples", example_name) as path:
        return path
