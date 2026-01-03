import json
from RsInstrument import RsInstrument

def load_rs_config(path="RS_config.json"):
    with open(path, "r") as f:
        return json.load(f)

def connect_to_instrument(name=None, config=None):
    if config is None:
        config = load_rs_config()

    if name is None:
        name = config.get("default_instrument")

    inst_cfg = config["instruments"][name]

    options = f"SelectVisa='{inst_cfg['visa_library']}'"

    instr = RsInstrument(
        inst_cfg["resource"],
        id_query=inst_cfg.get("id_query", True),
        reset=inst_cfg.get("reset", False),
        options=options
    )

    return instr

# Example usage
config = load_rs_config()
instr = connect_to_instrument("NRQ6", config)
print(instr.query("*IDN?"))
instr.close()
