from config_utils import AppConfig
import yaml
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f) or {}
except FileNotFoundError:
    config = {}



def test_config_validation():
    cfg = AppConfig.from_env()
    assert cfg.validate() is None
