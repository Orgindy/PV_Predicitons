from config import AppConfig


def test_config_validation():
    cfg = AppConfig.from_env()
    assert cfg.validate() is None
