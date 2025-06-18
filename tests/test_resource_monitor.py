from utils.resource_monitor import ResourceMonitor


def test_resource_monitoring():
    assert isinstance(ResourceMonitor.check_system_resources(), dict)
