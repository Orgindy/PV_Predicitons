from utils.errors import ErrorAggregator, ValidationError


def test_error_aggregation():
    aggregator = ErrorAggregator()
    aggregator.add_error(ValidationError("test"))
    assert aggregator.has_errors()
