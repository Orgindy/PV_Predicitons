import ast
import inspect


def load_compute_temperature_series():
    with open('app.py', 'r') as f:
        source = f.read()
    module = ast.parse(source)
    func_node = None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'compute_temperature_series':
            func_node = node
            break
    if func_node is None:
        raise RuntimeError('Function not found')
    func_source = '\n'.join(source.splitlines()[func_node.lineno-1:func_node.end_lineno])
    namespace = {}
    exec('import numpy as np', namespace)
    exec(func_source, namespace)
    return namespace['compute_temperature_series']


def test_ir_down_affects_temperature():
    func = load_compute_temperature_series()
    ghi = [800]
    tair = [300.0]
    wind = [2.0]
    zenith = [30]
    cfg = {}
    prof = {}
    low_ir = func(ghi, tair, [200.0], wind, zenith, cfg, prof, prof, prof)
    high_ir = func(ghi, tair, [500.0], wind, zenith, cfg, prof, prof, prof)
    assert high_ir.mean() > low_ir.mean()
