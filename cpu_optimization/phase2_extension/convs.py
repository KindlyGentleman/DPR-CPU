"""Captured-shape convolution microbenchmarks against actual PyTorch/oneDNN."""

import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from ..api import configure_threads, load_reference
from ..benchmarks.common import parser, new_run, finish, write_json, digest
from ..quality import compare_outputs
from ..statistics import paired_summary
from .prototypes import prototype_conv, eligible_hot_convolution


def convolution_specs(model, inventory):
    specs = {}
    for entry in inventory:
        module = model.get_submodule(entry['name'])
        if not isinstance(module, torch.nn.Conv2d) or not entry['inputs']:
            continue
        incoming = entry['inputs'][0]
        description = {'shape': incoming['shape'], 'stride': incoming['stride'],
                       'dtype': incoming['dtype'], 'weight_shape': list(module.weight.shape),
                       'kernel': list(module.kernel_size), 'conv_stride': list(module.stride),
                       'padding': list(module.padding), 'dilation': list(module.dilation),
                       'groups': module.groups, 'padding_mode': module.padding_mode,
                       'bias': module.bias is not None}
        description['kernel_input_shape'] = list(incoming['shape'])
        if module.padding_mode != 'zeros':
            description['kernel_input_shape'][-2] += 2*module.padding[0]
            description['kernel_input_shape'][-1] += 2*module.padding[1]
        key = json.dumps(description, sort_keys=True)
        if key not in specs:
            specs[key] = {**description, 'module_names': [], 'calls_per_forward': 0}
        specs[key]['calls_per_forward'] += 1
        if entry['name'] not in specs[key]['module_names']:
            specs[key]['module_names'].append(entry['name'])
    return list(specs.values())


def functional_convolution(layer, x):
    padding = layer.padding
    if layer.padding_mode != 'zeros':
        x = F.pad(x, layer._reversed_padding_repeated_twice, mode=layer.padding_mode)
        padding = (0, 0)
    return F.conv2d(x, layer.weight, layer.bias, layer.stride, padding, layer.dilation, layer.groups)


def _primitive(layer):
    result = torch.nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size,
                             stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                             groups=layer.groups, bias=layer.bias is not None,
                             padding_mode=layer.padding_mode, dtype=layer.weight.dtype).eval()
    with torch.no_grad():
        result.weight.copy_(layer.weight)
        if layer.bias is not None:
            result.bias.copy_(layer.bias)
    return result


def _parity(reference, actual):
    if torch.equal(reference, actual):
        return {'allclose': True, 'max_abs': 0., 'relative_l2': 0., 'violating_elements': 0}
    ref, value = reference.numpy(), actual.numpy()
    result = compare_outputs(ref, value)
    if value.shape == ref.shape:
        result['violating_elements'] = int(np.count_nonzero(
            ~np.isfinite(value) | (np.abs(value.astype(float)-ref) > 1e-5+1e-4*np.abs(ref))))
    else:
        result['violating_elements'] = int(value.size)
    if not result['finite'] or value.shape != ref.shape:
        result.update(max_abs=None, relative_l2=None,
                      reason='nonfinite output or shape mismatch; retain numerical failure')
    return result


def main():
    arguments = parser(__doc__)
    arguments.add_argument('--profile', type=Path, required=True)
    arguments.add_argument('--max-wall-seconds', type=float, default=600)
    arguments.set_defaults(repeats=36)
    args = arguments.parse_args()
    if args.repeats < 10 or args.warmups < 3 or not 0 < args.max_wall_seconds <= 1800:
        raise ValueError('Use >=10 repeats, >=3 warmups and at most 1800 seconds for this screen')
    configure_threads(args.threads)
    run = new_run(args.id, 'phase2_convolution_primitives', args.results_dir, args.checkpoint)
    model = load_reference(args.checkpoint)
    inventory_path = args.profile/'tensor_inventory.json'
    specs = convolution_specs(model, json.loads(inventory_path.read_text(encoding='utf-8')))
    write_json(run/'shapes.json', specs)
    write_json(run/'inputs.json', {'profile_inventory_sha256': digest(inventory_path),
                                  'checkpoint': str(args.checkpoint), 'seed': 20261016,
                                  'values': 'seeded random logical inputs, actual frozen checkpoint weights; primitive-only, not WPC mask execution'})
    rng = random.Random(20261016)
    started = time.perf_counter()
    summaries = []
    raw_path = run/'raw_convs.csv'
    with raw_path.open('x', newline='', encoding='utf-8') as handle, torch.inference_mode(), torch.autocast('cpu', enabled=False):
        writer = csv.DictWriter(handle, fieldnames=['shape_id', 'repeat', 'order', 'method', 'seconds', 'rss_bytes'])
        writer.writeheader()
        for shape_id, spec in enumerate(specs):
            if time.perf_counter()-started >= args.max_wall_seconds:
                break
            original = model.get_submodule(spec['module_names'][0])
            layer = _primitive(original)
            generator = torch.Generator().manual_seed(20261016+shape_id)
            random_values = torch.randn(spec['shape'], generator=generator, dtype=layer.weight.dtype)
            x = torch.empty_strided(spec['shape'], spec['stride'], dtype=layer.weight.dtype).copy_(random_values)
            del random_values
            input_hash = hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
            methods = {'functional': lambda: functional_convolution(layer, x), 'module': lambda: layer(x)}
            setup = {key: 0. for key in methods}
            eligible = eligible_hot_convolution(original)
            prepared = {}
            if eligible:
                for kind in ('winograd', 'tiled', 'tiled_small', 'unfold'):
                    start = time.perf_counter()
                    prepared[kind] = prototype_conv(layer, kind)
                    setup[kind] = time.perf_counter()-start
                    methods[kind] = lambda kind=kind: prepared[kind](x)
            reference = methods['functional']()
            parity = {}
            for name, operation in methods.items():
                parity[name] = _parity(reference, operation())
                for _ in range(args.warmups):
                    operation()
            timings = {name: [] for name in methods}
            for repeat in range(args.repeats):
                order = list(methods)
                rng.shuffle(order)
                for rank, name in enumerate(order):
                    start = time.perf_counter_ns()
                    actual = methods[name]()
                    seconds = (time.perf_counter_ns()-start)*1e-9
                    timings[name].append(seconds)
                    writer.writerow({'shape_id': shape_id, 'repeat': repeat, 'order': rank,
                                     'method': name, 'seconds': seconds, 'rss_bytes': psutil.Process().memory_info().rss})
                    handle.flush()
                    del actual
            for name in methods:
                comparison = paired_summary(timings['functional'], timings[name], draws=2000, block=6)
                summaries.append({'shape_id': shape_id, 'method': name, 'input_shape': str(spec['shape']),
                                  'kernel_input_shape': str(spec['kernel_input_shape']), 'padding_mode': spec['padding_mode'],
                                  'weight_shape': str(spec['weight_shape']), 'source_module': spec['module_names'][0],
                                  'calls_per_forward': spec['calls_per_forward'], 'input_sha256': input_hash,
                                  'setup_s': setup[name], 'max_abs': parity[name]['max_abs'],
                                  'relative_l2': parity[name]['relative_l2'], 'strict_parity': parity[name]['allclose'],
                                  'parity_reason': parity[name].get('reason', ''),
                                  'violating_elements': parity[name]['violating_elements'], **comparison})
            with (run/'convs.csv').open('w', newline='', encoding='utf-8') as summary_handle:
                summary_writer = csv.DictWriter(summary_handle, fieldnames=list(summaries[0]))
                summary_writer.writeheader(); summary_writer.writerows(summaries)
            print(f"shape {shape_id+1}/{len(specs)} {spec['shape']} / {spec['weight_shape']} methods={list(methods)}", flush=True)
            del reference, x, layer, prepared, methods
    finish(run, {'status': 'completed' if len({r['shape_id'] for r in summaries}) == len(specs) else 'bounded_stopped',
                 'reason': 'Primitive-only randomized paired microbenchmarks; no full-model promotion',
                 'unique_shapes': len(specs), 'measured_shapes': len({r['shape_id'] for r in summaries}),
                 'warmups': args.warmups, 'repeats': args.repeats, 'wall_s': time.perf_counter()-started,
                 'boundary': 'forward inference, fixed input/weights, prototype transforms/tiling/reassembly included; frozen weight setup separate',
                 'memory_scope': 'sampled RSS across primitives; not isolated candidate/reference full-model peak'})


if __name__ == '__main__':
    from ..benchmarks.common import cli
    cli(main)
