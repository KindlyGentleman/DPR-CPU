"""Separate untimed oneDNN verbose capture from a genuine DPR inference."""

import torch

from ..api import configure_threads, load_reference, prepare_inputs
from ..benchmarks.common import parser, new_run, load_cases, cases_identity, write_json, finish, cli
from ..quality import compare_outputs


def main():
    args = parser(__doc__).parse_args()
    configure_threads(args.threads)
    run = new_run(args.id, 'phase2_onednn_verbose', args.results_dir, args.checkpoint)
    case = load_cases(args.development)[0]
    write_json(run/'inputs.json', cases_identity([case]))
    model = load_reference(args.checkpoint)
    x, mask = prepare_inputs(case['intensity'], case['valid'])
    with torch.inference_mode(), torch.autocast('cpu', enabled=False), torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON):
        for _ in range(2):
            output = model(x, mask, false_scale=False)
    parity = compare_outputs(case['archived_reference'], output.numpy()[0, 0])
    finish(run, {'status': 'completed' if parity['allclose'] else 'numerical_failure',
                 'reason': 'Untimed oneDNN primitive log; two actual forwards, not a latency benchmark',
                 'archive_parity': parity, 'mkldnn_available': torch.backends.mkldnn.is_available(),
                 'mkldnn_enabled': torch.backends.mkldnn.enabled})


if __name__ == '__main__':
    cli(main)
