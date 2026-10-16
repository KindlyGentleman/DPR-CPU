"""Dynamo graph-break diagnostics, explicitly not successful Inductor compilation."""

import time
import traceback
import torch
from .common import *


def main():
    p = parser(__doc__)
    args = p.parse_args()
    configure_threads(args.threads)
    run = new_run(args.id, "dynamo_diagnostics", args.results_dir, args.checkpoint)
    model = load_reference(args.checkpoint)
    case = load_cases(args.development)[0]
    x, m = prepare_inputs(case["intensity"], case["valid"])
    write_json(run / "inputs.json", cases_identity([case]))
    try:
        with torch.inference_mode():
            model(x, m, false_scale=False)  # initialize WPC buffers before capture
            start = time.perf_counter()
            result = torch._dynamo.explain(model)(x, m, false_scale=False)
            elapsed = time.perf_counter() - start
        reasons = [str(r) for r in result.break_reasons]
        for i, graph in enumerate(result.graphs):
            (run / f"graph_{i}.py").write_text(graph.code, encoding="utf-8")
        finish(
            run,
            dict(
                status="inconclusive",
                reason="Dynamo capture diagnostics only; no native compiler or speedup implied",
                graph_count=result.graph_count,
                graph_break_count=result.graph_break_count,
                op_count=result.op_count,
                break_reasons=reasons,
                capture_s=elapsed,
                compile_times=str(result.compile_times),
            ),
        )
    except Exception as exc:
        (run / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finish(run, dict(status="blocked", reason=str(exc)))


if __name__ == "__main__":
    cli(main)
