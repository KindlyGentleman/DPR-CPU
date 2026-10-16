"""Instrumented stage/FourierUnit profile; timings are diagnostic and overlap explicitly."""

import csv
import types
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function as region
from .common import *


def fourier_forward(self, x):
    size = x.shape
    with region("FU.rfft2"):
        z = torch.fft.rfft2(x, norm="ortho")
    with region("FU.pack"):
        z = torch.view_as_real(z).permute(0, 1, 4, 2, 3).contiguous()
        z = z.view((size[0], -1) + z.shape[3:])
    with region("FU.spectral_conv"):
        z = self.conv_layer(z)
    with region("FU.batch_norm"):
        z = self.bn(z)
    with region("FU.relu"):
        z = self.relu(z)
    with region("FU.unpack"):
        z = z.view((size[0], -1, 2) + z.shape[2:]).permute(0, 1, 3, 4, 2).contiguous()
        z = torch.view_as_complex(z)
    with region("FU.irfft2"):
        return torch.fft.irfft2(z, s=size[2:], norm="ortho")


def wpc_forward(self, input, mask_in=None):
    """Original WPC operation order with named mask/feature scopes only."""
    size = tuple(input.shape)
    if mask_in is not None or self.last_size != size:
        self.last_size = size
        with torch.no_grad():
            if self.weight_maskUpdater.type() != input.type():
                self.weight_maskUpdater = self.weight_maskUpdater.to(input)
            if mask_in is None:
                mask = (
                    torch.ones(size[0], size[1], size[2], size[3]).to(input)
                    if self.multi_channel
                    else torch.ones(1, 1, size[2], size[3]).to(input)
                )
            else:
                mask = mask_in
            if self.trg_weight_model:
                if self.weight_model is None:
                    self.weight_model = self.get_weight_model(
                        size[2], size[3], sigma=min(size[2], size[3]) / self.object_size
                    ).to(input)
                with region("WPC.weighted_mask_update"):
                    self.update_mask = F.conv2d(
                        mask * self.weight_model,
                        self.weight_maskUpdater,
                        bias=None,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=1,
                    )
                with region("WPC.slide_weight"):
                    self.slide_weight = F.conv2d(
                        self.weight_model,
                        self.weight_maskUpdater,
                        bias=None,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=1,
                    )
            else:
                with region("WPC.mask_update"):
                    self.update_mask = F.conv2d(
                        mask.float(),
                        self.weight_maskUpdater,
                        bias=None,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=1,
                    )
                if self.slide_weight is None:
                    self.slide_weight = (
                        self.weight_maskUpdater.shape[1]
                        * self.weight_maskUpdater.shape[2]
                        * self.weight_maskUpdater.shape[3]
                    )
            with region("WPC.mask_ratio"):
                self.mask_ratio = torch.div(
                    self.slide_weight, torch.clamp(self.update_mask, min=1e-8)
                )
                self.update_mask = torch.ge(self.update_mask, 1e-8)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)
    with region("WPC.feature_conv"):
        raw_out = torch.nn.Conv2d.forward(
            self, torch.mul(input, mask) if mask_in is not None else input
        )
    with region("WPC.output_normalization"):
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)
    return (output, self.update_mask) if self.return_mask else output


def main():
    p = parser(__doc__)
    p.set_defaults(repeats=3)
    args = p.parse_args()
    configure_threads(args.threads)
    run = new_run(args.id, "reference_profile", args.results_dir, args.checkpoint)
    model = load_reference(args.checkpoint)
    case = load_cases(args.development)[0]
    x, m = prepare_inputs(case["intensity"], case["valid"])
    write_json(run / "inputs.json", cases_identity([case]))
    from deeppr.ffc import FourierUnit, SpectralTransform, FFC
    from deeppr.weightedpartialconv2d import WeightedPartialConv2d
    from deeppr.network import _fft2

    inventory = []
    tensor_inventory = []
    capturing = False
    activations = {}
    handles = []
    contexts = {}
    for name, module in model.named_modules():
        if isinstance(module, FourierUnit):

            def capture(mod, inp, n=name):
                if not capturing:
                    return
                a = inp[0]
                key = str(tuple(a.shape))
                inventory.append(
                    dict(
                        name=n,
                        shape=list(a.shape),
                        stride=list(a.stride()),
                        dtype=str(a.dtype),
                        contiguous=a.is_contiguous(),
                        channels_last=a.is_contiguous(
                            memory_format=torch.channels_last
                        ),
                    )
                )
                if key not in activations:
                    activations[key] = a.detach().numpy().copy()

            handles.append(module.register_forward_pre_hook(capture))
        if isinstance(
            module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, SpectralTransform)
        ):

            def capture_tensor(mod, inp, n=name):
                if not capturing:
                    return

                def flatten(values):
                    for a in values:
                        if isinstance(a, (tuple, list)):
                            yield from flatten(a)
                        elif isinstance(a, torch.Tensor):
                            yield a

                tensor_inventory.append(
                    dict(
                        name=n,
                        module_type=type(mod).__name__,
                        inputs=[
                            dict(
                                shape=list(a.shape),
                                stride=list(a.stride()),
                                dtype=str(a.dtype),
                                contiguous=a.is_contiguous(),
                                channels_last=a.is_contiguous(
                                    memory_format=torch.channels_last
                                ),
                            )
                            for a in flatten(inp)
                        ],
                    )
                )

            handles.append(module.register_forward_pre_hook(capture_tensor))
    with torch.inference_mode():
        for _ in range(args.warmups):
            model(x, m, false_scale=False)
        inventory.clear()
        capturing = True
        reference = model(x, m, false_scale=False).clone()
        capturing = False
    for hdl in handles:
        hdl.remove()
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, FourierUnit):
            module.forward = types.MethodType(fourier_forward, module)
        if isinstance(module, WeightedPartialConv2d):
            module.forward = types.MethodType(wpc_forward, module)
        if isinstance(
            module, (FourierUnit, SpectralTransform, FFC, WeightedPartialConv2d)
        ):

            def pre(mod, inp, n=name):
                ctx = region(f"module.{type(mod).__name__}:{n}")
                ctx.__enter__()
                contexts[id(mod)] = ctx

            def post(mod, inp, out):
                contexts.pop(id(mod)).__exit__(None, None, None)

            handles.extend(
                [
                    module.register_forward_pre_hook(pre),
                    module.register_forward_hook(post),
                ]
            )
    head, tail = model.model[:4], model.model[4:]

    def staged():
        with region("stage.normalization"):
            inp = torch.clamp(x * m, min=0)
            y = inp / torch.amax(inp, dim=(-2, -1), keepdim=True)
        with region("stage.wpc"):
            y = head((y, m))
        with region("stage.ffc_merge"):
            y = tail(y)
        with region("stage.projection"):
            projected = model.project_modulus(y, inp, m)
        with region("stage.refinement"):
            y = model.refine(torch.cat((y, projected), 1))
        with region("stage.scaling"):
            inten = torch.abs(_fft2(F.pad(y, (224, 224, 224, 224)))) ** 2
            scale = torch.sum(inp, dim=(-2, -1), keepdim=True) / torch.sum(
                inten * m, dim=(-2, -1), keepdim=True
            )
            return y * torch.sqrt(scale)

    with torch.inference_mode():
        actual = staged()
        if not torch.equal(reference, actual):
            raise RuntimeError("Instrumented forward differs from untouched reference")
        with profile(
            activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True
        ) as prof:
            for _ in range(args.repeats):
                staged()
    events = prof.key_averages(group_by_input_shape=True)
    rows = [
        dict(
            name=e.key,
            self_cpu_us=e.self_cpu_time_total,
            inclusive_cpu_us=e.cpu_time_total,
            count=e.count,
            input_shapes=str(e.input_shapes),
            self_cpu_memory=e.self_cpu_memory_usage,
        )
        for e in events
    ]
    with (run / "operators.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(run / "fourier_inventory.json", inventory)
    write_json(run / "tensor_inventory.json", tensor_inventory)
    np.savez_compressed(run / "fourier_inputs.npz", **activations)
    prof.export_chrome_trace(str(run / "trace.json"))
    self_total = sum(e.self_cpu_time_total for e in events)
    fft_time = sum(
        e.self_cpu_time_total
        for e in events
        if e.key in ["aten::_fft_r2c", "aten::_fft_c2r", "aten::_fft_c2c"]
    )
    finish(
        run,
        dict(
            status="inconclusive",
            reason="Instrumented diagnostics, not deployment latency",
            bitwise_instrumentation=True,
            repeats=args.repeats,
            warmups=args.warmups,
            fourier_calls_per_forward=len(inventory),
            fft_self_fraction=fft_time / self_total,
            fft_only_ideal_speedup_bound=1 / (1 - fft_time / self_total),
            self_cpu_total_us=self_total,
            scope_note="stage.* inclusive times are disjoint; FU.* overlap stage/module times; operator self times disjoint",
            scope_times_us={
                e.key: e.cpu_time_total / args.repeats
                for e in events
                if e.key.startswith(("FU.", "WPC.", "stage."))
            },
        ),
    )


if __name__ == "__main__":
    cli(main)
