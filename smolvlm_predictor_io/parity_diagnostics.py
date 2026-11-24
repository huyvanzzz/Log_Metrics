"""Bounded failure-only diagnostics for native connector GEMM batching parity."""
import numpy as np
import torch


def comparison_report(batch, reference, rtol, atol):
    if batch.shape != reference.shape or batch.dtype != np.float32 or reference.dtype != np.float32:
        raise ValueError('Connector parity requires matching FP32 geometry')
    finite = bool(np.isfinite(batch).all() and np.isfinite(reference).all())
    report = dict(ok=False, finite=finite, shape=list(batch.shape), rtol=rtol, atol=atol)
    if not finite:
        return report, None
    error = batch.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(error)
    allowed = atol + rtol * np.abs(reference.astype(np.float64))
    mismatch = ~np.isclose(batch, reference, rtol=rtol, atol=atol)
    report.update(ok=bool(np.allclose(batch, reference, rtol=rtol, atol=atol)),
        max_abs_error=float(absolute.max()),
        relative_l2=float(np.linalg.norm(error.ravel()) / max(np.linalg.norm(reference.astype(np.float64).ravel()), 1e-30)),
        reference_abs_max=float(np.abs(reference).max()),
        reference_rms=float(np.sqrt(np.mean(reference.astype(np.float64)**2))),
        mismatched_elements=int(mismatch.sum()), total_elements=int(batch.size),
        mismatch_fraction=float(mismatch.mean()),
        max_tolerance_ratio=float((absolute / np.maximum(allowed, np.finfo(np.float64).tiny)).max()),
        camera_max_abs_error=[float(absolute[:, camera].max()) for camera in range(batch.shape[1])])
    return report, (absolute, allowed, mismatch)


def fp64_selected_reference(backbone, raw_batch, batch, reference, details):
    """Reindex one source camera at a time, then evaluate selected CPU FP64 dots.

    Original FP32 weights/input values are promoted exactly; the live connector
    is never cast or replaced. These selected elements do not certify all outputs.
    """
    absolute, allowed, mismatch = details
    # Bound output and extra computation, without sorting the entire payload.
    flat = absolute.ravel()
    count = min(16, len(flat))
    largest = np.argpartition(flat, len(flat)-count)[-count:]
    ratio = (absolute / np.maximum(allowed, np.finfo(np.float64).tiny)).ravel()
    largest_ratio = np.argpartition(ratio, len(ratio)-count)[-count:]
    ordinary = np.linspace(0, len(flat)-1, min(16, len(flat)), dtype=np.int64)
    indices = np.unique(np.concatenate((largest, largest_ratio, ordinary)))
    coordinates = [tuple(map(int, np.unravel_index(int(index), batch.shape))) for index in indices]
    connector = backbone.native.model.connector
    projection = connector.modality_projection.proj
    if not isinstance(projection, torch.nn.Linear) or projection.bias is not None:
        raise ValueError('FP64 diagnostic requires native bias-free linear connector')
    channels = sorted({index[-1] for index in coordinates})
    with torch.inference_mode():
        selected_weights = projection.weight.detach().index_select(0,
            torch.tensor(channels, device=projection.weight.device)).to(device='cpu', dtype=torch.float64)
        weights = dict(zip(channels, selected_weights))
        elements = []
        previous = None
        for index in coordinates:
            group, camera, token, channel = index
            if previous != (group, camera):
                source = torch.from_numpy(np.ascontiguousarray(raw_batch[group, camera], dtype=np.float32))[None]
                shuffled = connector.pixel_shuffle(source, connector.scale_factor)[0]
                previous = (group, camera)
            vector = shuffled[token].double()
            products = vector * weights[channel]
            exact = float(products.sum())
            abs_dot_sum = float(products.abs().sum())
            batched, single = float(batch[index]), float(reference[index])
            elements.append(dict(index=list(index), fp64_value=exact, batch_value=batched,
                singleton_value=single, batch_abs_error=abs(batched-exact),
                singleton_abs_error=abs(single-exact), batch_vs_singleton_abs_error=float(absolute[index]),
                original_allowed_error=float(allowed[index]), original_mismatch=bool(mismatch[index]),
                abs_dot_sum=abs_dot_sum,
                cancellation_ratio=abs_dot_sum/abs(exact) if exact != 0. else None))
    return dict(scope='selected_projection_elements_cpu_float64', selected_count=len(elements),
        selection='largest_absolute_and_tolerance_ratio_plus_evenly_spaced_elements',
        no_global_accuracy_verdict=True,
        batch_max_abs_error=max(x['batch_abs_error'] for x in elements),
        singleton_max_abs_error=max(x['singleton_abs_error'] for x in elements), elements=elements)
