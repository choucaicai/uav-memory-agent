"""Global denominators and reporting for action and depth losses."""
import torch

TERM_NAMES = ('action', 'depth_reg', 'depth_edge')


def statistics_size(core):
    return 8


def distributed_loss(output, world_size, reduce_sum):
    terms = output['loss_terms']
    counts = torch.stack([terms[name][1].detach().float() for name in TERM_NAMES])
    reduce_sum(counts)
    return sum(terms[name][0] * terms[name][2] * world_size / counts[index].clamp_min(1e-6)
               for index, name in enumerate(TERM_NAMES))


def detached_statistics(output):
    values = [value.detach().double() for name in TERM_NAMES for value in output['loss_terms'][name][:2]]
    values += [output['depth_valid_count'].detach().double(), output['sample_count'].detach().double()]
    return torch.stack(values)


def describe_statistics(stats):
    if len(stats) != 8:
        raise ValueError('Unexpected statistics width')
    return {**{f'{name}_loss': float(stats[2 * index] / stats[2 * index + 1].clamp_min(1e-6))
               for index, name in enumerate(TERM_NAMES)},
            'depth_valid_fraction': float(stats[6] / stats[7].clamp_min(1))}
