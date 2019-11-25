from ...util import context, call, config_update
import multiprocessing as mp
import numpy as np
import os
import click


def minmax(el, v):
    return (el - np.min(v)) / (np.max(v) - np.min(v)) if np.max(v) - np.min(v) is not 0 else 0


@click.command()
@click.option('--target', default=None)
@click.option('--plotting', is_flag=True)
def cli(target, plotting):
    if not target:
        target = os.environ['THESIS_TARGET']
    main(target, plotting)


def main(target, plotting):
    with context(f'./data/{target}/200_creating_big_catalog', target=target, init=True) as ctx:
        make_models(ctx, True)


def make_models(ctx, save_sed=False):
    """
    :param ctx:
    :param save_sed:
    :return:
    """
    with ctx.using('./big_catalog'):
        props = ctx.vars.big_catalog
        call('pcigale init')
        config_update(props['pcigale_init'], {
            'analysis_method': 'savefluxes',
            'cores': mp.cpu_count() - 1
        })
        call('pcigale genconf')
        config_update(props['pcigale_genconf'], {
            'analysis_params': {
                'save_sed': save_sed
            }
        })
        call('pcigale check')

        # this is the generated models catalog with the properties as columns.
        #return Table.read('out/models-block-0.fits')