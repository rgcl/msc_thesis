from msc_thesis.util import call, config_update, context, merge
import os
import multiprocessing as mp
import click


@click.command()
@click.option('--target', default=None)
def cli(target):
    if not target:
        target = os.environ['THESIS_TARGET']
    main(target)


def main(target):

    with context(f'./data/{target}/models', target=target) as (env, _):

        vars = env.determining_filters

        call('pcigale init')

        config_update(vars['pcigale_init'], {
            'analysis_method': 'savefluxes',
            'cores': mp.cpu_count() - 1
        })

        call('pcigale genconf')

        config_update(vars['pcigale_genconf'], {
            'analysis_params': {
                'save_sed': True
            }
        })

        call('pcigale run')
