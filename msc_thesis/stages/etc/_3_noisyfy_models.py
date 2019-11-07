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
        pass