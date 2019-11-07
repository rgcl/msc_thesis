from msc_thesis.util import call, config_update, context, config_get
from pcigale.data import Database
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

    with context(f'./data/{target}/fitting', target) as (env, _):

        vars = env.determining_filters

        # recover the filter families created for this thesis. The hack here is that the description
        # is the name of the family, where the name is the name of the filter itself
        with Database() as db:
            filter_families = [row[0] for row in db.session.execute(
                "select distinct description from filters where name like 'thesis-filter%'"
            )]

        for filter_family in filter_families:

            with context(f'./{filter_family}', target):

                call('pcigale init')

                config_update(vars['pcigale_init'], {
                    'analysis_method': 'pdf_analysis',
                    'cores': mp.cpu_count() - 1
                })

                call('pcigale genconf')

        #config_update(vars['pcigale_genconf'], {
#
#        })

#        call('pcigale run')
