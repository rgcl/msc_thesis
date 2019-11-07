from msc_thesis.util import context
from pcigale.data import Database, Filter, _Filter as FilterTable
import os
import click
import numpy as np


@click.command()
@click.option('--target', default=None)
def cli(target):
    if not target:
        target = os.environ['THESIS_TARGET']
    main(target)


def main(target):
    with context(f'./data/{target}/filters', target=target) as (env, _):

        filters = create_filters(
            lambda_range_min=400.0,
            lambda_range_max=800.0,
            lambda_width=10.0
        )

        with Database(writable=True) as db:
            # I do not use base.add_filter, because that raise dont specific exception when there are duplicated
            # filter names. I need more like "insert or update" approach
            [db.session.merge(FilterTable(filter_)) for filter_ in filters]
            db.session.commit()


def create_filters(lambda_range_min, lambda_range_max, lambda_width, resolution=0.2):

    # determining the number of filters
    n_filters = np.ceil(np.ceil(lambda_range_max - lambda_range_min) / lambda_width)

    print(f'Making {n_filters} filters')

    filters = []
    for i in range(0, int(n_filters)):

        lambda_min = lambda_range_min + lambda_width * i
        lambda_max = lambda_min + lambda_width

        family = f'thesis-filter_{lambda_range_min}_{lambda_range_max}_{lambda_width}'
        name = f'{family}_{i}'

        wavelength = [lambda_min - 0.2, lambda_min, lambda_max, lambda_max + 0.2]
        transmission = [0., 1., 1., 0.]

        # Create the filter as a pcigale Filter instance
        filter_ = Filter(name, family, np.array([wavelength, transmission]))
        filter_.normalise()

        filters.append(filter_)

    return filters
