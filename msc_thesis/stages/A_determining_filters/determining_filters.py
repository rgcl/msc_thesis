from ...util import call, config_update, context, merge
from pcigale.data import Filter, Database, _Filter as FilterTable
import numpy as np
import os
import multiprocessing as mp
import click
from collections import namedtuple
from astropy.table import Table
import matplotlib.pyplot as plt

FilterFamily = namedtuple('FilterFamily', 'family,filters,lambda_range_min,lambda_range_max,lambda_width')


@click.command()
@click.option('--target', default=None)
def cli(target):
    if not target:
        target = os.environ['THESIS_TARGET']
    main(target)


def main(target):

    # Fist, we start the context for this... "experiment"(?). We will work in a
    with context(f'./data/{target}/A_determining_filters', target=target, init=True) as ctx:

        analysis_for_filter_family_params(
            target,
            lambda_range_min=400.0,
            lambda_range_max=800.0,
            lambda_width=10.0,
            noisy_sigma=100.
        )

        #analysis_for_filter_family_params(
        #    target,
        #    lambda_range_min=200.0,
        #    lambda_range_max=1200.0,
        #    lambda_width=5.0,
        #    noisy_sigma=100.
        #)

        #analysis_for_filter_family_params(
        #    target,
        #    lambda_range_min=400.0,
        #    lambda_range_max=800.0,
        #    lambda_width=30.0,
        #    noisy_sigma=100.
        #)


def analysis_for_filter_family_params(
        target, lambda_range_min, lambda_range_max, lambda_width, noisy_sigma=1., noises_samples=10., do_plots=False):

    with context(f'./{lambda_range_min}_{lambda_range_max}_{lambda_width}', target) as ctx:

        # Create the filter family
        filter_family = create_filter_family(
            lambda_range_min=lambda_range_min,
            lambda_range_max=lambda_range_max,
            lambda_width=lambda_width
        )

        # using pcigale to create sintetic models of galaxies
        models = make_canonic_models(ctx, filter_family, save_sed=do_plots)

        # plotting if requested
        if do_plots:
            plot_filter_family(ctx, models[0], lambda_range_min, lambda_range_max, lambda_width, filter_family)

        # make the noised version of the canonical models
        models_noised = noisyfy_models(ctx, models, filter_family, noisy_sigma, noises_samples=noises_samples)

        # determine the props from the derived data
        models_resolved = analyse_noised_models(ctx, filter_family, models_noised)

        #
        r = determine_filter_family_error(ctx, models, models_resolved)
        print(r)


def create_filter_family(lambda_range_min, lambda_range_max, lambda_width):
    """
    Create a collection of filters refer as FilterFamily tuple.
    Non isolated: this save the filters in the database
    :param lambda_range_min: float
    :param lambda_range_max: float
    :param lambda_width: float
    :return: FilterFamily tuple
    """

    # determining the number of filters
    n_filters = np.ceil(np.ceil(lambda_range_max - lambda_range_min) / lambda_width)

    family_name = f'thesis-filter_{lambda_range_min}_{lambda_range_max}_{lambda_width}'
    print(f'Making {n_filters} filters for FilterFamily: {family_name}')

    filters = []
    with Database(writable=True) as db:
        for i in range(0, int(n_filters)):

            lambda_min = lambda_range_min + lambda_width * i
            lambda_max = lambda_min + lambda_width

            filter_name = f'{family_name}_{i}'

            # the filter shape is a rectangular
            wavelength = [lambda_min - 0.2, lambda_min, lambda_max, lambda_max + 0.2]
            transmission = [0., 1., 1., 0.]

            # Create the filter as a pcigale.data.Filter instance
            filter_ = Filter(filter_name, family_name, np.array([wavelength, transmission]))
            filter_.normalise()

            filters.append(filter_)
            db.session.merge(FilterTable(filter_))
        # confirm the filters in the database
        db.session.commit()

    return FilterFamily(family_name, np.array(filters), lambda_range_min, lambda_range_max, lambda_width)


def make_canonic_models(ctx, filter_family, save_sed=False):
    """
    Uses pcigale to generate synthetic data from models. Each model represent a galaxy, with the fluxes in the
    given filter_family filters.
    :param ctx:
    :param filter_family:
    :param save_sed:
    :return:
    """
    props = ctx.vars.determining_filters
    with context('canon', ctx.target):
        call('pcigale init')
        config_update(props['pcigale_init'], {
            'analysis_method': 'savefluxes',
            'cores': mp.cpu_count() - 1
        })
        call('pcigale genconf')
        config_update(props['pcigale_genconf'], {
            'analysis_params': {
                'save_sed': save_sed
            },
            'bands': [f.name for f in filter_family.filters]
        })
        call('pcigale run')

        # this is the generated models catalog with the properties as columns.
        return Table.read('out/models-block-0.fits')


def noisyfy_models(ctx, models, filter_family, noisy_sigma, noises_samples):
    """

    :param models:
    :param filter_family:
    :param noises_samples: number of noised clones of the original model.
    :return:
    """
    redshift = models[0]['universe.redshift']
    noised_table = Table(
        names=('id', *[f.name for f in filter_family.filters], 'redshift'),
        dtype=('<U60', *np.repeat('f4', len(filter_family.filters)), 'f4')
    )
    for model in models:
        for noisy_i in range(int(noises_samples)):
            noised_table.add_row((
                f'noised_{model["id"]}_{noisy_i}',
                *[
                    model[f.name] + np.random.normal(0, noisy_sigma)
                    for f in filter_family.filters
                ],
                redshift
            ))
    return noised_table


def analyse_noised_models(ctx, filter_family, noised_table):
    """

    :param ctx:
    :param filter_family:
    :param noised_table:
    :return:
    """

    with context('noised', ctx.target):
        noised_table_filename = 'data_file.txt'
        noised_table.write(noised_table_filename, format='ascii.fixed_width', delimiter=None)

        determining_filters = ctx.vars.determining_filters
        properties_of_interest = ctx.vars.properties_of_interest

        call('pcigale init')
        config_update(determining_filters['pcigale_init'], {
            'analysis_method': 'pdf_analysis',
            'data_file': noised_table_filename,
            'cores': mp.cpu_count() - 1
        })
        call('pcigale genconf')
        config_update(determining_filters['pcigale_genconf'], {
            'analysis_params': {
                'save_best_sed': False,
                'variables': properties_of_interest
            },
            'bands': [f.name for f in filter_family.filters]
        })
        call('pcigale run')

        return Table.read('out/models_resolved.txt', format='ascii')


def determine_filter_family_error(ctx, models, models_resolved):
    properties_of_interest = ctx.vars.properties_of_interest

    r = [models_resolved[f'best.{prop}'] for prop in properties_of_interest]


def plot_filter_family(ctx, model, lambda_range_min, lambda_range_max, lambda_width, filter_family):
    """
    Plot the canonical sed and the filter family
    :param ctx:
    :param model:
    :param lambda_range_min:
    :param lambda_range_max:
    :param lambda_width:
    :param filter_family:
    :return:
    """
    with context('canon', ctx.target):
        best_model0 = Table.read('out/0_best_model.fits')
        best_model0 = best_model0[best_model0['wavelength'] > lambda_range_min - 50.]
        best_model0 = best_model0[best_model0['wavelength'] < lambda_range_max + 50.]

        x = [f.pivot_wavelength for f in filter_family.filters]
        y = [model[f.name] for f in filter_family.filters]

        plt.figure()
        plt.title('$\lambda_{min}$: ' + str(lambda_range_min) +
                  ' $\lambda_{max}$: ' + str(lambda_range_max) +
                  ' $\lambda_{width}$: ' + str(lambda_width) + ' (nm)')
        plt.plot(x, y, '.', best_model0['wavelength'], best_model0['Fnu'], '-')
        plt.xlabel('$\lambda$ (nm)')
        plt.ylabel('F$\\nu$ (mJy)')
        plt.savefig('original_model0.pdf')
