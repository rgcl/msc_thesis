from ...util import call, config_update, context, merge
from pcigale.data import Filter, Database, _Filter as FilterTable
import numpy as np
import os
import multiprocessing as mp
import click
from collections import namedtuple
from astropy.table import Table
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from ...radarplot import radar_factory


FilterFamily = namedtuple('FilterFamily', 'family,filters,lambda_range_min,lambda_range_max,lambda_width')


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

    # Fist, we start the context for this... "experiment"(?). We will work in a
    with context(f'./data/{target}/A_determining_filters', target=target, init=True) as ctx:
        properties_of_interest = ctx.vars.properties_of_interest

        errors = {}
        errors_by_prop = {prop: {} for prop in properties_of_interest}
        for bin in np.arange(2., 8., step=1.):
            error, error_by_prop, master = analysis_for_filter_family_params(
                target,
                lambda_range_min=400.0,
                lambda_range_max=800.0,
                lambda_width=bin,
                noisy_sigma=1.,
                do_plots=plotting
            )
            errors[bin] = error
            for prop in properties_of_interest:
                errors_by_prop[prop][bin] = error_by_prop[prop]

        print("final errors")
        print(errors)

        print("final errors bt prop"),
        print(errors_by_prop)

        plt.figure()
        plt.plot(list(errors.keys()), list(errors.values()), '.')
        plt.xlabel('$\lambda$ (nm)')
        plt.ylabel('Relative residuals')
        plt.savefig('general_error.svg')

        plt.figure()
        for prop in properties_of_interest:
            plt.plot(list(errors_by_prop[prop].keys()), list(errors_by_prop[prop].values()), '.', label=prop)
        plt.xlabel('$\lambda$ (nm)')
        plt.ylabel('Relative residuals')
        plt.legend()
        plt.savefig('error_by_prop.svg')

        #np.savetxt('errors.npy', errors)
        #np.savetxt('errors_by_prop.npy', errors_by_prop)

        #analysis_for_filter_family_params(
#            target,
#            lambda_range_min=200.0,
#            lambda_range_max=1200.0,
#            lambda_width=5.0,
#            noisy_sigma=1.
#        )

#        analysis_for_filter_family_params(
#            target,
#            lambda_range_min=400.0,
#            lambda_range_max=800.0,
#            lambda_width=30.0,
#            noisy_sigma=1.
#        )


def analysis_for_filter_family_params(
        target, lambda_range_min, lambda_range_max, lambda_width, noisy_sigma=1., do_plots=False):

    with context(f'./{lambda_range_min}_{lambda_range_max}_{lambda_width}', target) as ctx:

        # Create the filter family
        filter_family = create_filter_family(
            lambda_range_min=lambda_range_min,
            lambda_range_max=lambda_range_max,
            lambda_width=lambda_width
        )

        # using pcigale to create sintetic models of galaxies
        print("making canonical models")
        models = make_canonical_models(ctx, filter_family, save_sed=do_plots)

        # plotting if requested
        if do_plots:
            plot_filter_family(ctx, models[0], lambda_range_min, lambda_range_max, lambda_width, filter_family)

        # make the noised version of the canonical models
        print("noising models")
        noised_models = noisyfy_models(ctx, models, filter_family, noisy_sigma)

        # determine the props from the derived data
        noised_models_predicted = predict_for_noised_models(ctx, filter_family, noised_models)

        # calculate the residuals
        error_by_prop, master = calculate_noised_models_prediction_mse(ctx, models, noised_models_predicted)

        theta = radar_factory(len(error_by_prop), frame='polygon')

        fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1, subplot_kw=dict(projection='radar'))
        colors = ['b', 'r', 'g', 'm', 'y']

        error = np.mean(list(error_by_prop.values()))

        ax.set_title('Mean Error for $\lambda_{width}$=' + str(lambda_width) + f'nm: {error:.2E}', weight='bold')

        # normalising the error
        err_val = list(error_by_prop.values())
        err_val = [minmax(val, err_val) for val in err_val]

        print(err_val)

        ax.plot(theta, err_val)
        ax.fill(theta, err_val)
        ax.set_varlabels(list(error_by_prop.keys()))
        plt.savefig(f'mse_{filter_family.family}.svg')

        return error, error_by_prop, master


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


def make_canonical_models(ctx, filter_family, save_sed=False):
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
        print('Run pcigale to compute the modules')
        call('pcigale run')

        # this is the generated models catalog with the properties as columns.
        return Table.read('out/models-block-0.fits')


def noisyfy_models(ctx, models, filter_family, noisy_sigma):
    """

    :param models:
    :param filter_family:
    :param noisy_sigma:
    :return:
    """
    number_of_noises = ctx.vars.number_of_noises
    redshift = models[0]['universe.redshift']
    noised_models = Table(
        names=('id', *[f.name for f in filter_family.filters], 'redshift'),
        dtype=('<U60', *np.repeat('f4', len(filter_family.filters)), 'f4')
    )
    for model in models:
        for noisy_i in range(number_of_noises):
            noised_models.add_row((
                f'noised_{model["id"]}_{noisy_i}',
                *[
                    model[f.name] + np.random.normal(0, noisy_sigma)
                    for f in filter_family.filters
                ],
                redshift
            ))
    return noised_models


def predict_for_noised_models(ctx, filter_family, noised_models):
    """

    :param ctx:
    :param filter_family:
    :param noised_models:
    :return: Table
    """

    with context('noised', ctx.target):
        noised_table_filename = 'data_file.txt'
        noised_models.write(noised_table_filename, format='ascii.fixed_width', delimiter=None)

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

        return Table.read('out/results.fits')


def calculate_noised_models_prediction_mse(ctx, models, noised_models_prediction):
    """

    :param ctx:
    :param models: astropy.table.Table
    :param noised_models_prediction: astropy.table.Table
    :return:
    """
    properties_of_interest = ctx.vars.properties_of_interest

    master = Table(
        names=(
            'model_id',
            'model_id_noised',
            *properties_of_interest,
            *[f'{prop}_predicted' for prop in properties_of_interest]
        ),
        dtype=('<U60', '<U60', *np.repeat('f8', len(properties_of_interest) * 2))
    )
    # complete the master table
    model_ids = models['id'].data
    for noised_model_prediction in noised_models_prediction:
        model_id_noised = noised_model_prediction['id']
        model_id = model_id_noised.split('_')[1]
        i, = np.where(model_ids == int(model_id))
        canonical_model = models[i]

        master.add_row((
            model_id,
            model_id_noised,
            *[canonical_model[prop] for prop in properties_of_interest],
            *[noised_model_prediction[f'best.{prop}'] for prop in properties_of_interest]
        ))

    # save the tables
    master.write('master.fits')

    # MSE calculated with all the galaxies # todo nrmse
    rtmse = {
        prop: np.sqrt(
            np.mean(np.square(master[f'{prop}_predicted'] - master[prop]))
        ) / (np.quantile(master[prop], .75) - np.quantile(master[prop], .25))
        for prop in properties_of_interest
    }

    return rtmse, master


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
