import numpy as np

properties_of_interest = [
    'attenuation.powerlaw_slope',
    'attenuation.FUV',
    'sfh.sfr',
    'dust.luminosity',
    'stellar.m_star'
]

number_of_noises = 20

determining_filters = {
    'pcigale_init': {
        'sed_modules': [
            'sfhdelayedbq',
            'bc03',
            'nebular',
            'dustatt_modified_starburst',
            'redshifting'
        ]
    },
    'pcigale_genconf': {
        'savefluxes': {
            'save_sed': True
        },
        'sed_modules_params': {
            'sfhdelayedbq': {
                'tau_main': [2000, 4000],
                'age_main': 13000,
                'age_bq': [200.0, 500.0],
                'r_sfr': [2.50, 10.],
                'sfr_A': 1.0,
                'normalise': True
            },
            'bc03': {
                'imf': 1,
                'metallicity': 0.02,
                'separation_age': 10
            },
            'nebular': {
                'logU': -3.0,
                'f_esc': 0.0,
                'f_dust': 0.0,
                'lines_width': 300.0,
                'emission': True
            },
            'dustatt_modified_starburst': {
                'E_BV_lines': [.05, .6, .8],
                'E_BV_factor': [.25, .5, .75],
                'uv_bump_wavelength': 217.5,
                'uv_bump_width': 35.0,
                'uv_bump_amplitude': 3.0,
                'powerlaw_slope': [-1.2, -.1, .2],
                'Ext_law_emission_lines': 1,
                'Rv': 3.1,
                'filters': 'B_B90 & V_B90 & FUV',
            },
            'redshifting': {
                'redshift': 0
            }
        }
    }
}

big_catalog = {
    'pcigale_init': {
        'sed_modules': [
            'sfhdelayedbq',
            'bc03',
            'nebular',
            'dustatt_modified_starburst',
            'redshifting'
        ]
    },
    'savefluxes': {
        'save_sed': True
    },
    'pcigale_genconf': {
        'sed_modules_params': {
            'sfhdelayedbq': {
                'tau_main': list(np.linspace(2000, 4000, 30)),
                'age_main': 13000,
                'age_bq': list(np.linspace(200.0, 500.0, 30)),
                'r_sfr': list(np.linspace(2.50, 10., 30)),
                'sfr_A': 1.0,
                'normalise': True
            },
            'bc03': {
                'imf': 1,
                'metallicity': 0.02,
                'separation_age': 10
            },
            'nebular': {
                'logU': -3.0,
                'f_esc': 0.0,
                'f_dust': 0.0,
                'lines_width': 300.0,
                'emission': True
            },
            'dustatt_modified_starburst': {
                'E_BV_lines': list(np.linspace(.05, .8, 30)),
                'E_BV_factor': list(np.linspace(.25, .75, 30)),
                'uv_bump_wavelength': 217.5,
                'uv_bump_width': 35.0,
                'uv_bump_amplitude': 3.0,
                'powerlaw_slope': list(np.linspace(-1.2, .2, 30)),
                'Ext_law_emission_lines': 1,
                'Rv': 3.1,
                'filters': 'B_B90 & V_B90 & FUV',
            },
            'redshifting': {
                'redshift': 0
            }
        }
    }
}
