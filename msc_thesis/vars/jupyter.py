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
                'tau_main': 2000.0,
                'age_main': 5000,
                'age_bq': 500.0,
                'r_sfr': 0.1,
                'sfr_A': 1.0,
                'normalise': True
            },
            'bc03': {
                'imf': 0,
                'metallicity': 0.02,
                'separation_age': 10
            },
            'nebular': {
                'logU': -2.0,
                'f_esc': 0.0,
                'f_dust': 0.0,
                'lines_width': 300.0,
                'emission': True
            },
            'dustatt_modified_starburst': {
                'E_BV_lines': [0.5, 0.3],
                'E_BV_factor': 0.44,
                'uv_bump_wavelength': 217.5,
                'uv_bump_width': 35.0,
                'uv_bump_amplitude': 0.0,
                'powerlaw_slope': 0.0,
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