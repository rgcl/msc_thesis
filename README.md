# Determining the most important spectroscopic information for spectro-photometric modeling

> This is the code for my Astronomy MSc thesis. This is work in progress.

## Abstract

The emission of galaxies contains the imprint of some of their fundamental physical properties, such as their SFR, SFH, 
stellar mass, attenuation, dust mass, etc. Various techniques have been developed to extract this information, in 
particular the modeling of their UV-to-far-IR spectral energy distribution, or through the fit of their optical 
spectrum. Each approach has its own strengths and weaknesses. Combining both techniques, while promising the best of 
both worlds is however a challenge, partly because of the great discrepancy in the dimensionality of photometric and 
spectroscopic data. In this poster I will present the current status of a new effort towards addressing this problem, 
aiming in particular at determining subset of optical spectra that can be best combined with photometric observations.

## The code

This is the code for the investigation.

### Installation

Env: Python 3.6 or upper.

1. Install [pcigale](https://cigale.lam.fr/) (Python package)
2. Clone the repository
``$ git clone https://github.com/rgcl/msc_thesis.git`` 
3. Install ``$ pip install -e msc_thesis``

### Usage

> Caution: This code will install filters in pcigale database, prefixed as "thesis-filter_*".
> This does not interfere with the operation of pcigale. You can remove them later as indicated in uninstallation.

See msc_thesis/setup.py for the new commands. Current commands:

``$ msc-A [--target <target>] [--plotting]`` Perform the stage A for the investigation: Selecting the optimal filters parameters.

### Uninstallation

This is optional.

1. Delete the directory
2. Use some SQLite client, connect to <pcigale-installation-path>/pcigale/data/data.db (no pass required) and execute
```sql
delete from filters where name like '%thesis_filter%';
```

---------------------------------------------------------------------


_Un aporte de la UA al desarrollo sustentable de la Región de Antofagasta, a
través de la transferencia científica y tecnológica hacia los sectores
productivos, social y medioambiental._
PROYECTO ANT 1795, Universidad de Antofagasta. Chile.