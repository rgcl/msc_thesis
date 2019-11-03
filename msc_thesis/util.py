import subprocess
import os
import configobj
from collections import namedtuple
from importlib import import_module
import shutil


def merge(a, b, path=None):
    """"merges b into a (https://stackoverflow.com/a/7205107)"""
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                a[key] = b[key]
                #raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def call(cmd):
    subprocess.call(cmd.split(' '))


def config_get(config_filename='pcigale.ini'):
    return configobj.ConfigObj(config_filename, encoding='UTF8')


def config_update(*props, config_filename='pcigale.ini'):
    """Update the pcigale config file (usually pcigale.ini). The properties are merged"""
    config = config_get(config_filename)
    for i in range(len(props)):
        merge(config, props[i])
    config.write()


"""
Context is a namedtuble where .vars is the variables of vars directory (referer from the target)
and .owd means "old working directory", that is the previous directory. This object is used as
with context(dir,target) as ctx:
    pass
"""
Context = namedtuple('Context', 'vars,owd,path,target,makedir,init')


class context:
    def __init__(self, path, target, makedir=True, init=False):
        self.path = os.path.abspath(path)
        self.target = target
        self.makedir = makedir
        self.init = init

    def __enter__(self):
        self.owd = os.getcwd() # old work directory
        vars = import_module(f'msc_thesis.vars.{self.target}')
        if self.init:
            shutil.rmtree(self.path, ignore_errors=True)
        if self.makedir:
            os.makedirs(self.path, exist_ok=True)
        os.chdir(self.path)
        return Context(vars, self.owd, self.path, self.target, self.makedir, self.init)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.owd)

