# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinxcontrib.bibtex',
    'sphinx.ext.mathjax',
    'sphinx.ext.inheritance_diagram',
]
source_suffix = '.rst'
master_doc = 'index'
project = 'pymloc'
year = '2019-2020'
author = 'Daniel Bankmann'
copyright = '{0}, {1}'.format(year, author)
version = release = '0.1.0'

pygments_style = 'trac'
templates_path = ['.']
extlinks = {
    'issue':
    ('https://gitlab.tubit.tu-berlin.de/bankmann91/python-mloc/issues/%s',
     '#'),
    'pr': ('https://gitlab.tubit.tu-berlin.de/bankmann91/python-mloc/pull/%s',
           'PR #'),
}
# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only set the theme if we're building docs locally
    html_theme = 'sphinx_rtd_theme'

html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_sidebars = {
    '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = True

autosummary_generate = True
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True
}

mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

latex_docclass = {
    'howto': 'scrbook',
    'manual': 'scrbook',
}
