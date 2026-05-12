import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1] / 'ccatkidlib'))

project = 'ccatkidlib'
copyright = '2025, Darshan Patel'
author = 'Darshan Patel'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 
              'sphinx.ext.apidoc', 
              'sphinx.ext.napoleon',
              'sphinx.ext.coverage']

show_authors = False
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autoclass_content = 'both'
apidoc_modules = [
    {'path': '../../ccatkidlib', 'destination': './api'},
    {
        'exclude_patterns': ['**/test*', '**/kid_phase_fit*'],
        'max_depth': 6,
        'follow_links': False,
        'separate_modules': True,
        'include_private': False,
        'no_headings': True,
        'module_first': False,
        'implicit_namespaces': True,
        'automodule_options': {'members', 'show-inheritance'},
    },
]

rst_epilog =  """
.. |RFSoC| replace:: :term:`RFSoC`
.. |KID| replace:: :term:`KID`
.. |drone| replace:: :term:`drone`
.. |tone| replace:: :term:`tone`
.. |I| replace:: :math:`I`
.. |Q| replace:: :math:`Q`

"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_static_path = ['_static']
