# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

project = "pyobs-allsky-cloudcover"
copyright = "2026, Tim-Oliver Husser"
author = "Tim-Oliver Husser"

# -- General configuration ---------------------------------------------------

add_module_names = False

extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.autosectionlabel",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

source_suffix = ".rst"
master_doc = "index"
language = "en"
exclude_patterns = []
pygments_style = "sphinx"

# Be a little nitpicky
nitpicky = True
nitpick_ignore = []

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "display_version": False,
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "titles_only": False,
    "style_nav_header_background": "#cccccc",
}
html_logo = "_static/pyobs.gif"
