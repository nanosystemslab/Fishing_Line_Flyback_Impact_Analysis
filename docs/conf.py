"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys
from typing import Any
from typing import Dict


sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project: str = "Fishing Line Material Properties Analysis"
copyright: str = "2025, Nanosystems Lab"
author: str = "Nanosystems Lab"
release: str = "0.0.1"

# Custom title for the HTML pages
html_title: str = f"{project}<br/>v{release} documentation"

# -- General configuration ---------------------------------------------------
extensions: list[str] = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path: list[str] = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output ------------------------------------------------
html_theme: str = "pydata_sphinx_theme"
html_static_path: list[str] = ["_static"]

# PyData Sphinx Theme configuration
html_theme_options: Dict[str, Any] = {
    # Header links
    "github_url": (
        "https://github.com/nanosystemslab/" "Fishing_Line_Material_Properties_Analysis"
    ),
    "use_edit_page_button": True,
    # Navigation and TOC
    "show_toc_level": 2,
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    # Site navigation
    "external_links": [
        {"name": "Nanosystems Lab", "url": "https://github.com/nanosystemslab"}
    ],
    # Footer
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    # Search
    "search_bar_text": "Search the docs...",
    # Page elements - simplified to avoid template errors
    "show_prev_next": True,
}

# Configure edit button
html_context: Dict[str, str] = {
    "github_user": "nanosystemslab",
    "github_repo": "Fishing_Line_Material_Properties_Analysis",
    "github_version": "main",
    "doc_path": "docs/",
}

# Custom CSS (optional)
html_css_files: list[str] = [
    # "custom.css",  # You can create this file in docs/_static/custom.css
]

# -- Options for autodoc ----------------------------------------------------
autodoc_default_options: Dict[str, Any] = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
    "exclude-members": "__weakref__",
}

# Generate autosummary stub files
autosummary_generate: bool = True
autosummary_imported_members: bool = True

# Mock imports for packages that might not be available during doc build
autodoc_mock_imports: list[str] = []

# -- Options for intersphinx extension --------------------------------------
intersphinx_mapping: Dict[str, tuple[str, None]] = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for MyST parser --------------------------------------------
myst_enable_extensions: list[str] = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    # "linkify",  # Removed - requires additional dependency
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

myst_heading_anchors: int = 3  # Generate anchors for headings up to level 3
