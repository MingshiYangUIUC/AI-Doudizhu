from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'search_action_space',
        ['search_action_space.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++'
    ),
]

setup(
    name='search_action_space',
    ext_modules=ext_modules,
)