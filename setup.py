from setuptools import setup, find_packages

setup(name='dial-mpc',
      author="Haoru Xue",
      author_email="haoru-xue@berkeley.edu",
      packages=find_packages(include="dial_mpc"),
      version='0.0.1',
      install_requires=[
          'numpy',
          'matplotlib',
          'tqdm',
          'tyro',
          'jax',
          'jax-cosmo',
          'mujoco',
          'brax',
          'art',
      ],
      package_data={'dial-mpc': ['models/', 'examples/']},
      entry_points={
          'console_scripts': [
              'dial-mpc=dial_mpc.core.dial_core:main',
          ],
      },
      )
