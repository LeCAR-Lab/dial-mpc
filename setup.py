from setuptools import setup, find_packages

setup(name='dial-mpc',
      author="Haoru Xue",
      author_email="haoru-xue@berkeley.edu",
      packages=find_packages(include="dial_mpc"),
      version='0.0.2',
      install_requires=[
          'numpy<2.0.0',
          'matplotlib',
          'tqdm',
          'tyro',
          'jax[cuda12]',
          'jax-cosmo',
          'mujoco',
          'brax',
          'art',
          'emoji',
          'scienceplots'
      ],
      package_data={'dial-mpc': ['models/', 'examples/']},
      entry_points={
          'console_scripts': [
              'dial-mpc=dial_mpc.core.dial_core:main',
              'dial-mpc-sim2sim=dial_mpc.core.dial_sim2sim:main',
              'dial-mpc-sim2real=dial_mpc.core.dial_sim2real:main',
              'dial-mpc-sim=dial_mpc.deploy.dial_sim:main',
              'dial-mpc-real=dial_mpc.deploy.dial_real:main',
              'dial-mpc-plan=dial_mpc.deploy.dial_plan:main',
          ],
      },
      )
