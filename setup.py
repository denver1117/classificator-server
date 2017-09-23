"""
Run setup
"""

from distutils.core import setup

setup(name='classificator-server',
      version='0.1',
      description='Companion API and UI for the classificator package',
      url='https://github.com/denver1117/classificator-server',
      download_url='https://github.com/denver1117/classificator-server/archive/0.1.tar.gz',
      author='Evan Harris',
      author_email='emitchellh@gmail.com',
      license='MIT',
      packages=['serve'],
      zip_safe=False)
