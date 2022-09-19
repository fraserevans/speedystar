from setuptools import setup
setup(
name='speedystar',
version='1.0',
description='Initial Package',
long_description='This package allows you to generate, propagate and observe mock populations of hypervelocity stars. Please see github.com/fraserevans/speedystar for a comphrehensive README and example executables.',
author='Fraser Evans',
author_email='fraserevans1@gmail.com',
license='GNU GPLv3',
packages=['speedystar', 'speedystar/utils', 'speedystar/utils/imfmaster/imf','speedystar/utils/selectionfunctions'],
package_data = {'/speedystar/utils/': ['/speedystar/utils/*.txt']},
url = 'https://github.com/fraserevans/speedystar',
download_url = 'https://github.com/fraserevans/speedystar/archive/refs/tags/v1.tar.gz',
install_requires=[
'setuptools', 
'numpy',
'scipy', 
'scanninglaw', 
'tqdm',
#'os',
'astropy',
'galpy>=1.8.0',
'pandas',
'pygaia',
'amuse-framework',
'amuse-sse',
'healpy',
'h5py'
#'signal'
], 
include_package_data=True,
zip_safe=False)
