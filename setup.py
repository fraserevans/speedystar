from setuptools import setup
setup(
name='speedystar',
version='0.1',
description='Testing installation of Package',
url='#',
author='Fraser Evnas',
author_email='fraserevans1@gmail.com',
license='GNU GPLv3',
packages=['speedystar', 'speedystar/utils', 'speedystar/utils/imfmaster/imf','speedystar/utils/selectionfunctions'],
package_data = {'/speedystar/utils/': ['/speedystar/utils/*.txt']},
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
