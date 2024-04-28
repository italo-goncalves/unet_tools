from setuptools import setup


# def readme():
#     with open('README.rst') as f:
#         return f.read()


setup(name='u-net tools',
      version='0.0.1',
      description='Utilities for training U-nets',
      #      long_description=readme(),
      keywords=['machine learning', 'U-net'],
      # url='http://github.com/italo-goncalves/geoML',
      author='Ítalo Gomes Gonçalves',
      author_email='italogoncalves.igg@gmail.com',
      license='GPL3',
      packages=['unet_tools'],
      package_dir={'unet_tools': 'src/unet_tools'},
      zip_safe=False,
      install_requires=['scikit-image', 'numpy', 'tensorflow'])
