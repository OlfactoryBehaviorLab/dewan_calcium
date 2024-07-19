from setuptools import setup, find_packages


setup(
	name='dewan_calcium',
	description='Dewan Calcium Imaging Toolbox',
	version='1.0',
	author='Dewan Lab, Florida State University',
	author_email='olfactorybehaviorlab@gmail.com',
	url='https://github.com/OlfactoryBehaviorLab/dewan_calcium',
	packages=['dewan_calcium'],
	python_requires=">=3.7, <3.11",
	install_requires=[
						'numpy<2.0.0',
						'pandas',
						'matplotlib',
						'scikit-learn',
						'scipy',
						'jupyter',
						'openpyxl',
						'pyarrow',
						'pathlib',
						'tqdl',
						'PySide6',
						'PyQtDarkTheme',
						'opencv-python',
						#'oasis-deconv'
						#'isx>=1.9.4'
						]
	)