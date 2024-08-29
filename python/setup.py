from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


def install_requires():
    with open("biomed_genai/requirements.txt", 'r') as f:
        return f.read().splitlines()


setup(name='biomed_genai',
      version='0.01',
      description='Medallion Architecture & models for use of Biomedical Articles in GenAI Applications',
      long_description=readme(),
      long_description_content_type='text/markdown',
      classifiers=['Programming Language :: Python :: 3.10',
                   'Topic :: System :: Distributed Computing',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Database :: Database Engines/Servers'],
      keywords='Spark GenAI Databricks RAG',
      url='databricks.com',
      author='Brad Barker',
      author_email='brad.barker@databricks.com',
      license='DATABRICKS',
      packages=find_packages(),
      package_data={'': ['requirements.txt']},
      install_requires=install_requires(),
      include_package_data=True,
      zip_safe=True)
