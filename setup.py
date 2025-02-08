from setuptools import setup, find_packages

setup(
    name='counterfactual_explanation_generator',
    version='0.1.0',
    description='A project demonstrating counterfactual explanation generation using Alibi for a logistic regression model on the Iris dataset',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/counterfactual-explanation-generator',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'scikit-learn>=0.24',
        'alibi>=0.6',
        'numpy>=1.18',
        'pandas>=1.0',
        'matplotlib>=3.0',
        'joblib>=0.14',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.6',
)
