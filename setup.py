from setuptools import setup, find_packages  # or find_namespace_packages

setup(
    # ...
    packages=find_packages(
        # All keyword arguments below are optional:
        where='src',  # '.' by default
        include=['nr-openai-observability*'],  # ['*'] by default
    ),
    # install_requires=["newrelic", "openai>=0.8,<0.30", "tiktoken>=0.5.1"],
    # ...
)