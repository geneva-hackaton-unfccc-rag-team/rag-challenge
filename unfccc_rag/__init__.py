from importlib.metadata import version

import coloredlogs

coloredlogs.install(level="INFO")

__version__ = version("unfccc-rag")
