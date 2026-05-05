from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
    __version__ = version("towbintools")
except PackageNotFoundError:
    __version__ = "unknown"
