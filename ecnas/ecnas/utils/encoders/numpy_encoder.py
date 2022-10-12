import json
import jsbeautifier
import numpy as np

options = jsbeautifier.default_options()
options.indent_size = 2


class _NumpyEncoder(json.JSONEncoder):
    """Converts numpy objects to JSON-serializable format."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Matrices converted to nested lists
            return np.array2string(obj).split(sep="\n")
        elif isinstance(obj, np.generic):
            # Scalars converted to closest Python type
            return np.asscalar(obj)
        print(type(obj))
        return json.JSONEncoder.default(self, obj)
