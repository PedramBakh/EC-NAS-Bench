import importlib, sys
# re-export vendors.ec_nasbench.nasbench as top-level "nasbench"
sys.modules["nasbench"] = importlib.import_module("ecnas.vendors.ec_nasbench.nasbench")
