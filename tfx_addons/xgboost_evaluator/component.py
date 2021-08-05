from tfx import v1 as tfx
from tfx_addons.xgboost_evaluator import xgboost_predict_extractor

class XGBoostEvaluator(tfx.components.Evaluator):
  def __init__(self, **kwargs):
    kwargs["module_file"] = xgboost_predict_extractor.get_module_file()
    super().__init__(**kwargs)
