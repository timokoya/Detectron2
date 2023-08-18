# Just to test if fiftyone is properly installed and working wel
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.utils.random as four

dataset = foz.load_zoo_dataset(
"quickstart"
    )
dataset.persistent = True
session = fo.launch_app(dataset, port =5151)
session.wait()