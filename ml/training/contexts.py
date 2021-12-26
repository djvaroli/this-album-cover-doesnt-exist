from abc import ABC


class BaseModelTrainingContext(ABC):
    """[summary]

    Args:
        ABC ([type]): [description]
    """
    
    def __init__(
        self,
    ):
        super().__init__()

    
class MNISTGANContext(BaseModelTrainingContext):
    """[summary]

    Args:
        BaseModelTrainingContext ([type]): [description]
    """
    
    def __init__(self, model_name: str = "mnist-gan"):
        super().__init__(model_name)