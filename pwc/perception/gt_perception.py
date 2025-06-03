
from pwc.perception.perception_model import PerceptionModel

class PerceptionGT(PerceptionModel):
    def __init__(self,
                 config,):
        self.model_name = "GT-Precption"

    def load_model(self):
        pass

    def get_map(self, input):
        return input['gt_grid'].astype('float32')
    
    def count_misdetected(self, gt, pred):
        return 0