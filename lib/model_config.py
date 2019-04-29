from lib.model import *


class ModelConfig:
  """
  The default setting is for MTT with se-multi.
  """

  def __init__(self, block='se', multi=True, num_blocks=9, init_features=128, num_convs=1,
               amplifying_ratio=0.125, dropout=0.5, activation='sigmoid', num_classes=50, weight_decay=0.):

    # Configure block specific settings.
    if block == 'basic':
      block_fn = basic_block
    elif block.startswith('rese'):
      num_convs = int(block[-1])
      block_fn = rese_block
    elif block.startswith('res'):
      num_convs = int(block[-1])
      amplifying_ratio = None
      block_fn = rese_block
    elif block == 'se':
      block_fn = se_block
    else:
      raise Exception(f'Unknown block name: {block}')

    # Overall architecture configurations.
    self.multi = multi
    self.init_features = init_features

    # Block configurations.
    self.block = block
    self.block_fn = block_fn
    self.num_blocks = num_blocks
    self.amplifying_ratio = amplifying_ratio
    self.num_convs = num_convs

    # Training related configurations.
    self.dropout = dropout
    self.activation = activation
    self.num_classes = num_classes
    self.weight_decay = weight_decay

  def get_signature(self):
    s = self.block
    if self.multi:
      s += '_multi'
    return s

  def print_summary(self):
    print(f'''=> {self.get_signature()} properties:
      block             : {self.block}
      multi             : {self.multi}
      num_blocks        : {self.num_blocks}
      amplifying_ratio  : {self.amplifying_ratio}
      dropout           : {self.dropout}
      activation        : {self.activation}
      num_classes       : {self.num_classes}''')
