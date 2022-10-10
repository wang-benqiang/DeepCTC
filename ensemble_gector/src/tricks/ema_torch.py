class EMA:
    """
      # 初始化
      ema = EMA(model, 0.999)
      ema.register()

      # 训练过程中，更新完参数后，同步update shadow weights
      def train():
          optimizer.step()
          ema.update()

      # eval前，apply shadow weights；eval之后，恢复原来模型的参数
      def evaluate():
          ema.apply_shadow()
          # evaluate
          ema.restore()
    """
    def __init__(self, model, decay=0.9999, update_after_steps=10, update_every_steps=1):
        self.model = model
        self.decay = decay
        self.update_after_steps = update_after_steps
        self.update_every_steps = update_every_steps
        self.ith_update_step = -1
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        self.ith_update_step += 1
        if self.ith_update_step >= self.update_after_steps and self.ith_update_step % self.update_every_steps ==0:
        
          for name, param in self.model.named_parameters():
              if param.requires_grad:
                  assert name in self.shadow
                  new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                  self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

