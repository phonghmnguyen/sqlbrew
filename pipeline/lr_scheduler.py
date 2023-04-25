class TransformerScheduledOPT:
    """
    Implements the Transformer's learning rate scheduler.

    Args:
        optimizer: An instance of a PyTorch optimizer.
        init_lr: The initial learning rate.
        d_model: The dimensionality of embedding vector.
        n_warmup_steps: The number of warmup steps.

    """
    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps, last_step=0):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = last_step
    
    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        self.optimizer.zero_grad()

    def _lr_gamma(self):
        """
        Calculates the learning rate gamma factor.
        gamma = d_model ** -0.5 * min(step_num ** -0.5, step_num * warmup_steps ** -1.5)

        Returns:
            The learning rate gamma factor.
        """
        return self.d_model ** (-0.5) * \
            min(self.n_steps ** (-0.5), self.n_steps * self.n_warmup_steps ** (-1.5))

    def get_lr(self):
        """
        Calculates the learning rate.

        Returns:
            The learning rate.
        """
        return self.init_lr * self._lr_gamma()
    
    def step(self):
        """
            Update parameters and learning rate.
        """

        self.n_steps += 1
        lr = self.get_lr()
        

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.optimizer.step()