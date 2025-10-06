import tensorflow as tf
import tensorflow_addons as tfa  # Ensure tensorflow-addons is installed

class Ranger(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=1e-3, weight_decay=1e-4, name="Ranger", **kwargs):
        super().__init__(name=name, **kwargs)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.radam = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
        self.lookahead = tfa.optimizers.Lookahead(self.radam, sync_period=5, slow_step=0.5)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
    
    def _resource_apply_dense(self, grad, var, apply_state=None):
        return self.lookahead.apply_gradients(zip(grad, var))
    
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        return self.lookahead.apply_gradients(zip(grad, var))
    
    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        return self.lookahead.apply_gradients(grads_and_vars)

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        })
        return config
