from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import activations, initializers, regularizers, constraints


class MoS(Layer):
    """
        Mixture of softmaxes (MoS) implementation.

        The mixture of softmaxes is described on:
            https://arxiv.org/abs/1711.03953
            by Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, William W. Cohen.

        The official implementation -- in pyTorch -- can be found in:
            https://github.com/zihangdai/mos

    """

    def __init__(self,
                 units,
                 n_softmaxes,
                 projection_activation='tanh',
                 expert_activation='softmax',
                 mixture_activation='softmax',
                 projection_initializer='glorot_uniform',
                 expert_initializer='glorot_uniform',
                 mixture_initializer='glorot_uniform',
                 use_projection_bias=True,
                 use_expert_bias=True,
                 use_mixture_bias=False,
                 projection_bias_initializer='zeros',
                 expert_bias_initializer='zeros',
                 mixture_bias_initializer='zeros',
                 projection_kernel_regularizer=None,
                 expert_kernel_regularizer=None,
                 mixture_kernel_regularizer=None,
                 projection_bias_regularizer=None,
                 expert_bias_regularizer=None,
                 mixture_bias_regularizer=None,
                 projection_kernel_constraint=None,
                 expert_kernel_constraint=None,
                 mixture_kernel_constraint=None,
                 projection_bias_constraint=None,
                 expert_bias_constraint=None,
                 mixture_bias_constraint=None,
                 **kwargs):
        """ Initialize the parameters for a MoS Layer

        Args:
         units: number classes
         n_softmaxes: number of softmaxes to use in the mixture.
         projection_activation: activation for the projections step.
         expert_activation: changing this from the default value changes this
            layer from mixture of softmaxes to a mixture of experts. Change it
            at your peril.
         mixture_activation: activation for the mixture step. Changing this
            affects the mixture weights. For example, mixture weights will not
            sum to 1 if 'Relu' is used.

        Other arguments are self-explanatory.
        """
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MoS, self).__init__(**kwargs)

        self.units = units
        self.n_softmaxes = n_softmaxes

        self.projection_activation = activations.get(projection_activation)
        self.expert_activation = activations.get(expert_activation)
        self.mixture_activation = activations.get(mixture_activation)

        self.projection_initializer = initializers.get(projection_initializer)
        self.expert_initializer = initializers.get(expert_initializer)
        self.mixture_initializer = initializers.get(mixture_initializer)

        self.use_projection_bias = use_projection_bias
        self.use_expert_bias = use_expert_bias
        self.use_mixture_bias = use_mixture_bias

        self.projection_bias_initializer = initializers.get(
            projection_bias_initializer)
        self.expert_bias_initializer = initializers.get(
            expert_bias_initializer)
        self.mixture_bias_initializer = initializers.get(
            mixture_bias_initializer)

        self.projection_kernel_regularizer = regularizers.get(
            projection_kernel_regularizer)
        self.expert_kernel_regularizer = regularizers.get(
            expert_kernel_regularizer)
        self.mixture_kernel_regularizer = regularizers.get(
            mixture_kernel_regularizer)

        self.projection_bias_regularizer = regularizers.get(
            projection_bias_regularizer)
        self.expert_bias_regularizer = regularizers.get(
            expert_bias_regularizer)
        self.mixture_bias_regularizer = regularizers.get(
            mixture_bias_regularizer)

        self.projection_kernel_constraint = constraints.get(
            projection_kernel_constraint)
        self.expert_kernel_constraint = constraints.get(
            expert_kernel_constraint)
        self.mixture_kernel_constraint = constraints.get(
            mixture_kernel_constraint)

        self.projection_bias_constraint = constraints.get(
            projection_bias_constraint)
        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.mixture_bias_constraint = constraints.get(mixture_bias_constraint)

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.projection_kernel = self.add_weight(
            shape=(input_dim, input_dim * self.n_softmaxes),
            initializer=self.projection_initializer,
            regularizer=self.expert_kernel_regularizer,
            constraint=self.expert_kernel_constraint,
            name='projection_kernel')

        self.expert_kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.expert_initializer,
            regularizer=self.expert_kernel_regularizer,
            constraint=self.expert_kernel_constraint,
            name='expert_kernel')

        self.mixture_kernel = self.add_weight(
            shape=(input_dim, self.n_softmaxes),
            initializer=self.mixture_initializer,
            regularizer=self.mixture_kernel_regularizer,
            constraint=self.mixture_kernel_constraint,
            name='mixture_kernel')

        if self.use_projection_bias:
            self.projection_bias = self.add_weight(
                shape=(input_dim * self.n_softmaxes,),
                initializer=self.projection_bias_initializer,
                regularizer=self.projection_bias_regularizer,
                constraint=self.projection_bias_constraint,
                name='projection_bias')
        else:
            self.projection_bias = None

        if self.use_expert_bias:
            self.expert_bias = self.add_weight(
                shape=(self.units,),
                initializer=self.expert_bias_initializer,
                regularizer=self.expert_bias_regularizer,
                constraint=self.expert_bias_constraint,
                name='expert_bias')
        else:
            self.expert_bias = None

        if self.use_mixture_bias:
            self.mixture_bias = self.add_weight(
                shape=(self.n_softmaxes,),
                initializer=self.mixture_bias_initializer,
                regularizer=self.mixture_bias_regularizer,
                constraint=self.mixture_bias_constraint,
                name='mixture_bias')
        else:
            self.mixture_bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        """ Run one step of MoS.

        Args:
            inputs: input Tensor, must be 2-D, `[batch, input_size]`

        Returns:
            A 2-D tuple of size `[batch, self.units]`
        """
        projection = K.dot(inputs, self.projection_kernel)
        if self.use_projection_bias:
            projection = K.bias_add(projection, self.projection_bias)
        if self.projection_activation:
            projection = self.projection_activation(projection)

        expert_output = K.reshape(projection,
                                  (-1, self.n_softmaxes, inputs.shape[1]))
        expert_output = K.dot(expert_output, self.expert_kernel)

        if self.use_expert_bias:
            expert_output = K.bias_add(expert_output, self.expert_bias)

        if self.expert_activation:
            expert_output = self.expert_activation(expert_output, axis=-1)

        mixtures = K.dot(inputs, self.mixture_kernel)

        if self.use_mixture_bias:
            mixtures = K.bias_add(mixtures, self.mixture_bias)

        if self.mixture_activation:
            mixtures = self.mixture_activation(mixtures)

        mixtures = K.expand_dims(mixtures, axis=-1)
        mixtures = K.repeat_elements(mixtures, self.units, axis=-1)

        expert_output = expert_output * mixtures

        out = K.sum(expert_output, axis=1)

        return out

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'n_softmax': self.n_softmax,
            'projection_activation': self.projection_activation,
            'expert_activation': self.expert_activation,
            'mixture_activation': self.mixture_activation,
            'projection_initializer': self.projection_initializer,
            'expert_initializer': self.expert_initializer,
            'mixture_initializer': self.mixture_initializer,
            'use_projection_bias': self.use_projection_bias,
            'use_expert_bias': self.use_expert_bias,
            'use_mixture_bias': self.use_mixture_bias,
            'projection_bias_initializer': self.projection_bias_initializer,
            'expert_bias_initializer': self.expert_bias_initializer,
            'mixture_bias_initializer': self.mixture_bias_initializer,
            'projection_kernel_regularizer':
                self.projection_kernel_regularizer,
            'expert_kernel_regularizer': self.expert_kernel_regularizer,
            'mixture_kernel_regularizer': self.mixture_kernel_regularizer,
            'projection_bias_regularizer': self.projection_bias_regularizer,
            'expert_bias_regularizer': self.expert_bias_regularizer,
            'mixture_bias_regularizer': self.mixture_bias_regularizer,
            'projection_kernel_constraint': self.projection_kernel_constraint,
            'expert_kernel_constraint': self.expert_kernel_constraint,
            'mixture_kernel_constraint': self.mixture_kernel_constraint,
            'projection_bias_constraint': self.projection_bias_constraint,
            'expert_bias_constraint': self.expert_bias_constraint,
            'mixture_bias_constraint': self.mixture_bias_constraint
        }
        base_config = super(MoS, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
