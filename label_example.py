from typing import Any
import numpy as np
from ampligraph.datasets import load_from_csv
from ampligraph.latent_features import ScoringBasedEmbeddingModel
from ampligraph.evaluation import mrr_score, hits_at_n_score
from ampligraph.latent_features.loss_functions import NLLMulticlass
from ampligraph.latent_features.regularizers import get as get_regularizer
from tensorflow.python.keras import metrics as metrics_mod
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import math_ops
import tensorflow as tf

class LabelNLLMulticlass(NLLMulticlass):

    def __call__(
        self, scores_pos, scores_neg, eta, regularization_losses=None, labels=None
    ):
        """Interface to external world.

        This function does the input checks, preprocesses input and finally applies loss function.

        Parameters
        ----------
        scores_pos : tf.Tensor
            A tensor of scores assigned to positive statements.
        scores_neg : tf.Tensor
            A tensor of scores assigned to negative statements.
        eta: tf.Tensor
           Number of synthetic corruptions per positive.
        regularization_losses: list
           List of all regularization related losses defined in the layers.

        Returns
        -------
        loss : tf.Tensor
            The loss value that must be minimized.
        """

        loss_values = []

        scores_neg = tf.reshape(scores_neg, [eta, -1])

        loss = self._apply_loss(scores_pos, scores_neg)
        loss_values.append(tf.reduce_sum(loss))
        if regularization_losses:
            regularization_losses = losses_utils.cast_losses_to_common_dtype(
                regularization_losses
            )
            reg_loss = math_ops.add_n(regularization_losses)
            loss_values.append(reg_loss)

        loss_values = losses_utils.cast_losses_to_common_dtype(loss_values)
        total_loss = math_ops.add_n(loss_values)
        self._loss_metric.update_state(total_loss)
        # tf.print(labels)
        return total_loss


train = load_from_csv("..","train.csv")
valid = load_from_csv("..","valid.csv")
test = load_from_csv("..","test.csv")

# all_train = np.concatenate([train, valid, test], axis=0)
all_train = train

labels = np.random.randint(0, 5, size=(train.shape[0], 1), dtype=np.int32)
all_train = np.concatenate([all_train, labels], axis=1)

# Initialize a ComplEx neural embedding model: the embedding size is k,
# eta specifies the number of corruptions to generate per each positive,
# scoring_type determines the scoring function of the embedding model.
model = ScoringBasedEmbeddingModel(k=400,
                                   eta=30,
                                   scoring_type='TransE',
                                   data_label=True
                                   )

# Optimizer, loss and regularizer definition
optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss = LabelNLLMulticlass()
regularizer = get_regularizer('LP', {'p': 2, 'lambda': 1e-4})

# Compilation of the model
model.compile(optimizer=optim, loss=loss, entity_relation_regularizer=regularizer)

# For evaluation, we can use a filter which would be used to filter out
# positives statements created by the corruption procedure.
# Here we define the filter set by concatenating all the positives

# Early Stopping callback

# Fit the model on training and validation set
model.fit(all_train,
          batch_size=int(all_train.shape[0] / 150),
          epochs=4000,                    # Number of training epochs
          verbose=True                  # Enable stdout messages
          )


ft = {'test': np.concatenate([train, test], axis=0)}

# save_model(model, "train_model.pkl")

ranks = model.evaluate(
    test,
    use_filter = ft,
    corrupt_side="s,o"
)

mrr = mrr_score(ranks)
hits_1 = hits_at_n_score(ranks, n=1)
hits_3 = hits_at_n_score(ranks,n=3)
hits_10=hits_at_n_score(ranks,n=10)
print(mrr, hits_1, hits_3,hits_10)

