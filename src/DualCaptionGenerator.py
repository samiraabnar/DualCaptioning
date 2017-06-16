import tensorflow as tf
import numpy as np

class DualCaptionGenerator(object):

    def __init__(self):




    def define_network(self):
        """Builds the image model subgraph and generates image embeddings.
            Inputs:
              self.images
            Outputs:
              self.image_embeddings
            """
        xception_output =

        # Map inception output into embedding space.
        with tf.variable_scope("image_embedding") as scope:
            image_embeddings = tf.contrib.layers.fully_connected(
                inputs=inception_output,
                num_outputs=self.config.embedding_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                scope=scope)

        # Save the embedding size in the graph.
        tf.constant(self.config.embedding_size, name="embedding_size")

        self.image_embeddings = image_embeddings


        """Builds the input sequence embeddings.
            Inputs:
              self.input_seqs
            Outputs:
              self.seq_embeddings
            """
        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            embedding_map = tf.get_variable(
                name="map",
                shape=[self.config.vocab_size, self.config.embedding_size],
                initializer=self.initializer)
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

        self.seq_embeddings = seq_embeddings

        # Create the internal multi-layer cell for our RNN.
        def single_cell():
            return tf.contrib.rnn.GRUCell(self.hidden_dim)
        if self.cell_type == "LSTM":
            def single_cell():
                return tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)


        cell = single_cell()
        if self.number_of_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.number_of_layers)])



        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
          return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
              encoder_inputs,
              decoder_inputs,
              cell,
              num_encoder_symbols=self.input_dim,
              num_decoder_symbols=self.output_dim,
              embedding_size=self.embedding_size,
              output_projection=output_projection,
              feed_previous=do_decode,
              dtype=dtype)
