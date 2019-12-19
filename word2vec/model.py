import tensorflow as tf
import numpy as np
import re
import requests
import sys

URL = 'http://gutenberg.org/files/766/766-0.txt'
OOV_TOKEN = '<unk>'

def download_text():
    response = requests.get(URL)
    response.encoding = 'utf-8'
    text = response.content.decode('utf-8')
    
    start_index = text.index('CHAPTER 1.')
    end_index = text.index('End of the Project Gutenberg EBook')
    text = text[start_index: end_index]
    
    text = text.lower()
    text = re.sub("’", '', text)
    text = re.sub('([?.!,"-:;])', r' \1 ', text)
    text = re.sub('[^a-z?.!,"-:;]+', ' ', text)
    
    return text
    

def tokenize_text(text, max_vocab_size):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='', num_words=max_vocab_size, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts([text])
    tokens = tokenizer.texts_to_sequences([text])[0]
    return tokens, tokenizer


def create_skipgrams(tokens, num_neg_samples, batch_size, num_epochs):
    left_contexts = []
    right_contexts = []
    real_and_fake_targets = []
    labels = []

    shuffled_tokens = [np.array(tokens) for _ in range(num_neg_samples)]  # for fake targets
    for token_array in shuffled_tokens:
        np.random.shuffle(token_array)
    
    for n in range(1, len(tokens) - 1):
        left_contexts.append(tokens[n - 1])
        right_contexts.append(tokens[n + 1])
        real_and_fake_targets.append(
                [tokens[n]] + [shuffled_tokens[k][n] for k in range(num_neg_samples)])
        labels.append([1] + [0 for _ in range(num_neg_samples)])

    left_contexts = tf.data.Dataset.from_tensor_slices(np.array(left_contexts))
    right_contexts = tf.data.Dataset.from_tensor_slices(np.array(right_contexts))
    real_and_fake_targets = tf.data.Dataset.from_tensor_slices(np.array(real_and_fake_targets))
    labels = tf.data.Dataset.from_tensor_slices(np.array(labels))

    combined_dataset = tf.data.Dataset.zip(
            (left_contexts, right_contexts, real_and_fake_targets, labels))
    return combined_dataset.shuffle(10000) \
                           .batch(batch_size, drop_remainder=True) \
                           .repeat(num_epochs)


def build_model(num_neg_samples, vocab_size, embed_dims, embed_l2_reg,
                num_hidden_layers, hidden_dims):
    num_targets = 1 + num_neg_samples
    
    left_context_inputs = tf.keras.Input(shape=(), dtype='int32')
    right_context_inputs = tf.keras.Input(shape=(), dtype='int32')
    target_inputs = tf.keras.Input(shape=(num_targets,), dtype='int32')
    
    # Share the same embedding weights for left-context, right-context and targets.
    l2_reg = tf.keras.regularizers.l2(embed_l2_reg / (vocab_size * embed_dims))
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dims,
                                                embeddings_regularizer=l2_reg)
    left_context_embedded = embedding_layer(left_context_inputs)
    right_context_embedded = embedding_layer(right_context_inputs)
    target_embedded = embedding_layer(target_inputs)
    
    # Duplicate the context embeddings, so that can be paired up with any of the target keys.
    left_context_embedded_stacked = tf.keras.layers.RepeatVector(n=num_targets)(
            left_context_embedded)
    right_context_embedded_stacked = tf.keras.layers.RepeatVector(n=num_targets)(
            right_context_embedded)
    
    # Concatenate context and target vectors, then use dense layers to reduce to affinity score.
    context_target_embedded_concat = tf.keras.layers.Concatenate(axis=-1)(
            [left_context_embedded_stacked, right_context_embedded_stacked, target_embedded])
    
    context_target_hidden = context_target_embedded_concat
    for _ in range(num_hidden_layers):
        context_target_hidden = tf.keras.layers.Dense(units=hidden_dims, activation='relu')(
                context_target_hidden)
    
    logits = tf.keras.layers.Dense(units=1)(context_target_hidden)
    logits = tf.keras.layers.Reshape(target_shape=(num_targets,))(logits)
        # i.e. (batch_size, num_targets, 1) -> (batch_size, num_targets)
    
    # Take softmax to give probabilities of targets being real vs fake.
    probs = tf.keras.layers.Softmax(axis=-1)(logits)
    
    training_model = tf.keras.Model(
            inputs=[left_context_inputs, right_context_inputs, target_inputs],
            outputs=probs)
    
    return training_model, embedding_layer


@tf.function
def run_train_step(model, optimizer, left_contexts, right_contexts, targets, labels):
    with tf.GradientTape() as tape:
        probs = model([left_contexts, right_contexts, targets])
        loss = tf.losses.CategoricalCrossentropy()(labels, probs) + tf.add_n(model.losses)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss


def show_neighbours(embeddings, tokenizer, num_root_words, num_neighbours, save_path):
    normalised_embeddings = np.divide(
            embeddings,
            np.sqrt(np.sum(np.square(embeddings), axis=1, keepdims=True)))
    
    lines = []
    
    for root_index in range(1, num_root_words + 1):
        root_embedding = normalised_embeddings[root_index]
        displacements = np.subtract(normalised_embeddings, root_embedding)
        distances = np.sqrt(np.sum(np.square(displacements), axis=1))
        neighbour_indices = np.argsort(distances)[:num_neighbours]
        neighbour_words = [tokenizer.index_word[index] for index in neighbour_indices]
        neighbour_distances = distances[neighbour_indices]
        
        for word, dist in zip(neighbour_words, neighbour_distances):
            lines.append('{:20} : distance = {:.3f}\n'.format(word, dist))
        lines.append('\n')
    
    text = ''.join(lines)
    print(text)
    
    if save_path is not None:
        with open(save_path, 'w') as outfile:
            outfile.write(text)


def main(max_vocab_size, batch_size, num_epochs, num_neg_samples,
         embed_dims, embed_l2_reg, num_hidden_layers, hidden_dims,
         root_words_display, neighbours_display, save_path):
    '''
    Train Word2Vec model and display sample embeddings.
    
    :param max_vocab_size: maximum number of words to embed (will take most common ones)
    :param batch_size: training batch size
    :param num_epochs: number of training epochs
    :param num_neg_samples: number of fake target words to go with each real target word,
                            to create prediction task for training model to solve
    :param embed_dims: dimensionality of embedding space
    :param embed_l2_reg: strength of L2 regularisation on the embedding vectors
    :param num_hidden_layers: number of hidden layers to go from the embedding vectors
                              to the affinity scores
    :param hidden_dims: dimensionality of hidden layers
    :param root_words_display: number of words to display neighbours for
    :param neighbours_display: number of neighbours to display for each root word
    :param save_path: location to save output to
    '''
    text = download_text()
    tokens, tokenizer = tokenize_text(text, max_vocab_size)
    dataset = create_skipgrams(tokens, num_neg_samples, batch_size, num_epochs)
    
    training_model, embedding_layer = build_model(
            num_neg_samples, max_vocab_size, embed_dims, embed_l2_reg,
            num_hidden_layers, hidden_dims)
    
    optimizer = tf.optimizers.Adam()
    
    losses = []
    
    for batch_id, (left_contexts, right_contexts, targets, labels) in enumerate(dataset):
        loss = run_train_step(training_model, optimizer,
                              left_contexts, right_contexts, targets, labels)
        losses.append(loss)
        
        if (batch_id + 1) % 1000 == 0:
            ave_loss = sum(losses) / len(losses)
            print('After batch {}: loss = {:.3f}'.format(batch_id + 1, ave_loss))
            losses = []

    print('Training complete')
    show_neighbours(embedding_layer.weights[0].numpy(), tokenizer,
                    root_words_display, neighbours_display, save_path)
    
    
if __name__ == '__main__':
    save_path = sys.argv[1]
    
    main(max_vocab_size=5000,
         batch_size=256,
         num_epochs=100,
         num_neg_samples=8,
         embed_dims=64,
         embed_l2_reg=0.1,
         num_hidden_layers=2,
         hidden_dims=64,
         root_words_display=100,
         neighbours_display=8,
         save_path=save_path)
   