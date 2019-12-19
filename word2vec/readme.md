Word2Vec
======

This is my custom implementation of [Word2Vec](https://arxiv.org/abs/1301.3781).

**Example embeddings**

Word embeddings trained on the text of Charles Dickens' *David Copperfield*.

| Word       | Closest neighbours                                       |
| ---------- | -------------------------------------------------------- |
| `the`      | `our`, `his`, `my`, `your`, `a`                          |
| `i`        | `he`, `she`, `we`, `they`, `you`                         |
| `and`      | `or`, `nor`, `but`, `until`, `than`                      |
| `was`      | `is`, `were`, `had`, `has`, `looked`                     |
| `which`    | `whom`, `that`, `who`, `where`, `this`                   |
| `very`     | `too`, `extremely`, `so`, `pretty`, `prefectly`          |
| `could`    | `would`, `should`, `can`, `did`, `shall`                 |
| `said`     | `says`, `returned`, `cried`, `observed`, `replied`        |
| `on`       | `in`, `into`, `from`, `upon`, `by`                       |
| `never`    | `always`, `ever`, `certainly`, `can`, `really`           |
| `good`     | `long`, `true`, `pleasant`, `deeply`, `slight`           |
| `little`   | `young`, `painful`, `prostrate`, `warning`, `reproachful`|
| `peggotty` | `agnes`, `traddles`, `uriah`, `steerforth`, `ham`        |
| `,`        | `;`, `.`, `!`, `?`, `:`                                  |


**Implementation notes**

Overall, my implementation is fairly standard. The skipgrams are of the form
```
(left_context, target, right_context)
```

Given a (left_context, right_context) pair, and given a list of candidate targets,
the training objective is to predict which of the targets is the real target that slots in between the context words provided.
At training time, the network learns embeddings for the various words that, when fed to the classifier as inputs,
make this classification task easy.

There are however a few peculiarities about my implementation versus the standard one:

* I use a shared set of embedding vectors for the left context, target and right context.

* Rather than taking dot products between context vectors and target vectors to estimate their affinity,
I instead concatenate the context and target vectors and reduce this concatenated vector to an affinity score with the help of some dense layers.
This approach is inspired by the architecture of neural recommender system models.

There is no particular reason for preferring my architecture over the standard one. It is just a fun experiment!

More details can be found in the code.
