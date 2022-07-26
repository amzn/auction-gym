## AuctionGym

### Configuration Files

AuctionGym uses JSON configuration files that detail configurations about the environment, the type of auction, and bidders' behaviour.

#### General Format

| Key  | Description |
| ------------- | ------------- |
| `random_seed` | The random seed that is used as input to the random number generator  |
| `num_runs` | The number of runs to repeat and average results over  |
| `num_iter` | The number of iterations, bidders update their beliefs every iteration and metrics are reported per iteration  |
| `rounds_per_iter` | The number of rounds per iteration |
| `num_participants_per_round` | The number of participants in every auction round |
| `embedding_size` | The dimensionality of the underlying context and item embeddings |
| `embedding_var` | The variance of the Gaussian distribution from which underlying embeddings are sampled  |
| `obs_embedding_size` | The dimensionality of the observable context embeddings to the bidders  |
| `allocation` | The type of allocation: currently `FirstPrice` and `SecondPrice` are supported  |
| `agents` | A list of agent configurations that describe bidders behaviour |
| `output_dir` | A path to a directory that will contain results. If it does not exist, AuctionGym will create this directory.  |


#### Agent Format

| Key  | Description |
| ------------- | ------------- |
| `name` | An identifier for the agent  |
| `num_copies` | The number of agents with this configuration (but unique item catalogues). A suffix will be appended to the name if num_copies > 1 |
| `num_items` | The number of items in the ad catalogue |
| `allocator` | The allocator decides which ad to allocate, given a context. It also outputs welfare estimates. |
| `bidder` | The bidder decides how to bid, given a welfare estimate, allocated ad and context.  |


Allocators have types, and possible keyword arguments supporting those types. Possible allocators are `OracleAllocator` and                                    `PyTorchLogisticRegressionAllocator`, which takes `embedding_size` and `num_items`.

Bidders can be one of `TruthfulBidder`, `ValueLearningBidder`, `PolicyLearningBidder` or `DoublyRobustBidder`.
