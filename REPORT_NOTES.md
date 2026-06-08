# NEAT SlimeVolley Report Notes

These notes summarize the main implementation decisions and improvements made
during development. They are written as report material, not as user
instructions.

## Baseline Implementation

The initial agent used a minimal feed-forward NEAT genome:

- 12 input nodes for the SlimeVolley observation vector
- 1 bias node with constant value `1`
- 3 output nodes for the `MultiBinary(3)` action
- direct weighted connections from every input and the bias node to each output

The network forward pass evaluates enabled connections, applies each node's
operation/activation, and converts the three output values into environment
actions. The initial topology has no hidden nodes, so NEAT starts simple and
only adds complexity through mutation.

## Genome Representation

Each genome stores:

- node genes: input, output, bias, add, multiply, activation, and custom nodes
- connection genes: from-node, to-node, weight, enabled flag, and innovation ID
- input, output, and bias node IDs
- counters for the next node ID and next innovation ID

Innovation IDs are used to align matching connection genes during crossover.
When two parents have the same innovation ID, the child can inherit that
connection's weight/enabled state from either parent.

## Fitness Evaluation

Each genome is evaluated by running one or more SlimeVolley episodes:

1. Reset the environment.
2. Convert each observation to a network action.
3. Step the environment until the episode ends.
4. Sum rewards and optional shaping terms.
5. Average across episodes.

The environment reward is from the perspective of the right-side trained agent.
Positive reward means the agent scored points; negative reward means the
opponent scored points.

## Complexity Penalty

A connection-count complexity penalty was added to discourage unnecessarily large
networks:

```text
penalty_scale = sqrt(connection_count * gene_penalty + 1)
```

For positive reward, fitness is divided by this scale. For negative reward,
fitness is multiplied by this scale, making large losing networks worse. During
experimentation, this penalty was reduced/disabled because larger hidden-node
architectures were being removed before they had enough time to improve.

## Survival Bonus

A survival bonus was added as a training-only shaping signal:

```text
fitness = game reward + survival_bonus_per_step * timesteps_survived
```

This helped the agent avoid immediate losses and improved point ratios in some
runs. However, it can dominate the real game reward if it is too large. For
example, fitness values above the normal SlimeVolley score range (`-5` to `+5`)
indicate that survival bonus is strongly influencing selection. This can create
agents that stall or survive until the time limit instead of learning to win.

To reduce this risk, later experiments added more task-specific shaping based on
ball interaction rather than relying only on survival time.

## Ball-Interaction Shaping

Additional training-only shaping was added to encourage active volleyball
behavior:

- reward likely ball contact when the ball is close to the agent and its velocity changes
- reward moving closer to the ball when the ball is on the agent's side
- penalize conceding a point when the ball was behind the agent

This uses the state observation rather than pixels. In SlimeVolley, the
observation contains agent position/velocity and ball position/velocity from the
agent's perspective. This shaping is still heuristic because the environment
does not directly expose a "ball touched" event, so contact is inferred from
ball distance and velocity change.

Survival bonus also increases training wall-clock time. As agents learn to
survive longer, each episode contains more environment steps, so each generation
takes longer to evaluate. This makes a high survival bonus costly in two ways:
it can reward passive behavior and it can slow experimentation. A very small
bonus, or no survival bonus, is safer once the agent can already avoid immediate
losses.

## Net-Camping Penalty

The agent sometimes learned to camp near the net. A small training-only penalty
was added when:

- the agent is near the net, and
- the ball is behind the agent

This discourages the degenerate strategy where the player stays at the net while
the ball falls behind it. The penalty should be small because useful net play
should not be completely banned.

In later experiments, the original net-camping penalty was too small compared
with the survival bonus. A useful direction is to reduce survival reward and use
moderate position shaping, for example:

```text
survival_bonus_per_step = 0.0 to 0.0002
net_camping_penalty     = about 0.003
ball_tracking_penalty   = about 0.001
action_shaping_scale    = about 0.001
```

These values should be treated as experimental ranges, not final constants.

## Action Conversion

The three network outputs are converted to SlimeVolley actions. An exclusive
horizontal action mode was added so the agent does not press forward and
backward simultaneously. This made movement easier to interpret and reduced
jitter.

## Speciation Improvements

The first implementation clustered genomes into species but still selected
offspring mostly by the strongest current fitness. This meant hidden-node
architectures were created but often removed quickly.

The reproduction loop was changed so that:

- one strong representative from each surviving species is preserved when possible
- each surviving species receives an offspring budget
- extra offspring are allocated by shifted species fitness
- parents are selected within the same species

This makes speciation more meaningful: new architectures can survive long enough
for their weights to adapt.

## Stagnation Reset

A stagnation reset replaces the worst fraction of the population after many
generations without improvement. This helps escape local optima. In early runs,
resetting too often removed promising hidden-node species before they improved.
Increasing the stagnation window and lowering the reset fraction gave complex
architectures more time to develop.

## Parallel Evaluation

Fitness evaluation is parallelized with multiple worker processes. This is
useful because evaluating genomes in SlimeVolley is CPU-bound Python work. GPU
acceleration is not useful for the current implementation because the bottleneck
is environment stepping and small dynamic network evaluation, not large matrix
operations.

On Google Compute Engine, CPU workers were increased to match the machine size.
For example, a `c2-standard-30` VM exposes 30 logical CPUs and 15 physical
cores. Worker counts around 24-30 are reasonable starting points. Larger worker
counts, such as 36-60, are allowed by `ProcessPoolExecutor`, but may become
slower if the workload is CPU-bound because of context switching and process
coordination overhead. Generation time is a better metric than apparent CPU
percentage.

During GCE tests, process inspection confirmed that parallel evaluation was
active: one parent Python process spawned many worker Python processes, each
using high CPU. The earlier `e2-standard-8` VM was slower than the local Mac
because it had weaker CPUs and only 4 physical cores. The `c2-standard-30` VM
was substantially faster.

## GCE Roundtrip Workflow

Training was moved to Google Compute Engine using a roundtrip script:

1. upload current local source code to the VM
2. run training on the VM
3. generate `saved/best_gameplay.mp4` from `saved/best_genome.json`
4. download the VM `saved/` directory
5. replace the local `saved/` directory

The local `saved/` directory is only replaced after training, video generation,
and download succeed. If training is stopped midway, the current best genome may
exist on the VM but is not automatically copied back. In that case, the current
VM snapshot can be preserved by manually generating the video on the VM and
copying `/home/shusato/neat-slime/saved` back to the local project.

The script originally backed up local saved files, but backups were removed to
keep the workflow simple once results were being copied intentionally.

Real-time logs can be watched either directly from the roundtrip terminal or
with:

```bash
gcloud compute ssh gce-shu0 \
  --project gcp-samples-ic0 \
  --zone asia-northeast1-c \
  --command 'tail -f "$(ls -t /home/shusato/neat-slime/logs/train-*.log | head -n 1)"'
```

## Raw Evaluation Metrics

Training fitness may include shaping terms, so separate raw evaluation metrics
were added:

- `raw_points=a/b`: the agent scored `a` points out of `b` total point events
- `average raw reward`: average true point difference per episode
- `wins`: number of full episodes won
- `draws`: number of full episodes with zero net score
- `losses`: number of full episodes lost
- `average episode length`: average number of environment steps per episode

`raw_points` is useful early in training because full match wins are rare. For
example:

```text
raw_points=1/51
```

means the agent scored 1 point while the opponent scored 50 points. Later:

```text
raw_points=8/23
```

means the agent scored 8 points while the opponent scored 15 points. This is a
better point ratio, but it does not necessarily mean the agent won matches.

Average episode length helps distinguish two different cases:

- short episodes with many opponent points mean the agent loses quickly
- long episodes with few points may mean the agent survives or stalls until the
  time limit

Therefore, raw points should be interpreted together with wins/draws/losses,
average raw reward, and average episode length.

## Time-Limit Interpretation

SlimeVolley episodes end when one side loses 5 lives or when the environment
reaches 3000 timesteps. Therefore, low total raw point counts can indicate that
many episodes reached the time limit before either side scored 5 points.

For example, if 10 raw evaluation episodes normally allow up to about 50 opponent
points in complete losses, but the log shows:

```text
raw_points=8/23
```

then many episodes likely lasted until the time limit. This suggests the agent
survived longer and scored a better ratio of points, but this should be checked
with full win/draw/loss stats.

## Saving Best Genome

Two saving strategies were tested:

1. Save by training fitness.
2. Save by confirmed raw baseline reward.

Saving by raw reward sometimes selected boring policies that lost by fewer
points but looked worse behaviorally, such as net camping. The default was
changed back to saving by training fitness while keeping raw evaluation for
monitoring.

## Observed Progress

Early runs often had no hidden nodes in the best genome and produced almost no
points against the baseline opponent. Later runs showed:

- hidden nodes appearing in current generation best genomes
- improved overall training fitness
- improved raw point ratio
- fewer immediate 5-0 losses

However, high survival bonus can make the score look much better than true game
performance. A model with high shaped fitness should still be evaluated using:

```bash
PYTHONPATH=.:./slimevolleygym .venv/bin/python -m src.evaluate_saved --episodes 100
```

## Remaining Limitations

The current implementation is still a simplified NEAT system:

- no backpropagation
- no recurrent connections
- dynamic Python dict/list genomes, not vectorized JAX arrays
- environment stepping is CPU-bound
- raw game reward is sparse
- hidden-node structures may require many generations to become useful

The most important future improvement would be better fitness shaping or a
curriculum that rewards useful ball interaction without over-rewarding passive
survival.

## Offensive Ball Movement Shaping

The latest shaping change avoids direct rewards for simple ball contact or
moving toward the ball, because those rewards encouraged the agent to keep the
ball on its head. Training now includes targeted offensive signals:

- reward when the ball crosses from the agent's side to the opponent's side
- small reward when a likely hit sends the ball leftward toward the opponent
- small penalty after the ball stays on the agent's side for too many steps

The goal is to make long survival useful only when it also creates pressure on
the opponent side.
