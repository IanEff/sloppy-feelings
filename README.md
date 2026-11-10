# sloppy-feelings

![FEELINGS: a jeweled python coiled around a retro computer in a starfield](.github/banner.jpg)

## Run it

You need [uv](https://docs.astral.sh/uv/). It fetches Python 3.15 and every dependency itself.

```
git clone https://github.com/IanEff/sloppy-feelings.git
cd sloppy-feelings
uv run python main.py
```

The first run downloads two models from Hugging Face and two lexicons into `data/lexicons/`.
With those cached, the whole script takes about a minute on an M-series Mac. It writes
`out/graph.png` and `out/graph.html`; everything else lives in variables, so for poking at
`stanzas` or `system`, open `main.py` as a notebook (each `# %%` is a cell) in VS Code, Zed or
Jupyter with jupytext.

## What it computes

Everything `main.py` extracts from the 207 poems in `data/poems/`, or adds on top of them.
Listed in the order the code computes it.

### Per line: `lines`

| Column | What it holds |
|---|---|
| `line` | The line's text, whitespace intact. |
| `blank_before` | Count of blank or whitespace-only lines above it. |
| `poem` | The file name without `.txt`. |
| `stanza` | Stanza number within the poem. A blank line starts a new stanza; in a double-spaced poem (more than 90% of lines have a blank above), it takes a run of two. |
| `n_gaps` | Internal gaps: runs of 3+ spaces, or any tab, inside the stripped line. |

### Per stanza: `stanzas`

| Column | Kind | What it holds |
|---|---|---|
| `poem`, `stanza` | key | One row per stanza, sorted by poem then stanza. |
| `text` | text | The stanza's lines joined with spaces. |
| `n_lines` | form | Line count. Also the stanza's mass in both simulations. |
| `n_gaps` | form | Sum of the lines' `n_gaps`. |
| `lemmas` | lemma | Set of lowercased noun and verb lemmas, stop words dropped (spaCy `en_core_web_sm`). |
| `book` | metadata | `source` from `SOURCES.tsv`, with ` (Lauren Ireland)` stripped. Poems sourced from a URL get `uncollected`. |
| `date`, `year`, `place` | metadata | Parsed from dated file names like `april-30-2013-seattle` or `october-10-2008-queens-1`. Empty for undated poems. |
| `is_ritual` | metadata | File name starts with `ritual-for`. |
| `n_stanzas` | metadata | Stanzas in the poem. |
| `concreteness` | lexicon | Mean concreteness, 1 (abstract) to 5 (concrete), from Brysbaert et al. (2014). |
| `valence`, `arousal`, `dominance` | lexicon | Mean of each, 1 to 9, from Warriner, Kuperman & Brysbaert (2013). |
| `pron_i`, `pron_you`, `pron_we` | lexicon | Share of the stanza's words that are first-person singular, second-person, or first-person plural pronouns. |
| `novelty` | embedding | Cosine distance from the previous stanza of the same poem. Empty for each poem's first stanza. |
| `emotion` | emotion | The top label of the 28 GoEmotions labels, from `SamLowe/roberta-base-go_emotions` (RoBERTa trained on Reddit comments). |
| `emo_<label>` | emotion | That model's score for each of the 28 labels, one column per label. |
| `topic`, `topic_name` | topic | BERTopic over the stanza embeddings, clustered by KMeans into `n_topics = 12`. |
| `x`, `y` | layout | t-SNE position over `distance`. |
| `fold_x`, `fold_y` | dynamics | Where the stanza settled when its poem was simulated alone (`evolve_poem`). |
| `system_x`, `system_y` | dynamics | Where the stanza settled with every poem in one box (`evolve_system`). |

The lexicon columns average over the stanza's alphabetic, non-stop words found in the lexicon,
looking each up by lemma first, then by the word itself. The lexicons download into
`data/lexicons/` on the first run.

### Stanza by stanza: the matrices

| Name | Shape | What it holds |
|---|---|---|
| `poem_vectors` | stanzas × embedding | Unit-normed `all-MiniLM-L6-v2` sentence embedding of each stanza's `text`. |
| `similarity` | stanzas × stanzas | Cosine similarity of `poem_vectors`. |
| `overlap` | stanzas × stanzas | Jaccard overlap of `lemmas`. |
| `distance` | stanzas × stanzas | `(1 - similarity) + alpha * overlap` with `alpha = 2.0`, symmetrized, zero diagonal. Input to t-SNE. |
| `G` | graph | Each stanza joined to its `k = 3` most similar stanzas, dropping pairs below `floor = 0.2`. Edges weighted by similarity. |

### The simulations

Each poem is a molecule and each stanza a particle. Springs join stanza pairs, stiffness from
`similarity` (negative clipped to zero), rest length `1 - similarity`, mass `n_lines`, friction
`frictionCoeff = 0.5`, step `dt = 0.05`. The setup follows Klyshko, Example 3.2 and section 5.
`evolve_system` differs: its in-poem springs keep negative similarity as a push, a backbone bond
(`ks_bond = 1`, rest length `novelty`) joins consecutive stanzas, every pair attracts by gravity
`g m_i m_j / r^2` (`g = 1e-3`, softened at `sigma`), and friction is light (`frictionCoeff = 0.02`)
so the system is still moving at the end.

| Name | Shape | What it holds |
|---|---|---|
| `trajectories[poem]` | (steps + 1) × stanzas in poem × 2 | `evolve_poem`: one poem alone for `T = 20`, starting as a jittered straight chain. |
| `system` | (steps + 1) × all stanzas × 2 | `evolve_system`: every poem together for `T = 150`, each starting as a chain at a random spot and angle in a 60 × 60 box. In-poem pairs get full springs. Cross-poem edges of `G` get springs at `ks_inter = 0.5` of their weight. Any two stanzas closer than `sigma = 0.3` push apart with stiffness `ks_rep = 20`. |

### What could drive forces

The simulations use only `n_lines` (mass) and `similarity` (springs) so far. These are the other
scalars that could become physical properties. The roles are suggestions, not implemented.

Per stanza, as particle properties:

| Column | Range | Possible role |
|---|---|---|
| `n_lines` | count ≥ 1 | Mass. Already used. |
| `n_gaps` | count ≥ 0 | Radius, i.e. a per-stanza `sigma`: whitespace takes up room. |
| `valence` | 1 to 9 | Signed charge, `valence - 5`. The only clearly two-sided scale here. Like charges repel under Coulomb, so use `-q_i q_j` if moods should clump. |
| `arousal` | 1 to 9 | Temperature: strength of a random kick each step. |
| `dominance` | 1 to 9 | Friction, or anchoring: resistance to being pushed. |
| `concreteness` | 1 to 5 | Friction, or a mass multiplier: concrete is heavy, abstract drifts. |
| `pron_i`, `pron_you`, `pron_we` | 0 to 1 | Affinities, e.g. attraction proportional to `pron_i[i] * pron_you[j]`. |
| `emo_<label>` | 0 to 1 | Each is a scalar. Together the 28 are a vector, whose cosine between stanzas could be a second spring stiffness. |
| `stanza` | 0, 1, 2, ... | Backbone bonds between consecutive stanzas. Today every in-poem pair is bonded alike. |
| `year` | 2008, ... | An external field along one axis, or springs between poems close in time. |
| `n_stanzas` | count | A per-poem (molecule) property. |

Per pair, as spring stiffness or rest length:

| Name | Possible role |
|---|---|
| `similarity` | Stiffness and rest length. Already used. |
| `overlap` | A second spring term. Used only in `distance`, for t-SNE. |
| `novelty` | Already a property of the bond from the previous stanza: its rest length, or its looseness. |
| `G` weights | Cross-poem springs. Already used. |

Not scalars: `text`, `lemmas`, `book`, `place`, `emotion`, `topic`, `topic_name` and `is_ritual`
are labels, sets or flags. They could mark species, with same-species pairs attracting, but
their values aren't magnitudes: topic 7 isn't more than topic 3. `x`, `y`, `fold_*` and
`system_*` are outputs, so feeding them back in would be circular.

Before using any of these:

- Fill the gaps. The lexicon columns are empty for stanzas with no lexicon hits, `novelty` for
  each poem's first stanza, and `year` for undated poems.
- Normalize. Counts, shares, and 1-to-5 and 1-to-9 scales differ widely; z-score or map to
  0 to 1 so no one property swamps the rest.
- Keep mass, radius, friction and temperature positive.
- Expect noise. A short stanza's lexicon mean may rest on two or three words.
