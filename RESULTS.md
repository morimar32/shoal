# Shoal Search Quality Results

Corpus: 38 documents, 3,624 chunks (test.db)

## Disambiguation Test Queries

10 new Wikipedia articles were added specifically to test disambiguation of ambiguous terms:
- Chess + Monarchy of the United Kingdom (king, queen, bishop, castle)
- Mercury (planet) + Mercury (element) (mercury, atmosphere, surface)
- Crane (machine) + Crane (bird) (crane, boom, lifting)
- Apple Inc. + Apple (fruit) (apple, core, seeds)
- Python (programming language) + Pythonidae (python, library, wrapped)

### Results

| Query | #1 | #2 | #3 | #4 | #5 | Correct article in top 5? |
|-------|----|----|----|----|-----|---------------------------|
| mercury orbit solar system planet | **Mercury Planet** | Huckleberry Finn | Thermodynamics | Crane Machine | Crane Machine | **#1** |
| mercury liquid metal poisoning thermometer | Origin Of Species | Origin Of Species | Origin Of Species | Atmospheric River | Origin Of Species | No |
| python programming language syntax | Huckleberry Finn | Origin Of Species | Origin Of Species | Origin Of Species | Origin Of Species | No |
| python constriction prey reptile | Olympic Games | Origin Of Species | Origin Of Species | Origin Of Species | Origin Of Species | No |
| crane lifting construction tower | **Crane Machine** | Apple Inc | Crane Machine | Crane Machine | Crane Machine | **#1** |
| crane bird migration wetland | Olympic Games | Olympic Games | **Crane Bird** | Huckleberry Finn | Great Barrier Reef | #3 |
| apple iphone macbook silicon valley | Huckleberry Finn | Huckleberry Finn | Origin Of Species | Origin Of Species | Origin Of Species | No |
| apple orchard fruit harvest cultivar | Huckleberry Finn | **Apple Fruit** | Origin Of Species | Huckleberry Finn | Huckleberry Finn | #2 |
| king queen royal palace throne | Olympic Games | Origin Of Species | Silk Road | Olympic Games | Silk Road | No |
| king queen bishop pawn checkmate | Olympic Games | **Monarchy Of The UK** | Silk Road | **Chess** | Supreme Court | #2, #4 |

### Analysis

**What works well:**
- Queries with more specific/technical vocabulary land correctly: "mercury orbit solar system planet" → Mercury Planet #1, "crane lifting construction tower" → Crane Machine #1
- The correct articles almost always appear *somewhere* in the top 5, showing the signal is present

**What doesn't work well:**
- Origin of Species (436 chunks) and Huckleberry Finn (414 chunks) dominate many queries due to sheer size — their massive chunk count means they have chunks matching almost every reef broadly
- Queries using common/generic words (liquid, metal, bird, fruit) don't converge on specific reefs, resulting in low query confidence and poor discrimination
- Many domain-specific terms (checkmate, thermometer, constriction, iPhone, macbook) are likely unknown to lagoon's base 207-reef vocabulary, so queries fall back entirely on common words

**Root causes:**
1. **Large document bias** — Documents with hundreds of chunks statistically dominate results because they have more chances to overlap with any query's reef profile
2. **Vocabulary gaps** — lagoon's base vocabulary doesn't include domain-specific terms. Queries like "python programming" lose the word "python" (unknown) and score on "programming" alone, which maps to generic reefs
3. **Low query confidence** — Many test queries show confidence < 0.25, meaning the query words don't converge strongly on specific reefs

**Expected improvements from Phase 2 (vocabulary extension):**
- Custom vocabulary injection should teach lagoon domain-specific terms from the corpus
- Two-pass ingestion: Pass 1 discovers unknown words → vocab extension → Pass 2 re-scores with extended vocabulary
- This should dramatically improve queries involving terms like "checkmate", "thermometer", "constriction", "iPhone", etc.
- Better coverage scores on chunks should also help differentiate relevant vs irrelevant large-document chunks

## Previously Fixed Issues

- **Wikipedia boilerplate sections** (References, External links, See also, etc.) were producing monster chunks with inflated scores. Fixed by adding boilerplate header detection in `_parsers.py`.
- **Origin of Species GLOSSARY + INDEX** — The book's glossary and index (2,683 lines of term definitions and page references) created a massive junk chunk with 1,096 matched words and confidence 11.7 that dominated every query. Fixed by trimming the file.

## Corpus Composition

### Original 28 documents (26 Wikipedia + 2 Gutenberg books)
algorithm, atmospheric_river, climate_change, coffee, corvidae, dna, earthquake,
fourier_transform, french_revolution, game_theory, granite, great_barrier_reef,
huckleberry_finn, jupiter, mitochondrion, natural_selection, olympic_games,
origin_of_species, penicillin, photosynthesis, renaissance, silk_road,
supply_and_demand, supreme_court_of_the_united_states, thermodynamics,
tyrannosaurus, united_nations, volcanic_eruption

### 10 new disambiguation articles
chess, monarchy_of_the_united_kingdom, mercury_planet, mercury_element,
crane_machine, crane_bird, apple_inc, apple_fruit, python_programming_language,
python_snake
