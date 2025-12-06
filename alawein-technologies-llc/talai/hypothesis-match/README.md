# HypothesisMatch - Scientific Tinder for Researchers

Match researchers based on complementary skills, shared interests, and hypothesis compatibility. Swipe on research hypotheses to find collaborators.

## What It Does

HypothesisMatch helps researchers find collaborators by:

1. **Profile creation** - Skills, interests, research hypotheses
2. **Hypothesis swiping** - SUPPORT, OPPOSE, or DEBATE on research ideas
3. **Smart matching** - Find complementary, similar, or adversarial collaborators
4. **Compatibility scoring** - Quantify collaboration potential

## Installation

```bash
cd hypothesis-match
# No dependencies - pure Python 3.11+
```

## Usage

### Create Your Profile

```bash
python matcher.py create-profile \
  --name "Dr. Alice Smith" \
  --institution "MIT" \
  --skills "machine-learning,computer-vision,statistics" \
  --interests "medical-imaging,autonomous-vehicles" \
  --h-index 25 \
  --looking-for complementary
```

### Add Research Hypotheses

```bash
python matcher.py add-hypothesis \
  --profile-id 1 \
  --text "Deep learning models can surpass human radiologists in cancer detection within 5 years" \
  --field "medical-imaging" \
  --tags "AI,healthcare,diagnosis"
```

### Swipe on Hypotheses

```bash
# Support a hypothesis
python matcher.py swipe --profile-id 1 --hypothesis-id 42 --action SUPPORT

# Oppose a hypothesis
python matcher.py swipe --profile-id 1 --hypothesis-id 43 --action OPPOSE

# Want to debate
python matcher.py swipe --profile-id 1 --hypothesis-id 44 --action DEBATE
```

### Find Matches

```bash
# Find complementary collaborators
python matcher.py match --profile-id 1 --mode complementary --limit 10

# Find similar researchers
python matcher.py match --profile-id 1 --mode similar --limit 5

# Find adversarial debaters
python matcher.py match --profile-id 1 --mode adversarial --limit 5
```

### List All Profiles

```bash
python matcher.py list
```

## Matching Modes

### Complementary (Default)
- Rewards different skill sets
- Values some shared expertise
- Good for interdisciplinary teams
- **Best for**: Building well-rounded teams

### Similar
- Rewards skill overlap
- Finds researchers doing similar work
- **Best for**: Deepening expertise, co-authoring

### Adversarial
- Finds opposing viewpoints
- Productive debate partners
- **Best for**: Challenging assumptions, peer review

## Match Score Factors

**Skills** (up to 40 points)
- Complementary skills: +8 per unique skill
- Shared skills: +3 per overlapping skill (complementary mode)
- Similar skills: +10 per overlap (similar mode)

**Interests** (up to 24 points)
- +12 per shared research interest

**H-index Compatibility** (up to 10 points)
- Similar impact levels: +10
- Moderate difference: +5

**Hypothesis Alignment** (up to 30 points)
- Shared hypotheses: +10 per agreement
- Opposing hypotheses: +15 per disagreement (adversarial mode)

**Total**: 0-100 compatibility score

## Output Example

```
Matches for Dr. Alice Smith (looking for complementary collaborators):
======================================================================

1. Dr. Bob Chen @ Stanford
   Compatibility: 78.5/100
   Collaboration Potential: 82.3/100
   Match Type: complementary

   Shared Interests: medical-imaging, AI

   Why matched:
     - Complementary skills: deep-learning, signal-processing, statistics
     - Shared expertise: machine-learning, computer-vision
     - Similar research impact (h-index: 25 vs 28)
     - 2 shared hypotheses

   Hypothesis Alignment:
     AGREE: Deep learning models can surpass human radiolo...
     DISAGREE: Transfer learning always improves medical im...
```

## Collaboration Potential

Estimates likelihood of successful collaboration (0-100):

- **Base**: 60% of compatibility score
- **Productive debate**: +15 (if 1-3 disagreements)
- **Skill diversity**: +2 per unique combined skill

High potential (>70): Strong match, reach out!
Medium potential (40-70): Worth exploring
Low potential (<40): Probably not a good fit

## Use Cases

1. **Finding co-authors** - Complementary expertise
2. **Building research teams** - Balanced skill sets
3. **Peer review** - Adversarial matches for critical feedback
4. **Conference networking** - Pre-match before attending
5. **PhD advisor search** - Find compatible mentors
6. **Grant proposals** - Assemble strong teams

## Data Storage

- **profiles.json** - All researcher profiles
- **hypotheses.json** - All research hypotheses

Files are created automatically in the current directory.

## Current Limitations

- Simple keyword-based matching (no semantic understanding)
- No network effects (mutual connections)
- No publication history analysis
- No geographic/timezone matching
- No automated profile scraping (manual entry)

## Revenue Model

- **Individual**: $49/month (unlimited matches)
- **Lab/Group**: $199/month (10 users)
- **Institution**: $999/month (university-wide)
- **Freemium**: 5 matches/month free

## Build Info

- Build time: 6 hours
- Credit used: ~$90
- Lines of code: 580
- Status: Functional prototype

## Future Enhancements

- Semantic embedding for better matching
- Publication network analysis
- Automated profile import (Google Scholar, ORCID)
- Chat/messaging integration
- Event/conference matching
- Mutual match notifications
- Reputation system (successful collaborations)
- NLP for hypothesis similarity
