"""Example usage of LitReviewBot

This script demonstrates how to use LitReviewBot to:
1. Create a paper collection
2. Add papers with metadata
3. Cluster papers by theme
4. Identify research gaps
5. Generate a comprehensive literature review
"""

from lit_review_bot import LitReviewBot, PaperCollection, LiteratureReview


def main():
    # Initialize the bot
    bot = LitReviewBot()

    print("=" * 80)
    print("LitReviewBot Example Usage")
    print("=" * 80)
    print()

    # Step 1: Add papers to a collection
    print("Step 1: Adding papers to collection...")
    print("-" * 80)

    # Add transformer paper
    paper1 = bot.add_paper(
        title="Attention Is All You Need",
        authors=["Vaswani, A.", "Shazeer, N.", "Parmar, N.", "Uszkoreit, J.", "Jones, L."],
        year=2017,
        venue="NeurIPS",
        abstract=("The dominant sequence transduction models are based on complex recurrent or "
                 "convolutional neural networks. We propose a new simple network architecture, "
                 "the Transformer, based solely on attention mechanisms. Our results show that "
                 "the Transformer achieves state-of-the-art performance on machine translation "
                 "tasks while being more parallelizable."),
        keywords=["transformer", "attention", "neural networks", "machine translation", "NLP"],
        citations=50000,
        doi="10.48550/arXiv.1706.03762",
        collection_id=1
    )
    print(f"✓ Added: {paper1.title} ({paper1.year})")

    # Add AlphaFold paper
    paper2 = bot.add_paper(
        title="Highly accurate protein structure prediction with AlphaFold",
        authors=["Jumper, J.", "Evans, R.", "Pritzel, A.", "Green, T."],
        year=2021,
        venue="Nature",
        abstract=("Proteins are essential to life, and understanding their structure enables us "
                 "to understand their function. We demonstrate that deep learning can predict "
                 "protein structure with atomic accuracy. Our method, AlphaFold, achieved "
                 "unprecedented accuracy in CASP14. These results show that deep learning can "
                 "solve the protein folding problem."),
        keywords=["protein folding", "deep learning", "AlphaFold", "structure prediction", "biology"],
        citations=15000,
        doi="10.1038/s41586-021-03819-2",
        collection_id=1
    )
    print(f"✓ Added: {paper2.title} ({paper2.year})")

    # Add GPT-3 paper
    paper3 = bot.add_paper(
        title="Language Models are Few-Shot Learners",
        authors=["Brown, T.", "Mann, B.", "Ryder, N.", "Subbiah, M."],
        year=2020,
        venue="NeurIPS",
        abstract=("Recent work has demonstrated that scaling up language models greatly improves "
                 "task-agnostic, few-shot performance. We trained GPT-3, a 175 billion parameter "
                 "autoregressive language model, and found that it can perform many NLP tasks "
                 "with no gradient updates or fine-tuning. Our results showed that scaling laws "
                 "continue to hold at unprecedented model sizes."),
        keywords=["language models", "GPT-3", "few-shot learning", "scaling laws", "NLP"],
        citations=25000,
        doi="10.48550/arXiv.2005.14165",
        collection_id=1
    )
    print(f"✓ Added: {paper3.title} ({paper3.year})")

    # Add deep learning survey
    paper4 = bot.add_paper(
        title="Deep Learning",
        authors=["LeCun, Y.", "Bengio, Y.", "Hinton, G."],
        year=2015,
        venue="Nature",
        abstract=("Deep learning allows computational models that are composed of multiple "
                 "processing layers to learn representations of data with multiple levels of "
                 "abstraction. This survey covers the major developments in deep learning, "
                 "including convolutional networks, recurrent networks, and their applications. "
                 "However, limitations remain in understanding why these methods work so well."),
        keywords=["deep learning", "neural networks", "survey", "convolutional networks", "machine learning"],
        citations=35000,
        doi="10.1038/nature14539",
        collection_id=1
    )
    print(f"✓ Added: {paper4.title} ({paper4.year})")

    # Add ResNet paper
    paper5 = bot.add_paper(
        title="Deep Residual Learning for Image Recognition",
        authors=["He, K.", "Zhang, X.", "Ren, S.", "Sun, J."],
        year=2016,
        venue="CVPR",
        abstract=("Deeper neural networks are more difficult to train. We present residual "
                 "learning to ease the training of networks that are substantially deeper. "
                 "Our experimental results showed that residual networks are easier to optimize "
                 "and gain accuracy from increased depth. We observed that very deep networks "
                 "achieved state-of-the-art performance on ImageNet."),
        keywords=["residual networks", "ResNet", "image recognition", "computer vision", "deep learning"],
        citations=40000,
        doi="10.1109/CVPR.2016.90",
        collection_id=1
    )
    print(f"✓ Added: {paper5.title} ({paper5.year})")

    print(f"\n✓ Total papers added: 5")
    print()

    # Step 2: Cluster papers by theme
    print("Step 2: Clustering papers by theme...")
    print("-" * 80)

    clusters = bot.cluster_papers(collection_id=1, num_clusters=3)

    for i, cluster in enumerate(clusters, 1):
        print(f"\nCluster {i}: {cluster.theme} ({len(cluster.papers)} papers)")
        print(f"  Representative keywords: {', '.join(cluster.representative_keywords[:5])}")
        for paper_id in cluster.papers[:3]:  # Show first 3 papers
            collection = bot.get_collection(1)
            paper = next((p for p in collection.papers if p.paper_id == paper_id), None)
            if paper:
                print(f"  - {paper.title} ({paper.year})")

    print()

    # Step 3: Identify research gaps
    print("Step 3: Identifying research gaps...")
    print("-" * 80)

    gaps = bot.identify_gaps(collection_id=1)

    for gap in gaps:
        print(f"\n{gap.gap_type.upper()} GAP")
        print(f"  Description: {gap.description}")
        print(f"  Importance: {gap.importance_score:.1f}/10")
        print(f"  Suggestion: {gap.suggested_direction}")

    print()

    # Step 4: Generate literature review (narrative style)
    print("Step 4: Generating literature review (narrative style)...")
    print("-" * 80)

    review = bot.generate_review(collection_id=1, style="narrative")

    print(f"\n{review.title}")
    print(f"Collection ID: {review.collection_id} ({review.num_papers} papers)")
    print(f"Style: {review.style}")
    print(f"Generated: {review.generated_date}")
    print(f"Word Count: {review.word_count}")
    print()
    print("=" * 80)
    print(review.content[:1000] + "..." if len(review.content) > 1000 else review.content)
    print("=" * 80)
    print()

    # Step 5: List all collections
    print("Step 5: Listing all collections...")
    print("-" * 80)

    collections = bot.list_collections()

    for collection in collections:
        print(f"\nCollection {collection.collection_id}: {collection.name}")
        print(f"  Papers: {len(collection.papers)}")
        print(f"  Created: {collection.created_date}")
        if collection.papers:
            years = [p.year for p in collection.papers]
            print(f"  Year range: {min(years)}-{max(years)}")
            print(f"  Total citations: {sum(p.citations for p in collection.papers):,}")

    print()
    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Try adding more papers to the collection")
    print("2. Experiment with different clustering numbers")
    print("3. Generate reviews in different styles (systematic, meta-analysis)")
    print("4. Use the CLI for interactive exploration")
    print()


if __name__ == "__main__":
    main()
